import uuid
import json
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import config
from ingestion.parser import ParsedTranscript, Turn


@dataclass
class Chunk:
    """A chunk ready for embedding + storage."""
    chunk_id:     str
    call_id:      str
    speaker:      str
    speaker_role: str         # ae | prospect | se | ciso | cs | other
    start_time:   str
    end_time:     Optional[str]
    text:         str

    @property
    def citation(self) -> str:
        return f"[{self.call_id} | {self.start_time} | {self.speaker}]"

    def to_embedding_text(self) -> str:
        """Text sent to embedder — enriched with metadata for better retrieval."""
        return (
            f"Call: {self.call_id}\n"
            f"Speaker: {self.speaker} (Role: {self.speaker_role})\n"
            f"Time: {self.start_time}\n"
            f"Content: {self.text}"
        )


@dataclass
class ChunkedTranscript:
    call_id:      str
    chunks:       list[Chunk] = field(default_factory=list)
    speaker_roles: dict[str, str] = field(default_factory=dict)  # speaker → role


class Chunker:
    """
    Converts ParsedTranscript → list[Chunk].

    Strategy:
    1. Use LLM to classify speaker_role for every unique participant (once per call)
    2. Merge consecutive same-speaker turns up to MAX_CHUNK_TURNS or MAX_CHUNK_TOKENS
    3. Never merge across different speakers (preserves dialogue integrity)
    """

    VALID_ROLES = {"ae", "prospect", "se", "ciso", "cs", "pricing", "legal", "procurement", "other"}

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def chunk(self, transcript: ParsedTranscript) -> ChunkedTranscript:
        # Step 1: LLM classifies all speakers at once (single API call)
        speaker_roles = self._classify_speakers(
            call_id=transcript.call_id,
            participants=transcript.participants,
            sample_turns=self._get_sample_turns(transcript.turns),
        )

        # Step 2: Merge + chunk turns
        chunks = self._build_chunks(
            turns=transcript.turns,
            call_id=transcript.call_id,
            speaker_roles=speaker_roles,
        )

        return ChunkedTranscript(
            call_id=transcript.call_id,
            chunks=chunks,
            speaker_roles=speaker_roles,
        )

    # ── Speaker Role Classification ───────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _classify_speakers(
        self,
        call_id: str,
        participants: list[str],
        sample_turns: dict[str, str],   # speaker → sample text
    ) -> dict[str, str]:
        """
        Single LLM call to classify ALL speakers in the call.
        Returns: {"Jordan (AE)": "ae", "Priya (RevOps Director)": "prospect", ...}
        """

        participants_block = "\n".join(
            f"- {p}: \"{sample_turns.get(p, 'no sample available')}\""
            for p in participants
        )

        prompt = f"""You are analyzing a B2B sales call transcript to classify each participant's role.

Call ID: {call_id}

Participants and a sample of what they said:
{participants_block}

Classify each participant into exactly one of these roles:
- ae         → Account Executive / Sales Rep (selling side)
- prospect   → Buyer / Customer / Lead (buying side: RevOps, Finance, Procurement, Legal, Security from buyer)
- se         → Sales Engineer / Technical Pre-Sales (selling side)
- ciso       → CISO / Security Lead (selling side)
- cs         → Customer Success / Onboarding (selling side)
- pricing    → Pricing Strategist / Deal Desk (selling side)
- legal      → Legal Counsel (selling side)
- procurement→ Procurement (buying side)
- other      → Anyone else

Return ONLY a valid JSON object. No explanation. No markdown.
Format: {{"speaker_name": "role", ...}}

Example output: {{"Jordan (AE)": "ae", "Priya (RevOps Director)": "prospect"}}"""

        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,    # deterministic classification
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        classified = json.loads(raw)

        # Validate + fallback unknown roles
        return {
            speaker: role if role in self.VALID_ROLES else "other"
            for speaker, role in classified.items()
        }

    def _get_sample_turns(self, turns: list[Turn], n: int = 2) -> dict[str, str]:
        """Get first N turns per speaker as classification sample."""
        samples: dict[str, list[str]] = {}
        for turn in turns:
            if turn.speaker not in samples:
                samples[turn.speaker] = []
            if len(samples[turn.speaker]) < n:
                samples[turn.speaker].append(turn.text[:150])  # first 150 chars

        return {speaker: " | ".join(texts) for speaker, texts in samples.items()}

    # ── Chunking ──────────────────────────────────────────────────────────────

    def _build_chunks(
        self,
        turns: list[Turn],
        call_id: str,
        speaker_roles: dict[str, str],
    ) -> list[Chunk]:
        """
        Merge consecutive same-speaker turns into chunks.
        Respects MAX_CHUNK_TURNS and MAX_CHUNK_TOKENS limits.
        Never merges across different speakers.
        """
        chunks = []
        if not turns:
            return chunks

        # Seed with first turn
        buffer_turns  = [turns[0]]
        buffer_tokens = self._estimate_tokens(turns[0].text)

        for turn in turns[1:]:
            same_speaker  = turn.speaker == buffer_turns[0].speaker
            within_limits = (
                len(buffer_turns) < config.MAX_CHUNK_TURNS and
                buffer_tokens + self._estimate_tokens(turn.text) < config.MAX_CHUNK_TOKENS
            )

            if same_speaker and within_limits:
                buffer_turns.append(turn)
                buffer_tokens += self._estimate_tokens(turn.text)
            else:
                # Flush buffer → create chunk
                chunks.append(self._turns_to_chunk(buffer_turns, call_id, speaker_roles))
                buffer_turns  = [turn]
                buffer_tokens = self._estimate_tokens(turn.text)

        # Flush last buffer
        if buffer_turns:
            chunks.append(self._turns_to_chunk(buffer_turns, call_id, speaker_roles))

        return chunks

    def _turns_to_chunk(
        self,
        turns: list[Turn],
        call_id: str,
        speaker_roles: dict[str, str],
    ) -> Chunk:
        speaker = turns[0].speaker
        return Chunk(
            chunk_id     = str(uuid.uuid4()),
            call_id      = call_id,
            speaker      = speaker,
            speaker_role = speaker_roles.get(speaker, "other"),
            start_time   = turns[0].start_time,
            end_time     = turns[-1].end_time,
            text         = " ".join(t.text for t in turns),
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: 1 token ≈ 4 chars."""
        return len(text) // 4