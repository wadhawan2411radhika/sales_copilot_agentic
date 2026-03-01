import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    """A single speaker turn parsed from the transcript."""
    speaker:     str
    start_time:  str          # "MM:SS"
    end_time:    Optional[str]  # filled in after parsing full transcript
    text:        str
    raw_line:    str


@dataclass
class ParsedTranscript:
    """Full parsed transcript — output of parser."""
    call_id:       str
    filename:      str
    raw_text:      str
    turns:         list[Turn] = field(default_factory=list)
    participants:  list[str]  = field(default_factory=list)  # unique speakers
    duration_sec:  Optional[int] = None


class TranscriptParser:
    """
    Parses transcript files with format:
        [MM:SS] Speaker Name (Role): dialogue text
    
    Handles:
    - Multi-line turns
    - Multiple speakers per call
    - Duration extraction from last timestamp
    """

    # Matches: [00:23] AE (Jordan):  text...
    #      or: [01:05] SE (Luis):    text...
    #      or: [00:05] Prospect (Priya – RevOps Director): text...
    LINE_PATTERN = re.compile(
        r"^\[(\d{2}:\d{2})\]\s+"   # [MM:SS]
        r"([^\:]+?)"               # speaker name (non-greedy)
        r":\s+"                    # colon separator
        r"(.+)$"                   # dialogue text
    )

    def parse(self, raw_text: str, call_id: str, filename: str) -> ParsedTranscript:
        transcript = ParsedTranscript(
            call_id=call_id,
            filename=filename,
            raw_text=raw_text,
        )

        turns = self._extract_turns(raw_text)
        turns = self._fill_end_times(turns)

        transcript.turns        = turns
        transcript.participants = self._extract_participants(turns)
        transcript.duration_sec = self._extract_duration(turns)

        return transcript

    def _extract_turns(self, raw_text: str) -> list[Turn]:
        turns = []
        current_turn: Optional[Turn] = None

        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue

            match = self.LINE_PATTERN.match(line)

            if match:
                # Save previous turn
                if current_turn:
                    turns.append(current_turn)

                timestamp, speaker, text = match.groups()
                speaker = speaker.strip()
                text    = text.strip()

                current_turn = Turn(
                    speaker    = speaker,
                    start_time = timestamp,
                    end_time   = None,
                    text       = text,
                    raw_line   = line,
                )

            else:
                # Continuation line — append to current turn text
                if current_turn and line:
                    current_turn.text += " " + line

        # Don't forget the last turn
        if current_turn:
            turns.append(current_turn)

        return turns

    def _fill_end_times(self, turns: list[Turn]) -> list[Turn]:
        """Set end_time of turn[i] = start_time of turn[i+1]."""
        for i in range(len(turns) - 1):
            turns[i].end_time = turns[i + 1].start_time

        # Last turn has no successor
        if turns:
            turns[-1].end_time = turns[-1].start_time

        return turns

    def _extract_participants(self, turns: list[Turn]) -> list[str]:
        """Unique speakers in order of first appearance."""
        seen = []
        for turn in turns:
            if turn.speaker not in seen:
                seen.append(turn.speaker)
        return seen

    def _extract_duration(self, turns: list[Turn]) -> Optional[int]:
        """Duration in seconds from last timestamp."""
        if not turns:
            return None
        last_ts = turns[-1].start_time   # "MM:SS"
        try:
            mm, ss = last_ts.split(":")
            return int(mm) * 60 + int(ss)
        except ValueError:
            return None


def timestamp_to_seconds(ts: str) -> int:
    mm, ss = ts.split(":")
    return int(mm) * 60 + int(ss)