from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config import config


class LLMClient:

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    # ── Embeddings ────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch embed — one API call for all chunks in a transcript."""
        response = self.client.embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=texts,
        )
        # API returns embeddings in same order as input
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    # ── Chat ──────────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def chat(
        self,
        prompt: str,
        system: str = "You are a helpful sales intelligence assistant.",
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        kwargs = dict(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            temperature=temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


# Singleton
llm_client = LLMClient()