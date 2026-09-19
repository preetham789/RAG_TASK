from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Protocol

from app.embeddings import TOKEN_RE
from app.models import ChunkHit


UNKNOWN_ANSWER = "I don't know from the provided documents."


class GenerationError(RuntimeError):
    """Raised when the answer generator fails."""


class AnswerGenerator(Protocol):
    provider_name: str
    model_name: str

    def answer(self, question: str, chunks: list[ChunkHit]) -> str:
        ...


@dataclass
class OpenAIAnswerGenerator:
    model_name: str = "gpt-4.1-mini"
    provider_name: str = "openai"

    def __post_init__(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise GenerationError("OPENAI_API_KEY is required for OpenAI generation.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise GenerationError("OpenAI generation requires the openai package.") from exc
        self._client = OpenAI()

    def answer(self, question: str, chunks: list[ChunkHit]) -> str:
        prompt = build_grounded_prompt(question, chunks)
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer only from the supplied document chunks. "
                            f"If the chunks do not contain the answer, say: {UNKNOWN_ANSWER}"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:
            raise GenerationError(
                f"OpenAI answer call failed after {time.perf_counter() - start:.2f}s."
            ) from exc
        content = response.choices[0].message.content or ""
        return content.strip() or UNKNOWN_ANSWER


@dataclass
class ExtractiveAnswerGenerator:
    provider_name: str = "extractive"
    model_name: str = "token-overlap-sentences"

    def answer(self, question: str, chunks: list[ChunkHit]) -> str:
        question_tokens = {
            token
            for token in TOKEN_RE.findall(question.lower())
            if len(token) > 2 and token not in STOPWORDS
        }
        if not question_tokens:
            return UNKNOWN_ANSWER

        best_sentence = ""
        best_score = 0
        for chunk in chunks:
            for sentence in split_sentences(chunk.text):
                sentence_tokens = {
                    token
                    for token in TOKEN_RE.findall(sentence.lower())
                    if len(token) > 2 and token not in STOPWORDS
                }
                score = len(question_tokens & sentence_tokens)
                if score > best_score:
                    best_score = score
                    best_sentence = sentence

        if best_score < 2:
            return UNKNOWN_ANSWER
        return best_sentence.strip()


def build_grounded_prompt(question: str, chunks: list[ChunkHit]) -> str:
    context_blocks = []
    for chunk in chunks:
        location = chunk.metadata.source_name
        if chunk.metadata.page_number is not None:
            location = f"{location}, page {chunk.metadata.page_number}"
        context_blocks.append(
            f"[{chunk.chunk_id}] score={chunk.score} source={location}\n{chunk.text}"
        )
    context = "\n\n".join(context_blocks)
    return (
        "Use the chunks below to answer the question. "
        "Cite chunk ids in square brackets when a sentence depends on a chunk. "
        f"If the answer is not present, respond exactly with: {UNKNOWN_ANSWER}\n\n"
        f"Question: {question}\n\n"
        f"Chunks:\n{context}"
    )


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def create_answer_generator(provider: str, model_name: str) -> AnswerGenerator:
    if provider == "openai":
        return OpenAIAnswerGenerator(model_name=model_name)
    if provider == "extractive":
        return ExtractiveAnswerGenerator()
    raise GenerationError(f"Unknown generation provider: {provider}")


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "how",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "you",
    "your",
}

