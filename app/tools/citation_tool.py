from __future__ import annotations

import re

from app.models.schemas import Citation, RetrievedChunk

_STOPWORDS = {
    "the",
    "is",
    "a",
    "an",
    "what",
    "where",
    "and",
    "of",
    "in",
    "to",
    "for",
    "on",
    "with",
    "our",
    "based",
    "against",
    "compare",
}


def _tokenize(text: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[A-Za-z0-9\-]+", text) if word.lower() not in _STOPWORDS}


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    if len(sentences) == 1 and "\n" in text:
        sentences = [line.strip() for line in text.splitlines() if line.strip()]
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _best_quote_for_query(query: str, chunk_text: str) -> str:
    query_tokens = _tokenize(query)
    sentences = _split_sentences(chunk_text)
    if not sentences:
        return chunk_text[:280]

    best_sentence = sentences[0]
    best_score = -1.0

    for sentence in sentences:
        sentence_tokens = _tokenize(sentence)
        overlap = len(query_tokens.intersection(sentence_tokens))
        digit_bonus = 1.5 if re.search(r"\d", sentence) else 0.0
        score = overlap + digit_bonus
        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence[:500]


class CitationTool:
    name = "citation_extractor"
    description = "Extract page-numbered, quote-level citations from retrieved chunks."

    def run(self, query: str, retrieved_chunks: list[RetrievedChunk], limit: int = 3) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[int, str]] = set()

        for chunk in retrieved_chunks:
            quote = _best_quote_for_query(query=query, chunk_text=chunk.content)
            key = (chunk.page_number, quote)
            if key in seen:
                continue
            seen.add(key)

            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    quote=quote,
                    source_type=chunk.source_type,
                )
            )
            if len(citations) >= limit:
                break

        return citations
