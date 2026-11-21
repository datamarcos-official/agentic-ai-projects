#!/usr/bin/env python3
"""A minimal, offline retrieval-augmented generation (RAG) pipeline.

This example is intentionally small. It uses only the Python standard library so
you can run it without installing packages or calling a paid API.

What it does:
  1. Loads local Markdown/text files.
  2. Splits them into overlapping chunks.
  3. Builds a TF-IDF index over those chunks.
  4. Retrieves the most similar chunks for a question.
  5. Returns an extractive answer with citations, or "I don't know" when
     nothing in the corpus is similar enough.

A production RAG system would usually add embeddings, a vector database, and a
language model to write the final answer. Plug those in later; the loop below
is the part you should understand first.

Usage:
  python mini_rag.py "What does MCP stand for?"
  python mini_rag.py --data ./data --k 3 --min-score 0.08 "What is RAG?"
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping numbers (useful for years and versions)."""
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    tokens: tuple[str, ...]


@dataclass
class Retrieval:
    chunk: Chunk
    score: float


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Retrieval] = field(default_factory=list)
    confident: bool = False


def load_documents(data_dir: Path) -> list[tuple[str, str]]:
    """Return (relative_name, text) for every .md or .txt file under data_dir."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs: list[tuple[str, str]] = []
    paths = sorted(
        list(data_dir.rglob("*.md")) + list(data_dir.rglob("*.txt")),
        key=lambda p: str(p).lower(),
    )
    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append((path.relative_to(data_dir).as_posix(), text))
    if not docs:
        raise ValueError(f"No .md or .txt files found in {data_dir}")
    return docs


def chunk_text(
    source: str,
    text: str,
    chunk_size: int = 80,
    overlap: int = 20,
) -> list[Chunk]:
    """Split a document into overlapping word windows.

    `chunk_size` and `overlap` are measured in whitespace-separated words, not
    tokens after punctuation stripping. Overlap helps a fact that sits on a
    boundary appear in more than one chunk.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 1
    step = chunk_size - overlap
    while start < len(words):
        window = words[start : start + chunk_size]
        body = " ".join(window)
        tokens = tuple(tokenize(body))
        if tokens:
            chunks.append(
                Chunk(
                    chunk_id=f"{source}#c{index}",
                    source=source,
                    text=body,
                    tokens=tokens,
                )
            )
            index += 1
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def _term_frequencies(tokens: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


class TfidfIndex:
    """In-memory TF-IDF index with cosine similarity search."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Cannot build an index from an empty chunk list")
        self.chunks = chunks
        self._df: dict[str, int] = {}
        self._vectors: list[dict[str, float]] = []
        self._norms: list[float] = []
        self._build()

    @property
    def document_count(self) -> int:
        return len(self.chunks)

    def _build(self) -> None:
        tfs: list[dict[str, int]] = []
        for chunk in self.chunks:
            tf = _term_frequencies(chunk.tokens)
            tfs.append(tf)
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1

        n = len(self.chunks)
        for tf in tfs:
            weights = {
                term: (count / sum(tf.values())) * math.log((n + 1) / (self._df[term] + 1)) + 1e-12
                for term, count in tf.items()
            }
            norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
            self._vectors.append(weights)
            self._norms.append(norm)

    def _vectorize_query(self, tokens: list[str]) -> tuple[dict[str, float], float]:
        if not tokens:
            return {}, 0.0
        tf = _term_frequencies(tokens)
        n = len(self.chunks)
        weights: dict[str, float] = {}
        for term, count in tf.items():
            df = self._df.get(term, 0)
            # Smoothed IDF so rare query terms that happen to be in the corpus
            # still have weight, and unseen terms do not dominate.
            idf = math.log((n + 1) / (df + 1))
            weights[term] = (count / sum(tf.values())) * idf
        norm = math.sqrt(sum(value * value for value in weights.values()))
        return weights, norm

    def search(self, question: str, k: int = 3) -> list[Retrieval]:
        if k <= 0:
            raise ValueError("k must be positive")
        tokens = tokenize(question)
        q_weights, q_norm = self._vectorize_query(tokens)
        if q_norm == 0.0:
            return []

        scored: list[Retrieval] = []
        for chunk, weights, norm in zip(self.chunks, self._vectors, self._norms):
            dot = sum(weights[t] * q_weights[t] for t in q_weights if t in weights)
            score = dot / (norm * q_norm)
            if score > 0:
                scored.append(Retrieval(chunk=chunk, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]


def build_index(
    data_dir: Path, chunk_size: int = 80, overlap: int = 20
) -> TfidfIndex:
    chunks: list[Chunk] = []
    for source, text in load_documents(data_dir):
        chunks.extend(chunk_text(source, text, chunk_size=chunk_size, overlap=overlap))
    return TfidfIndex(chunks)


def answer_question(
    index: TfidfIndex,
    question: str,
    k: int = 3,
    min_score: float = 0.08,
) -> Answer:
    """Return an extractive, cited answer, or a refusal if retrieval is weak."""
    hits = index.search(question, k=k)
    strong = [hit for hit in hits if hit.score >= min_score]
    if not strong:
        return Answer(
            question=question,
            text=(
                "I don't know based on the files in this knowledge base. "
                "Try a more specific question, or add a document that covers the topic."
            ),
            citations=hits,
            confident=False,
        )

    lines = [
        "Here is what the local documents say (extractive answer, not a language model):",
        "",
    ]
    for hit in strong:
        lines.append(f"- ({hit.chunk.chunk_id}, score {hit.score:.3f}) {hit.chunk.text}")
    lines.append("")
    lines.append("Sources: " + ", ".join(dict.fromkeys(hit.chunk.source for hit in strong)))
    return Answer(
        question=question,
        text="\n".join(lines),
        citations=strong,
        confident=True,
    )


def format_answer(answer: Answer) -> str:
    header = f"Q: {answer.question}"
    return f"{header}\n{answer.text}\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mini extractive RAG example.")
    parser.add_argument("question", nargs="+", help="The question to ask the local corpus.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Directory of .md or .txt files (default: ./data next to this script).",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.08,
        help="Minimum cosine similarity required to attempt an answer.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=80,
        help="Chunk size in words.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=20,
        help="Overlap between consecutive chunks, in words.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    question = " ".join(args.question).strip()
    if not question:
        print("Please provide a question.", file=sys.stderr)
        return 2
    try:
        index = build_index(args.data, chunk_size=args.chunk_size, overlap=args.overlap)
        result = answer_question(index, question, k=args.k, min_score=args.min_score)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_answer(result))
    return 0 if result.confident else 3


if __name__ == "__main__":
    raise SystemExit(main())
