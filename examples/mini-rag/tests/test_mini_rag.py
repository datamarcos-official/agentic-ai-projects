#!/usr/bin/env python3
"""Tests for the mini extractive RAG example. Run from this folder:

    python -m unittest tests/test_mini_rag.py
"""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mini_rag  # noqa: E402


DATA_DIR = ROOT / "data"


class ChunkingTests(unittest.TestCase):
    def test_overlap_keeps_a_boundary_phrase_in_two_chunks(self) -> None:
        words = " ".join(f"w{i}" for i in range(12)) + " uniqueanchorterm " + " ".join(
            f"x{i}" for i in range(12)
        )
        chunks = mini_rag.chunk_text("demo.md", words, chunk_size=10, overlap=4)
        self.assertGreaterEqual(len(chunks), 2)
        holding = [chunk for chunk in chunks if "uniqueanchorterm" in chunk.tokens]
        self.assertGreaterEqual(len(holding), 1)

    def test_overlap_copies_boundary_words_into_the_next_window(self) -> None:
        words = " ".join(f"word{i:02d}" for i in range(20))
        chunks = mini_rag.chunk_text("demo.md", words, chunk_size=10, overlap=4)
        self.assertGreaterEqual(len(chunks), 2)
        first_tail = set(chunks[0].text.split()[-4:])
        second_head = set(chunks[1].text.split()[:4])
        self.assertTrue(first_tail & second_head)

    def test_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            mini_rag.chunk_text("demo.md", "hello world", chunk_size=4, overlap=4)


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = mini_rag.build_index(DATA_DIR)

    def test_corpus_loads_three_teaching_notes(self) -> None:
        sources = {chunk.source for chunk in self.index.chunks}
        self.assertEqual(
            sources,
            {"what-is-an-fde.md", "what-is-mcp.md", "what-is-rag.md"},
        )

    def test_mcp_question_retrieves_the_mcp_document(self) -> None:
        hits = self.index.search("What does MCP stand for in AI applications?", k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.source, "what-is-mcp.md")
        self.assertGreater(hits[0].score, 0.05)
        joined = " ".join(hit.chunk.text.lower() for hit in hits)
        self.assertIn("model context protocol", joined)

    def test_rag_question_retrieves_the_rag_document(self) -> None:
        hits = self.index.search(
            "What are the four steps of a retrieval-augmented generation pipeline?",
            k=3,
        )
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.source, "what-is-rag.md")

    def test_fde_question_retrieves_the_fde_document(self) -> None:
        hits = self.index.search(
            "What does a forward deployed engineer do with a customer team?",
            k=3,
        )
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.source, "what-is-an-fde.md")

    def test_confident_answer_includes_citation(self) -> None:
        result = mini_rag.answer_question(
            self.index, "What is the Model Context Protocol?", k=2, min_score=0.08
        )
        self.assertTrue(result.confident)
        self.assertTrue(result.citations)
        self.assertIn("what-is-mcp.md", result.text)
        self.assertIn("Sources:", result.text)

    def test_unrelated_question_is_refused(self) -> None:
        result = mini_rag.answer_question(
            self.index,
            "What is the boiling point of vanadium in degrees Kelvin?",
            k=3,
            min_score=0.08,
        )
        self.assertFalse(result.confident)
        self.assertIn("I don't know", result.text)

    def test_empty_query_returns_no_hits(self) -> None:
        self.assertEqual(self.index.search("   ", k=3), [])


class CliTests(unittest.TestCase):
    def test_cli_answers_a_grounded_question(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = mini_rag.main(["--data", str(DATA_DIR), "What is RAG?"])
        self.assertEqual(code, 0)
        self.assertIn("what-is-rag.md", buf.getvalue())

    def test_cli_returns_distinct_code_when_unsure(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = mini_rag.main(
                ["--data", str(DATA_DIR), "Who won the 1974 FIFA World Cup?"]
            )
        self.assertEqual(code, 3)
        self.assertIn("I don't know", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
