# Mini-RAG: an offline extractive RAG example

A small retrieval-augmented generation (RAG) pipeline you can run without installing packages or calling a model API.

It is extractive on purpose. The program retrieves passages and shows them with citations. It does not ask a language model to rewrite the answer. That keeps the example honest, free to run, and easy to test. Once you understand this loop, you can swap in embeddings, a vector database, and a model for the final write-up.

## What it does

1. Loads every `.md` and `.txt` file in `data/`.
2. Splits each file into overlapping word windows.
3. Builds a TF-IDF index (term frequency × inverse document frequency).
4. Scores the question against every chunk with cosine similarity.
5. Returns the top chunks, or says it does not know if the best score is below a threshold.

## Run it

Python 3.10 or newer. No third-party packages.

```bash
cd examples/mini-rag
python3 mini_rag.py "What does MCP stand for?"
python3 mini_rag.py "What are the four steps of a RAG pipeline?"
python3 mini_rag.py "Who won the 1974 FIFA World Cup?"
```

Useful flags:

```bash
python3 mini_rag.py --data ./data --k 3 --min-score 0.08 "What is an FDE?"
```

The last example above should refuse, because the 1974 World Cup is not in the corpus. A refusal is a correct behaviour, not a bug.

## Tests

```bash
python3 -m unittest tests/test_mini_rag.py
```

The tests check chunk overlap, that each sample question retrieves the right document, that answers include citations, and that unrelated questions are refused.

## Sample corpus

The `data/` directory has three short teaching notes, written for this example:

- `what-is-rag.md` — the retrieval-augmented generation pattern
- `what-is-mcp.md` — the Model Context Protocol
- `what-is-an-fde.md` — what a forward deployed engineer does

Add your own Markdown or text files and ask questions about them.

## Ideas for a next version

- Replace TF-IDF with embeddings (for example a small sentence-transformers model) and compare retrieval quality on the same questions.
- Store chunks in [pgvector](https://github.com/pgvector/pgvector) or another vector store.
- Pass the retrieved passages to a language model and require the model to cite chunk IDs. Keep the refusal path.
- Measure groundedness with a library such as [Ragas](https://docs.ragas.io/), or with a simple “does the answer appear in the cited chunk?” check.

## Why there is no default model call

A default path that needs an API key would make the tests and the first run dependent on a vendor and a secret. The retrieval loop is the part this example is trying to teach. Wiring a model in is a good follow-on project, not a requirement to get started.
