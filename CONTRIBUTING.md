# Contributing

Thank you for helping improve these project ideas and examples.

## What is useful

- Bug fixes and clearer error messages in `examples/mini-rag`.
- Extra tests that catch a real failure case.
- New project ideas that are specific, testable and no larger than they need to be.
- Alternative implementations (for example an embeddings-based retriever) in a new folder under `examples/`, with a README and tests.

## What we will not merge

- Promotional or affiliate links.
- Unverifiable claims about jobs, salaries or rankings.
- Examples that require a paid API key in order to run the default path.
- Copies of another organisation’s exercises or wording.

## Running the tests

```bash
cd examples/mini-rag
python3 -m unittest tests/test_mini_rag.py
```

Please run the tests before you open a pull request, and mention what you ran.
