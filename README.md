# Agentic AI projects: beginner to advanced

A collection of project ideas for learning agentic AI — systems where a language model can use tools, retrieve evidence, and follow a loop with limits — plus one fully worked example you can run offline.

The ideas are deliberately small. Finish one, write down what failed, then move on. A finished repository with tests and a README is more useful than a list of frameworks you have imported but not used.

Companion reading, if you want a course-shaped path rather than a project list: the free [Agentic AI roadmap](https://www.datamarcos.com/guides/agentic-ai-roadmap) and [Agentic AI projects for beginners](https://www.datamarcos.com/guides/agentic-ai-projects-for-beginners) on datamarcos.com.

## Worked example: mini-RAG (start here)

[`examples/mini-rag`](examples/mini-rag) is a retrieval-augmented generation pipeline that uses **only the Python standard library**. It does not call an API and does not need an API key.

It loads a few local Markdown files, chunks them, builds a TF-IDF index, retrieves the best-matching passages, and returns an extractive answer with citations — or “I don’t know” when nothing in the corpus is similar enough.

```bash
cd examples/mini-rag
python3 mini_rag.py "What does MCP stand for?"
python3 -m unittest tests/test_mini_rag.py
```

Read the [example README](examples/mini-rag/README.md) for how it works and what to build next (embeddings, a vector database, a language model for the final write-up).

## How to use the project list

For each project:

1. Write a one-paragraph brief: who it is for, what “done” looks like, and what it will not do.
2. Keep the first version narrower than you would like.
3. Add tests or a fixed evaluation set before you add another feature.
4. Publish a README with how to run it, a limitation section, and one short demo.

Do not start with a multi-agent framework. Get a single loop you can explain, then add complexity only when a test shows you need it. Anthropic makes the same point in [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents).

## Beginner

These should be possible after basic Python, HTTP and one week of prompting practice.

### 1. Structured ticket classifier

**Goal:** given a short support ticket, return JSON with a category, a priority, and a one-line rationale. Validate the JSON in code.

**Done when:** invalid model output is rejected and retried or marked as a failure; you have ten labelled examples and a pass rate.

**Stretch:** log every failure and add two of them to the evaluation set.

### 2. Tool-calling loop in plain Python

**Goal:** an agent with two mock tools (for example a calculator and a document lookup that reads local files). No framework.

**Done when:** the model can choose a tool, your code runs it, the result is sent back, and the loop stops after a fixed number of steps.

**Stretch:** record a trace of every step to a JSONL file.

### 3. Offline mini-RAG (implemented in this repo)

**Goal:** run and then extend [`examples/mini-rag`](examples/mini-rag).

**Done when:** you can answer questions about a corpus you added yourself, with citations, and the tests still pass.

**Stretch:** swap TF-IDF for embeddings, or add a second corpus.

## Intermediate

### 4. RAG that is allowed to say “I don’t know”

**Goal:** a question-answering tool over a policy, syllabus or product manual. Every answer cites a passage. If the best retrieval score is below a threshold, the system refuses.

**Done when:** a held-out set includes at least five questions the corpus cannot answer, and all five are refused.

See also: [RAG vs fine-tuning](https://www.datamarcos.com/guides/rag-vs-fine-tuning).

### 5. Human-approval workflow

**Goal:** a two-step workflow that drafts an action (a reply, a ticket update, a file change) and waits for an explicit approve/reject before it takes effect. A simple command-line prompt is enough; a graph library is optional.

**Done when:** you can show a rejected draft that was not applied, and an approved one that was.

### 6. A small MCP-style tool server

**Goal:** expose one local tool (file search, SQLite query, or a calculator) behind a small server that a client can call. Use the official [Model Context Protocol](https://modelcontextprotocol.io/) if you are ready for it, or a plain HTTP interface first if you are not.

**Done when:** a client can list the tool, call it, and get a structured result, with secrets kept on the server.

Companion: [How to build an MCP server](https://www.datamarcos.com/guides/how-to-build-an-mcp-server).

## Advanced

### 7. Evaluation harness

**Goal:** a fixed suite of tasks with expected outcomes. Changing a prompt, model or retriever prints pass rate, latency and (if you use a paid model) estimated cost.

**Done when:** a regression in retrieval quality fails the suite. [Ragas](https://docs.ragas.io/) is useful for RAG-specific metrics, but a JSON file plus a script is a valid start.

### 8. Prompt-injection and least-privilege tests

**Goal:** a short red-team list for your agent, informed by the [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/). Try to override instructions, call a disallowed tool, or leak a secret.

**Done when:** each attack has a result (succeeded or blocked) and you have at least one fix in the code, not only in the prompt.

### 9. A deployable service with a handover pack

**Goal:** wrap project 4 or 5 behind an API. Add logging, timeouts, rate limits and secret management. Write an architecture note, a runbook and a five-minute demo script.

**Done when:** a teammate can start the service from the README alone and replay one traced failure.

This is the closest of these projects to forward deployed work. The [FDE self-study roadmap](https://github.com/datamarcos-official/fde-roadmap) puts it in weeks 11–12.

## Choosing a framework later

Once you can explain your own loop, it is reasonable to adopt a library. [LangGraph](https://langchain-ai.github.io/langgraph/) is a good fit for stateful workflows with approval steps. For a comparison of common options, see [LangGraph vs CrewAI vs AutoGen](https://www.datamarcos.com/guides/langgraph-vs-crewai-vs-autogen) and [n8n vs LangGraph](https://www.datamarcos.com/guides/n8n-vs-langgraph).

## Contributing

New project ideas, failing tests, and clearer write-ups are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Please do not send pull requests whose only change is a promotional link.

## License

This repository is released under the [MIT License](LICENSE).
