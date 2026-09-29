# 🎉 ReAct Agent Demo (Ollama Edition)

This is a minimal **ReAct** agent built with the latest versions of **LangChain** and **LangGraph**, powered by **Ollama**.

> 🚀 *Quick start* – just run the snippet below after installing the dependencies.

## Setup

```bash
# 1️⃣ Create the project folder (if you haven't already)
mkdir react_agent
cd react_agent

# 2️⃣ (Optional) Create a virtual environment
echo "Use your favourite method. On Windows, for example:"
python -m venv .venv
.
.venv\Scripts\Activate

# 3️⃣ Install deps
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

You’ll see a prompt:

```
You: What is 7 plus 8?
```

The agent will respond with the answer, possibly showing intermediate “thoughts” if you enable debug printing.

## Switching to OpenAI (or other LLMs)

If you want to use an OpenAI model instead of Ollama, change the `llm` init in `main.py`:

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

…and comment out or remove the `langchain-ollama` dependency.

---

## File overview

- **`main.py`** – Full agent definition (LLM, tool, graph) and a tiny REPL loop.
- **`requirements.txt`** – Exact package list.
- **`.env`** – Holds your Ollama base URL if you need to override the default.
- **`README.md`** – This file.

Happy hacking! 🎈
