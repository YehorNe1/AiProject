# AI Chef – Interactive Recipe Generator

## ✨ Features
* **Local LLMs with Ollama** – works fully offline.
* **LangChain Orchestration** – modular prompts, memory & RAG.
* **Vector RAG** – FAISS index built from `data/recipes_sample.json`.
* **Conversation Memory** – remembers your tastes during the session.
* **Prompt Tests with Pytest** – automated checks for dietary restrictions.
* GUI **Streamlit UI**.

## 🖥️ Prerequisites

| Tool       | Tested version | Install                                                      |
|------------|----------------|--------------------------------------------------------------|
| Python     | 3.10+          | https://python.org                                           |
| **Ollama** | ≥ 0.1.26       | `curl -fsSL https://ollama.com/install.sh | sh`              |
| Git        | any            | –                                                            |

Pull required models:

```bash
ollama pull llama3:8b
ollama pull nomic-embed-text
