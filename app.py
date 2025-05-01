import sys
import requests
from pathlib import Path
from typing import NoReturn

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate


# ── Configuration ──────────────────────────────────────────────────────────────
LLM_MODEL = "llama3:8b"
EMB_MODEL = "nomic-embed-text"
INDEX_DIR = Path("faiss_index")  # folder containing index.faiss / index.pkl
SYS_PROMPT = Path("prompts/system_prompt.txt").read_text(encoding="utf-8")
OLLAMA_URL = "http://localhost:11434"
# ───────────────────────────────────────────────────────────────────────────────


def abort(msg: str) -> NoReturn:
    """Print an error and exit the program."""
    sys.stderr.write(f"❌  {msg}\n")
    sys.exit(1)


def ensure_ollama_running() -> None:
    """Ping the Ollama server; exit if not reachable."""
    try:
        requests.get(f"{OLLAMA_URL}/api/version", timeout=2)
    except requests.exceptions.RequestException:
        abort("Ollama is not running. Start it with `ollama serve` first.")


def load_vector_store():
    """Load FAISS index created by initialize_index.py."""
    if not INDEX_DIR.exists():
        abort("Vector index missing – run initialize_index.py first.")

    embeddings = OllamaEmbeddings(model=EMB_MODEL)
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,  # we created the file ourselves
    )


def build_chain():
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0.7,
        base_url=OLLAMA_URL,
    )

    retriever = load_vector_store().as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYS_PROMPT),
            ("human", "📋 Context:\n{context}\n\n🤔 Question: {question}"),
        ]
    )

    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )


def cli() -> None:
    ensure_ollama_running()
    chain = build_chain()

    print("\n👨‍🍳  Welcome to AI Chef – ask for a recipe, or type 'exit'!\n")
    try:
        while True:
            question = input("You > ")
            if question.lower() in {"exit", "quit"}:
                break
            answer = chain.invoke({"question": question})["answer"]  # <- no deprecation
            print("\nAI Chef >\n" + answer + "\n")
    except KeyboardInterrupt:
        print("\n👋  Bye!")
    finally:
        sys.exit(0)


if __name__ == "__main__":
    cli()
