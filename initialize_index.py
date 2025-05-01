import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings

DATA_PATH = Path("data/recipes_sample.json")
INDEX_DIR = Path("faiss_index")
EMB_MODEL = "nomic-embed-text"


def main() -> None:
    if INDEX_DIR.exists():
        print("Index already exists – nothing to do.")
        return

    recipes = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    docs = [f"{r['name']}\n{r['description']}" for r in recipes]

    embeddings = OllamaEmbeddings(model=EMB_MODEL)
    store = FAISS.from_texts(docs, embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    store.save_local(INDEX_DIR.as_posix())
    print(f"Built vector store with {len(docs)} docs → {INDEX_DIR}/")


if __name__ == "__main__":
    main()
