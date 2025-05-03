# AI Chef – Streamlit GUI (v4)
# -------------------------------------------------------------
# Launch with:
#     streamlit run app.py
# -------------------------------------------------------------

from __future__ import annotations

import random
from pathlib import Path
from typing import List, TypedDict, cast

import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# ── Configuration ────────────────────────────────────────────────────────────
LLM_MODEL = "llama3:8b"
EMB_MODEL = "nomic-embed-text"
INDEX_DIR = Path("faiss_index")
SYS_PROMPT = Path("prompts/system_prompt.txt").read_text(encoding="utf-8")
OLLAMA_URL = "http://localhost:11434"
# ----------------------------------------------------------------------------


class ChatMsg(TypedDict):
    role: str  # "user" | "assistant"
    content: str


# ── LangChain helpers ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_chain() -> ConversationalRetrievalChain:
    llm = ChatOllama(model=LLM_MODEL, temperature=0.7, base_url=OLLAMA_URL)

    embeddings = OllamaEmbeddings(model=EMB_MODEL)
    vectordb = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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


chain = _load_chain()

# ── Streamlit UI setup ───────────────────────────────────────────────────────
st.set_page_config(page_title="AI Chef", page_icon="🍳", layout="wide")

# Session‑state defaults ------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # type: List[ChatMsg]
if "favorites" not in st.session_state:
    st.session_state["favorites"] = []  # type: List[str]

messages: List[ChatMsg] = cast(List[ChatMsg], st.session_state["messages"])
favorites: List[str] = cast(List[str], st.session_state["favorites"])

# Sidebar – user preferences --------------------------------------------------
with st.sidebar:
    st.header("🍽️ Preferences")
    dietary = st.text_input("Dietary restrictions (comma‑separated)")
    dislikes = st.text_input("Disliked ingredients (comma‑separated)")
    cuisine = st.text_input("Preferred cuisine style")

    st.divider()
    st.header("⭐ Favourite recipes")
    if favorites:
        for recipe_name in favorites:
            st.markdown(f"• {recipe_name}")
    else:
        st.caption("No favourites yet.")

# Main header -----------------------------------------------------------------
st.title("👨‍🍳 AI Chef – Interactive Recipe Generator")

# Display chat history --------------------------------------------------------
for msg in messages:
    st.chat_message(msg["role"]).markdown(msg["content"])


# Helper – build composite question ------------------------------------------

def _compose_question(raw: str) -> str:
    parts: List[str] = []
    if dietary:
        parts.append(f"My dietary restrictions: {dietary}.")
    if dislikes:
        parts.append(f"I dislike: {dislikes}.")
    if cuisine:
        parts.append(f"I'd love {cuisine} cuisine.")
    parts.append(raw)
    return " ".join(parts)


# Helper – favourites logic ---------------------------------------------------

def _add_to_favourites(recipe_title: str) -> None:
    if recipe_title and recipe_title not in favorites:
        favorites.append(recipe_title)
        st.toast("Saved to favourites!", icon="❤️")
        st.session_state["_just_saved"] = True


# Unified function to process a user prompt -----------------------------------

def _process_prompt(user_prompt: str) -> None:
    # 1) echo user message
    messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").markdown(user_prompt)

    # 2) call chain
    with st.spinner("Cooking up your recipe…"):
        answer = cast(str, chain.invoke({"question": _compose_question(user_prompt)})["answer"])

    # 3) render assistant reply
    st.chat_message("assistant").markdown(answer)

    # 4) remember last recipe title for favourite button
    st.session_state["last_recipe_title"] = answer.splitlines()[0].lstrip("# ").strip()

    # 5) store reply in history
    messages.append({"role": "assistant", "content": answer})


# Chat input ------------------------------------------------------------------
user_input = st.chat_input(placeholder="Ask AI Chef for a recipe…")
if user_input:
    _process_prompt(user_input)

# Surprise‑me button ----------------------------------------------------------
if st.button("🎲 Surprise me"):
    _process_prompt(random.choice([
        "Suggest a quick vegan lunch.",
        "I have 20 minutes and love Italian flavours.",
        "Make me a spicy, protein‑rich dinner.",
    ]))

# Global "Add to favourites" button ------------------------------------------
if "last_recipe_title" in st.session_state:
    last_title: str = st.session_state["last_recipe_title"]
    if st.button("❤️ Add last recipe to favourites", key="fav_global"):
        _add_to_favourites(last_title)
        st.rerun()  # ← новый API вместо experimental_rerun()

# Celebration after each save --------------------------------------------------
if st.session_state.pop("_just_saved", False):
    st.balloons()
