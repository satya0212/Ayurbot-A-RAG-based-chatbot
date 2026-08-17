import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

CUSTOM_PROMPT_TEMPLATE = """
You are a helpful Ayurveda assistant. Use the context below to answer the user's question
in a clear, warm, and easy-to-understand way. Avoid technical jargon unless necessary,
and explain things as if talking to someone with no prior Ayurveda knowledge.

If the context doesn't fully answer the question, say so honestly, then share the
closest relevant information you do have instead of refusing outright.

Context: {context}
Question: {question}

Answer:
"""

st.set_page_config(
    page_title="AyurBot",
    page_icon="🌿",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
    }
    .main-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 700;
        color: #6fcf97;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #a3b3ab;
        font-size: 1rem;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }

    /* Chat message bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 10px 14px;
        margin-bottom: 8px;
        background-color: #1c1f26;
    }

    /* Force readable text color inside all chat messages */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div {
        color: #f0f2f0 !important;
        font-size: 1rem;
        line-height: 1.6;
    }

    /* Chat input box text */
    [data-testid="stChatInput"] textarea {
        color: #f0f2f0 !important;
    }

    /* Spinner text */
    .stSpinner p {
        color: #a3b3ab !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return db


def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )


def load_llm():
    return ChatGroq(
        model_name="openai/gpt-oss-120b",
        temperature=0.3,
        groq_api_key=GROQ_API_KEY,
    )


def main():
    st.markdown('<p class="main-title">🌿 AyurBot</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Your friendly guide to the Ayurveda Encyclopedia</p>',
        unsafe_allow_html=True,
    )

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is missing. Add it to your .env file and restart.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # replay chat history
    for message in st.session_state.messages:
        avatar = "🧑" if message["role"] == "user" else "🌿"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about Ayurveda...")

    if prompt:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar="🌿"):
            with st.spinner("Thinking..."):
                try:
                    vectorstore = get_vectorstore()

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=load_llm(),
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=False,
                        chain_type_kwargs={"prompt": set_custom_prompt()},
                    )

                    response = qa_chain.invoke({"query": prompt})
                    answer = response["result"].strip()

                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    error_msg = f"Sorry, something went wrong: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


if __name__ == "__main__":
    main()
