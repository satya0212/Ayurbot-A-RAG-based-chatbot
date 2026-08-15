import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

MODEL_PATH = r"C:\Users\user\OneDrive\Desktop\ayurbot\models\AyurParam-2.9b-it-fp16.gguf"
DB_FAISS_PATH = "vectorstore/db_faiss"


def load_llm():
    return LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.5,
        max_tokens=512,
        n_ctx=2048,
        n_threads=os.cpu_count(),
        verbose=False,
    )


custom_prompt_template = """
You are a helpful Ayurveda assistant. Use the context below to answer the user's question
in a clear, warm, and easy-to-understand way. Avoid technical jargon unless necessary,
and explain things as if talking to someone with no prior Ayurveda knowledge.

If the context doesn't fully answer the question, say so honestly, then share the
closest relevant information you do have instead of refusing outright.

Context: {context}
Question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(custom_prompt_template)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = load_llm()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def ask(query: str) -> str:
    try:
        answer = rag_chain.invoke(query)
        return answer.strip()
    except Exception as e:
        return f"Sorry, I couldn't get an answer right now. ({e})"


if __name__ == "__main__":
    print("AyurBot -- ask me anything about Ayurveda (type 'exit' to quit)\n")
    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in {"exit", "quit"}:
            print("Take care!")
            break
        if not user_query:
            continue

        print("Thinking...")
        answer = ask(user_query)
        print(f"\nAyurBot: {answer}\n")