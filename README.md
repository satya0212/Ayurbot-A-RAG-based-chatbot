# 🌿 AyurBot — Ayurveda Knowledge Assistant

AyurBot is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about Ayurveda by grounding its responses in *The Ayurveda Encyclopedia*. It combines document retrieval with a large language model to give clear, context-aware answers instead of relying purely on the model's general knowledge.

🔗 **Live demo:** [https://ayurbot-a-rag-based-chatbot-mhpqj6ew9dhbzeepuxz28p.streamlit.app/]

---

## ✨ Features

- **Conversational chat interface** built with Streamlit
- **Retrieval-Augmented Generation (RAG)** — answers are grounded in real source material, reducing hallucination
- **Fast responses** powered by Groq's LLM inference
- **Semantic search** over document content using FAISS vector similarity
- **Clean, dark-themed UI** with chat history

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Orchestration | LangChain |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| PDF Parsing | pypdf |
| UI | Streamlit |

---

## 📂 Project Structure

```
ayurbot/
├── data/                     # Source PDF(s)
├── vectorstore/db_faiss/     # Generated FAISS vector index
├── create_memory_for_llm.py  # Builds the vector store from PDFs
├── medibot.py                # Streamlit chatbot application
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/ayurbot.git
cd ayurbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com/keys).

### 4. Build the knowledge base
Place your PDF(s) inside the `data/` folder, then run:
```bash
python create_memory_for_llm.py
```
This creates the FAISS vector store used for retrieval.

### 5. Launch the chatbot
```bash
streamlit run medibot.py
```
The app will open at `http://localhost:8501`.

---

## 💡 How It Works

1. **Document ingestion** — PDFs are loaded and split into overlapping text chunks.
2. **Embedding** — Each chunk is converted into a vector using a sentence-transformer model.
3. **Storage** — Vectors are indexed in FAISS for fast similarity search.
4. **Retrieval** — When a user asks a question, the most relevant chunks are retrieved.
5. **Generation** — Retrieved context + the user's question are passed to the LLM, which generates a grounded answer.

---

## ⚠️ Disclaimer

AyurBot is an educational project for exploring Ayurvedic knowledge from a reference text. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare provider for medical concerns.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
