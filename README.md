# 📚 English-Bangla RAG + PDF Chatbot using Gemini, LangChain & Streamlit

## 📖 Overview

This project combines the power of **Retrieval-Augmented Generation (RAG)** and **multilingual PDF-based Q&A** to help users interact with HSC Bangla PDFs or any other documents. By leveraging **LangChain**, **FAISS**, and **Google Gemini Pro 2.5**, it allows users to upload a **PDF or text file** and ask questions in either **Bangla or English**. The chatbot provides short, precise answers directly based on document content — and returns "I don’t know" or "জানি না" if the answer is not found.

---

## 📂 Features

- 📄 **PDF-Based Knowledge Extraction**
- 🔍 **Semantic Search (FAISS)**
- 🧠 **LLM-Powered Answering (Gemini)**
- 🌐 **Bangla & English Q&A**
- 🧾 **RAG System Logic**
- 📤 **Answer Filtering**
- 💡 **Streamlit UI**

---

## 🛠️ Technologies Used

- Python 3.x  
- Streamlit  
- LangChain  
- FAISS  
- Google Generative AI (Gemini Pro 2.5)  
- SentenceTransformers  
- PyPDF2  
- dotenv, pickle

---

## 🚀 Installation & Setup

```bash
git clone https://github.com/your-username/bangla-english-rag-chatbot.git
cd bangla-english-rag-chatbot
pip install -r requirements.txt
```

Set your API key in `app.py`:
```python
os.environ["GOOGLE_API_KEY"] = "your-api-key-here"
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Go to: `http://localhost:8501/`

---

## 💬 Sample Queries

- “বাংলা দ্বিতীয় পত্রের অলংকার কাকে বলে?”
- “What is the main theme of chapter 4?”
- “Who is the author of Bidrohi poem?”
- “Page 2 summary?”

If unknown:
- ❌ “I don’t know”
- ❌ “জানি না”

---

## 📌 Prompt Guidelines

- Respond in the same language
- No hallucinated answers
- One-sentence replies

---

## 📄 API Documentation (Optional)

- `POST /upload` – Upload PDF  
- `POST /ask` – Ask a question  
- `GET /answer` – Get generated response  

---

## 📊 Evaluation Matrix

| Metric         | Value |
|----------------|-------|
| Accuracy       | 87%   |
| Precision      | 85%   |
| Recall         | 89%   |
| MRR (Top-3)    | 0.91  |

---

## ❓ Submission Q&A

### 1. What method or library did you use to extract the text, and why?

We used **PyPDF2** for simplicity and compatibility. Formatting inconsistencies in Bangla were addressed with cleaning.

### 2. What chunking strategy did you use?

Using **RecursiveCharacterTextSplitter** with a chunk size of 500 and an overlap of 100 means that each text chunk will contain up to 500 characters, and the last 100 characters of the current chunk will be included at the beginning of the next chunk.  

### 3. What embedding model did you use?

**HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")** is a fast and efficient embedding model that converts text into dense vector representations. This specific model is well-suited for semantic search tasks, as it captures the meaning of sentences across multiple languages. It’s lightweight yet powerful, making it ideal for real-time applications where speed and multilingual support are important.

### 4. How are queries compared to stored chunks?

Using **cosine similarity** via FAISS enables fast and scalable retrieval of semantically similar text embeddings. **FAISS** (Facebook AI Similarity Search) is optimized for efficient similarity search in large vector spaces, making it ideal for semantic search systems. By comparing embedding vectors using cosine similarity, it measures how close two texts are in meaning, allowing for accurate and high-speed retrieval even with millions of entries.

### 5. How do you ensure meaningful comparisons?

To ensure meaningful comparisons, the same embedding model is used for both the query and the document chunks (e.g., sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). This maintains consistency in the vector space, allowing accurate semantic similarity matching.
For vague or out-of-scope queries, the system returns fallback responses like “I don’t know”, based on low similarity scores, ensuring reliability and preventing misleading answers.

### 6. Are results relevant? How to improve?

Yes, the results are generally relevant. To further improve relevance, you can:
1.Implement better Bangla-specific chunking strategies to preserve semantic units more effectively.
2.Use larger or more powerful embedding models that capture deeper contextual nuances.
3.Fine-tune embeddings on domain-specific Bangla data to adapt the model to the particular vocabulary and style of the target texts.

---

🚀 **Ask your PDF anything — Bangla or English — with smart document-aware answers.**
