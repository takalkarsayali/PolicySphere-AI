# 🌐 PolicySphere AI

## Intelligent Company Policy Assistant

PolicySphere AI is a Retrieval-Augmented Generation (RAG) application that allows users to upload company policy PDFs and ask contextual questions. The system retrieves relevant policy sections using semantic search and generates answers strictly from the provided documents.

## Preview
![img](https://github.com/user-attachments/assets/db745222-8474-4d5e-84a4-8d647e8628d9)
![img](https://github.com/user-attachments/assets/4c611a8b-4cf6-4be1-8591-f3e688bbd4f3)

## 🚀 Features

- 📄 Upload multiple company policy PDFs
- 🧠 Semantic search using FAISS vector store
- 💬 Conversational memory (context-aware follow-up questions)
- 🔒 Strict prompt guardrails (no hallucinations)
- 📌 Source citation (File name + Page number)
- 🗂 Clean project structure (uploads separated from code)

---

## 🏗 Architecture Overview

```
User Question
      ↓
Convert to Embedding
      ↓
FAISS Vector Search
      ↓
Retrieve Relevant Chunks
      ↓
LLM (Groq - LLaMA 3.1)
      ↓
Strict Prompt Enforcement
      ↓
Answer + Source Citation
```

---

## 📁 Project Structure

```
PolicySphere-AI/
│
├── main.py
├── data/
│     └── uploads/        # Uploaded PDFs
│
├── vectorstore/          # (Optional) Stored FAISS index
├── requirements.txt
└── README.md
```

---

## 🛠 Tech Stack

- **Streamlit** – UI
- **LangChain** – RAG framework
- **FAISS** – Vector database
- **HuggingFace Embeddings**
- **Groq (LLaMA 3.1-8B)** – LLM
- **Python**

---

## ⚙ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/PolicySphere-AI.git
cd PolicySphere-AI
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Environment Variables

Create a `.env` file in root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## ▶ Run the Application

```bash
streamlit run main.py
```

---

## 📌 How It Works

1. Upload one or more policy PDFs.
2. PDFs are stored in `data/uploads/`.
3. Documents are split into chunks.
4. Chunks are converted into embeddings.
5. Embeddings are stored in FAISS.
6. User question is converted into embedding.
7. Most relevant chunks are retrieved.
8. LLM generates answer strictly from retrieved context.
9. File name and page number are displayed as sources.

---

## 🔐 Strict Answering Policy

The system is designed to:

- Answer ONLY from uploaded documents
- Avoid assumptions
- Prevent hallucinations
- Return:  
  `"The policy document does not contain this information."`  
  if answer is not found

---

## 💡 Example Use Cases

- HR policy assistant
- Employee handbook Q&A
- Compliance document search
- Legal document retrieval
- Internal knowledge assistant

---

## 🧠 Future Improvements

- Persist FAISS index to disk
- ChatGPT-style chat UI
- PDF page preview
- Multi-user authentication
- Cloud deployment
- Replace FAISS with Pinecone (for scalability)

---

## 📜 License

This project is for educational and demonstration purposes.

---

## 👩‍💻 Author

**Sayali Takalkar**  
Software & Data Engineering Enthusiast  
India 🇮🇳

---

⭐ If you found this project helpful, consider giving it a star!
