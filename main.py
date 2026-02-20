import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()

st.set_page_config(page_title="PolicySphere AI", page_icon="🌐")
st.title("🌐 PolicySphere AI")
st.write("Upload company policy PDFs and ask questions.")

# ---------------------------
# Strict Prompt
# ---------------------------
strict_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a policy assistant AI.

Rules:
1. Answer ONLY from the provided context.
2. If the answer is not in the context, say:
   "The policy document does not contain this information."
3. Do not make assumptions.
4. Be clear and professional.
5. Mention page number if available.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ---------------------------
# Upload PDFs
# ---------------------------
uploaded_files = st.file_uploader(
    "Upload Policy PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and st.button("Index Policies"):

    documents = []

    UPLOAD_FOLDER = "data/uploads"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    with st.spinner("Processing PDFs..."):

        for file in uploaded_files:
            file_path = os.path.join(UPLOAD_FOLDER, file.name)

            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            # Add page number metadata properly
            for doc in docs:
                doc.metadata["page_number"] = doc.metadata.get("page", "Unknown")

            documents.extend(docs)

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings()

        vectorstore = FAISS.from_documents(split_docs, embeddings)

        st.session_state.vectorstore = vectorstore

    st.success("Policies indexed successfully!")

# ---------------------------
# Ask Questions
# ---------------------------
if "vectorstore" in st.session_state:

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    query = st.text_input("Ask a question about company policies:")

    if query:

        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={"prompt": strict_prompt},
            return_source_documents=True
        )

        with st.spinner("Generating answer..."):
            result = qa_chain({"question": query})

        answer = result["answer"]

        # Store chat history
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("AI", answer))

        # Display Answer
        st.subheader("Answer")
        st.write(answer)

        # Display Sources
        st.subheader("Sources")
        displayed_sources = set()

        for doc in result["source_documents"]:
            source_path = doc.metadata.get("source", "Unknown")
            file_name = os.path.basename(source_path)
            page = doc.metadata.get("page_number", 0)
            if isinstance(page, int):
                page += 1

            source_label = f"{file_name} - Page {page}"

            if source_label not in displayed_sources:
                st.write(f"📄 {source_label}")
                displayed_sources.add(source_label)

        # Display Chat History
        st.subheader("Chat History")

        for speaker, message in st.session_state.chat_history:
            if speaker == "You":
                st.markdown(f"**🧑 You:** {message}")
            else:
                st.markdown(f"**🤖 AI:** {message}")
