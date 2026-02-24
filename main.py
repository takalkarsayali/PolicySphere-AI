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
# CONFIG
# ---------------------------
load_dotenv()
st.set_page_config(layout="wide", page_title="PolicySphere AI", page_icon="🌐")

# ---------------------------
# CUSTOM CSS (Panels + Layout)
# ---------------------------

st.markdown("""
<style>
           
div[data-testid="column"] > div:first-child {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    height: 80vh;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
}

div[data-testid="column"]:nth-of-type(2) {
    padding-left: 40px;
    padding-right: 40px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# STRICT PROMPT
# ---------------------------
strict_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a policy assistant AI.

Rules:
1. Answer ONLY from the provided context.
2. If the answer is not in the context, say:
   "The policy document does not contain this information."
3. Be clear and professional.
4. Mention page number if available.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ---------------------------
# MEMORY
# ---------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

# ---------------------------
# LAYOUT
# ---------------------------
left, center, right = st.columns([1, 2, 1])

# ---------------------------
# LEFT PANEL (UPLOAD)
# ---------------------------
with left:

    with st.container(border=True):

        st.markdown(
            "<h3 style='margin-top:0;'>📂 <u>Upload Policies</u></h3>",
            unsafe_allow_html=True
        )

        uploaded_files = st.file_uploader(
            "",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            if st.button("Load Policies", use_container_width=True):

                documents = []
                UPLOAD_FOLDER = "data/uploads"
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)

                with st.spinner("Loading Policies..."):

                    for file in uploaded_files:
                        file_path = os.path.join(UPLOAD_FOLDER, file.name)

                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())

                        loader = PyPDFLoader(file_path)
                        docs = loader.load()

                        for doc in docs:
                            doc.metadata["page_number"] = doc.metadata.get("page", 0)

                        documents.extend(docs)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    split_docs = splitter.split_documents(documents)

                    embeddings = HuggingFaceEmbeddings()
                    vectorstore = FAISS.from_documents(split_docs, embeddings)

                    st.session_state.vectorstore = vectorstore

                st.success("Policies loaded successfully!")
                
        else:
            st.error("No documents uploaded.", icon="❌")
# ---------------------------
# CENTER PANEL (CHAT)
# ---------------------------
with center:
    st.title("PolicySphere AI 🌐")
    st.badge("AI-powered assistant for company policy documents.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("💬 Ask Policy Questions")
    query = st.text_input(
        "Type your question here:",
        key="user_query",
        disabled="vectorstore" not in st.session_state
    )

    # ❌ If policies NOT loaded
    if "vectorstore" not in st.session_state and uploaded_files:
        st.info("📂 Please load policies to enable Q&A.")
    
    if query and "vectorstore" in st.session_state:

        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            ),
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={"prompt": strict_prompt},
            return_source_documents=True
        )

        with st.spinner("Generating answer..."):
            result = qa_chain({"question": query})

        answer = result["answer"]

        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")

        displayed = set()
        for doc in result["source_documents"]:
            source_path = doc.metadata.get("source", "")
            file_name = os.path.basename(source_path)
            page = doc.metadata.get("page_number", 0)
            if isinstance(page, int):
                page += 1

            label = f"{file_name} - Page {page}"
            if label not in displayed:
                st.write(f"📄 {label}")
                displayed.add(label)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# RIGHT PANEL (MEMORY)
# ---------------------------

with right:
    with st.container(border=True):

        st.markdown(
            "<h3 style='margin-top:0;'>🧠 <u>Conversation Memory</u></h3>",
            unsafe_allow_html=True
        )
        # 🔥 CLEAR BUTTON
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            # Clear LangChain memory
            st.session_state.memory.clear()

        memory_container = st.container()

        with memory_container:
            for message in st.session_state.memory.chat_memory.messages:
                if message.type == "human":
                    st.markdown(
                        f"<div class='memory-text'><b>🧑 You:</b><br>{message.content}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='memory-text'><b>🗿 AI:</b><br>{message.content}</div>",
                        unsafe_allow_html=True
                    )