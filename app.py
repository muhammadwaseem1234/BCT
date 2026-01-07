import streamlit as st
st.set_page_config(layout="wide")

from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch

# ------------------------
# Utility functions
# ------------------------

def read_policy(policy_file):
    if policy_file.name.endswith(".pdf"):
        reader = PdfReader(policy_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        return policy_file.read().decode("utf-8", errors="ignore")

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    return splitter.split_text(text)

@st.cache_data
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)

# ------------------------
# Load SMALL LLM (CPU safe)
# ------------------------

@st.cache_resource(show_spinner="Loading small LLM (CPU-safe)...")
def load_llm():
    model_name = "google/flan-t5-small"  # ✅ small & reliable

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.0,   # deterministic QA
        do_sample=False
    )

    return HuggingFacePipeline(pipeline=pipe)

# ------------------------
# Prompt
# ------------------------

PROMPT = PromptTemplate(
    template="""
Answer the question using ONLY the context below.
If the answer is not in the context, say "Not found in document".

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# ------------------------
# UI
# ------------------------

st.title("📄 Lightweight RAG (VM-Compatible)")

st.info("💻 Running on CPU (optimized for low-RAM VM)")

left, right = st.columns(2)

with left:
    st.header("Upload Policy File")
    uploaded_file = st.file_uploader(
        "Upload PDF or TXT",
        type=["pdf", "txt"]
    )

    if uploaded_file:
        with st.spinner("Indexing document..."):
            text = read_policy(uploaded_file)
            chunks = chunk_text(text)
            vectorstore = build_vectorstore(chunks)
            st.session_state["vectorstore"] = vectorstore
        st.success("✅ Document indexed")

with right:
    st.header("Ask Questions")

    if "vectorstore" not in st.session_state:
        st.info("Upload a document to begin")
    else:
        llm = load_llm()
        retriever = st.session_state["vectorstore"].as_retriever(
            search_kwargs={"k": 3}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        query = st.text_input("Enter your question")

        if query:
            with st.spinner("Answering..."):
                result = qa_chain.invoke({"query": query})

            st.subheader("Answer")
            st.write(result["result"])

            with st.expander("Source Chunks"):
                for i, doc in enumerate(result["source_documents"], 1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(doc.page_content)
