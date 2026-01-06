import streamlit as st

# ⚡ Must be the VERY FIRST Streamlit command - before ANY other code
st.set_page_config(layout="wide")

from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
import torch

# -----------------------------
# Utility functions
# -----------------------------
def read_policy(policy_file):
    """Read PDF or TXT file."""
    if policy_file.name.endswith(".pdf"):
        reader = PdfReader(policy_file)
        return "\n".join(page.extract_text() for page in reader.pages)
    else:
        return policy_file.read().decode("utf-8", errors="ignore")

def chunk_text(text):
    """Split text into chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

# -----------------------------
# Caching
# -----------------------------
@st.cache_data
def build_vectorstore(chunks):
    """Build FAISS vectorstore from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

@st.cache_resource(show_spinner="Loading LLM (this may take a few minutes)")
def load_llm():
    """Load the HuggingFace LLM with caching."""
    model_name = "microsoft/phi-2"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"}
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    return HuggingFacePipeline(pipeline=pipe)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("SLM + RAG Pipeline")

left_col, right_col = st.columns(2)

with left_col:
    st.header("Upload Policy File")
    uploaded_file = st.file_uploader(
        "Drag and Drop the Policy File",
        type=["pdf", "txt"]
    )

    if uploaded_file:
        with st.spinner("Processing Policy..."):
            text = read_policy(uploaded_file)
            chunks = chunk_text(text)
            vectorstore = build_vectorstore(chunks)
            st.success("Policy indexed successfully!")

        st.session_state["vectorstore"] = vectorstore

with right_col:
    st.header("Ask Questions")
    if "vectorstore" not in st.session_state:
        st.info("Upload a policy file to start querying")
    else:
        # Load LLM only when needed (lazy loading)
        llm = load_llm()
        
        retriever = st.session_state["vectorstore"].as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        query = st.text_input("Enter your query")
        if query:
            with st.spinner("Processing..."):
                response = qa_chain.invoke({"query": query})
                answer = response["result"]
                sources = response["source_documents"]

            st.subheader("Answer")
            st.write(answer)

            if sources:
                st.subheader("Source Chunks")
                for doc in sources:
                    st.write(doc.page_content)