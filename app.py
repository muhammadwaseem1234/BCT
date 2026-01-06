import streamlit as st

st.set_page_config(layout="wide")

from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch

def read_policy(policy_file):
    if policy_file.name.endswith(".pdf"):
        reader = PdfReader(policy_file)
        return "\n".join(page.extract_text() for page in reader.pages)
    else:
        return policy_file.read().decode("utf-8", errors="ignore")

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

@st.cache_data
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

@st.cache_resource(show_spinner="Loading LLM (this may take a few minutes)")
def load_llm():
    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
        max_new_tokens=256,  # Increased from 128
        temperature=0.3,      # Slightly increased for better generation
        do_sample=True,       # Changed to True for better quality
        repetition_penalty=1.2,  # Prevent repetition
        pad_token_id=tokenizer.eos_token_id
    )
    return HuggingFacePipeline(pipeline=pipe)

# Custom prompt template optimized for Phi-2
prompt_template = """Use the following context to answer the question. Extract relevant information directly from the context.

Context:
{context}

Question: {question}

Answer based on the context above:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

st.title("SLM + RAG Pipeline")

left_col, right_col = st.columns(2)

with left_col:
    st.header("Upload Policy File")
    uploaded_file = st.file_uploader("Drag and Drop the Policy File", type=["pdf", "txt"])
    
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
        llm = load_llm()
        retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 3})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        query = st.text_input("Enter your query")
        if query:
            with st.spinner("Processing..."):
                response = qa_chain.invoke({"query": query})
                answer = response["result"]
                sources = response["source_documents"]
            
            st.subheader("Answer")
            # Clean up the answer (remove the prompt echo if present)
            if "Answer based on the context above:" in answer:
                answer = answer.split("Answer based on the context above:")[-1].strip()
            st.write(answer)
            
            if sources:
                with st.expander("📄 View Source Chunks"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc.page_content)
                        st.divider()