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
    # Using a more reliable model for QA tasks
    model_name = "microsoft/phi-2"  # Better for question answering
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.info(f"🖥️ Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    # Proper generation config to prevent gibberish
    pipe = pipeline(
        "text2text-generation",  # Changed from text-generation
        model=model,
        tokenizer=tokenizer,
        max_length=512,  # Changed from max_new_tokens
        min_length=20,   # Ensure minimum quality response
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,  # Prevents repeating phrases
        early_stopping=True,
        num_beams=4  # Beam search for better quality
    )
    
    return HuggingFacePipeline(pipeline=pipe)

# Optimized prompt template
prompt_template = """Answer the question based on the context below. Be specific and concise.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

st.title("🚀 SLM + RAG Pipeline")

# Display device status
if torch.cuda.is_available():
    st.success("✅ GPU Detected - Fast Mode")
else:
    st.info("💻 Running on CPU")

left_col, right_col = st.columns(2)

with left_col:
    st.header("Upload Policy File")
    uploaded_file = st.file_uploader("Drag and Drop the Policy File", type=["pdf", "txt"])
    
    if uploaded_file:
        with st.spinner("Processing Policy..."):
            text = read_policy(uploaded_file)
            chunks = chunk_text(text)
            vectorstore = build_vectorstore(chunks)
            st.success("✅ Policy indexed successfully!")
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
            with st.spinner("Generating answer..."):
                try:
                    response = qa_chain.invoke({"query": query})
                    answer = response["result"]
                    sources = response["source_documents"]
                    
                    # Clean the answer
                    if "Answer:" in answer:
                        answer = answer.split("Answer:")[-1].strip()
                    
                    # Remove any remaining prompt artifacts
                    answer = answer.replace("Context:", "").replace("Question:", "").strip()
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    answer = "Unable to generate response. Please try again."
                    sources = []
            
            st.subheader("Answer")
            st.write(answer)
            
            if sources:
                with st.expander("📄 View Source Chunks"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc.page_content)
                        st.divider()