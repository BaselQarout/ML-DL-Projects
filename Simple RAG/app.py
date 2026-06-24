import os
import pickle
import faiss
import certifi

# Fix potential SSL paths before loading libraries
os.environ["SSL_CERT_FILE"] = certifi.where()

import streamlit as st
import ollama
from sentence_transformers import SentenceTransformer
from src.retrieval import retrieve_top_k

st.set_page_config(page_title="Fictional Town Analysis", page_icon="🏙️")
st.title("🏙️ Fictional Town Analysis")

# ---------------------------------------------------------
# LOADING PRE-CALCULATED DATA (Instantaneous)
# ---------------------------------------------------------
@st.cache_resource
def load_saved_rag_resources():
    # 1. Load the pre-compiled FAISS geometric map
    index = faiss.read_index("storage/faiss_index.bin")
    
    # 2. Load the original text chunks list
    with open("storage/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
        
    # 3. Load the model (needed exclusively to convert incoming queries)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return model, index, chunks

# Check if the user ran the ingestion script first
if not os.path.exists("storage/faiss_index.bin"):
    st.error("⚠️ storage files not found! Please run 'python ingest_and_save.py' in your terminal first.")
    st.stop()

model, index, chunks = load_saved_rag_resources()

# ---------------------------------------------------------
# CHAT INTERACTION INTERFACE
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! The vector database files are loaded from disk. Ask me anything!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_query := st.chat_input("Ask about tactics or regulations..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Querying disk-loaded vector index..."):
            
            # Use the index to fetch matches directly
            retrieved_context = retrieve_top_k(user_query, model, index, chunks, k=2)
            context_str = "\n---\n".join(retrieved_context)
            
            prompt = f"""Use the following context to answer the question.
Context:
{context_str}

Question: {user_query}
Answer:"""
            
            response = ollama.generate(model='llama3', prompt=prompt)
            output_text = response['response']
            
            st.write(output_text)
            st.session_state.messages.append({"role": "assistant", "content": output_text})