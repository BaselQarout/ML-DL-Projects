import os
import pickle
import faiss
import certifi

# Fix potential SSL paths
os.environ["SSL_CERT_FILE"] = certifi.where()

from src.ingestion import load_and_chunk_pdfs, generate_embeddings
from src.retrieval import create_faiss_index

def run_one_time_ingestion():
    print("🔄 Step 1: Loading and chunking PDFs...")
    chunks, metadata = load_and_chunk_pdfs("data")
    
    print(f"🔄 Step 2: Generating text embeddings for {len(chunks)} chunks...")
    model, embeddings = generate_embeddings(chunks)
    
    print("🔄 Step 3: Creating FAISS Index map...")
    index = create_faiss_index(embeddings)
    
    # Create storage directory if it doesn't exist
    os.makedirs("storage", exist_ok=True)
    
    print("💾 Step 4: Saving FAISS index binary to disk...")
    # FAISS built-in function to save the C++ memory array to a file
    faiss.write_index(index, "storage/faiss_index.bin")
    
    print("💾 Step 5: Saving text chunks to disk...")
    # Pickle saves the standard Python list of text strings to a file
    with open("storage/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
        
    print("\n✅ Ingestion complete! 'storage/' folder is successfully packed.")

if __name__ == "__main__":
    run_one_time_ingestion()