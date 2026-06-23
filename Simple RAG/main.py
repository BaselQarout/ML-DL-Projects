import os
import certifi

# Force Python to use the valid, local certificate bundle from certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

import ollama
from src.ingestion import load_and_chunk_pdfs, generate_embeddings
from src.retrieval import create_faiss_index, retrieve_top_k

def main():
    print("Initializing Ingestion Phase...")
    chunks, metadata = load_and_chunk_pdfs("data")
    model, embeddings = generate_embeddings(chunks)
    
    print("Building Vector Index...")
    index = create_faiss_index(embeddings)
    
    # Let's run a test query on our special football files
    query = "Who works at Pine Stone Clinic?"
    print(f"\nUser Query: {query}")
    
    # Step 1: Retrieve matching context chunks
    retrieved_context = retrieve_top_k(query, model, index, chunks, k=2)
    context_str = "\n---\n".join(retrieved_context)
    
    # Step 2: Construct the augmented prompt
    prompt = f"""Use the following context to answer the user's question. 
If you do not know the answer based on the context, say that you don't know.

Context:
{context_str}

Question: {query}
Answer:"""

    print("\nSending Context and Prompt to LLM...")
    # Step 3: Generate the grounded answer using a local model
    response = ollama.generate(model='llama3', prompt=prompt)
    
    print("\n=== RAG System Response ===")
    print(response['response'])

if __name__ == "__main__":
    main()