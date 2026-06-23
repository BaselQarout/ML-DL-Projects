import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    # IndexFlatL2 measures exact Euclidean distance (L2 distance)
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def retrieve_top_k(query, model, index, chunks, k=2):
    # Embed the incoming user query using the same transformer model
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, k)
    
    # Map indices back to the original text chunks
    results = []
    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results