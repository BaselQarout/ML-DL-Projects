import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

def load_and_chunk_pdfs(data_dir, chunk_size=500, overlap=50):
    chunks = []
    metadata = []
    
    for file in os.listdir(data_dir):
        if file.endswith('.pdf'):
            path = os.path.join(data_dir, file)
            reader = PdfReader(path)
            
            # Extract entire text from file
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            # Sliding window chunking
            start = 0
            while start < len(full_text):
                end = start + chunk_size
                chunk = full_text[start:end]
                chunks.append(chunk)
                metadata.append({"source": file, "start_idx": start})
                start += (chunk_size - overlap)
                
    return chunks, metadata

def generate_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    # This downloads a highly efficient 384-dimensional transformer model
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return model, embeddings