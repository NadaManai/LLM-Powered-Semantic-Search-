import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

CHUNK_JSON = "data/chunks/bourse_chunks.json"
OUTPUT_DIR = "data/faiss_index"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load chunks
with open(CHUNK_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)
    
print(f"Number of chunks loaded: {len(chunks)}")
print("Sample chunk:", chunks[0])

texts = [c["text"] for c in chunks]

print(f"Loaded {len(texts)} chunks for embedding")

# load multilingual SBERT model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# generate embeddings
embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# normalize embeddings (cosine similarity)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# create FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  
index.add(embeddings)

print(f"FAISS index created with {index.ntotal} vectors")

# save index + metadata
faiss.write_index(index, os.path.join(OUTPUT_DIR, "bourse_index.faiss"))
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("FAISS index and metadata saved to", OUTPUT_DIR)
