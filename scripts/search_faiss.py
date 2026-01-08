import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/faiss_index/bourse_index.faiss"
META_PATH = "data/faiss_index/metadata.json"

# load index and metadata
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Index size:", index.ntotal)

# load same model (CRITICAL)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def search(query, k=5):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    scores, indices = index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),
            "text": metadata[idx]["text"]
        })
    return results

# test
results = search("marché boursier tunisien", k=5)
for r in results:
    print("\nScore:", r["score"])
    print(r["text"][:300])
