import os
import json
from clean_text import load_and_clean_pdfs  

CHUNK_SIZE = 150   
CHUNK_OVERLAP = 50 
PDF_DIR = "data/raw/pdfs"
OUTPUT_DIR = "data/chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    docs = load_and_clean_pdfs(PDF_DIR)
    all_chunks = []

    for idx, doc in enumerate(docs):
        # if doc["lang"] != "fr": continue  
        chunks = split_into_chunks(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "paragraph_index": idx,
                "chunk_index": i,
                "lang": doc["lang"],  
                "text": chunk
            })

    output_file = os.path.join(OUTPUT_DIR, "bourse_chunks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Created {len(all_chunks)} chunks and saved to {output_file}")
