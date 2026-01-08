import os
import json
from clean_text import load_and_clean_pdfs
from collections import defaultdict

# -------- CONFIG --------
PDF_DIR = "data/raw/pdfs"
OUTPUT_DIR = "data/chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default chunk settings
DEFAULT_CHUNK_SIZE = 150    # words per chunk
DEFAULT_CHUNK_OVERLAP = 50  # words overlapping between chunks
AUTO_CHUNK = True           # if True, adjust chunk size based on document length

def split_into_chunks(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def get_dynamic_chunk_size(text):
    """Optional: calculate chunk size based on text length (longer text → bigger chunks)."""
    word_count = len(text.split())
    chunk_size = max(DEFAULT_CHUNK_SIZE, min(300, word_count // 10))
    overlap = int(chunk_size * 0.3)
    return chunk_size, overlap

# -------- MAIN --------
if __name__ == "__main__":
    # Load all lines
    docs = load_and_clean_pdfs(PDF_DIR)

    # Group lines by PDF source and language
    grouped_docs = defaultdict(list)
    for doc in docs:
        key = (doc["source"], doc["lang"])
        grouped_docs[key].append(doc["text"])

    all_chunks = []

    # Chunk the merged text per document
    for (source, lang), texts in grouped_docs.items():
        full_text = " ".join(texts)

        # Decide chunk size
        if AUTO_CHUNK:
            chunk_size, overlap = get_dynamic_chunk_size(full_text)
        else:
            chunk_size, overlap = DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

        chunks = split_into_chunks(full_text, chunk_size=chunk_size, overlap=overlap)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": source,
                "chunk_index": i,
                "lang": lang,
                "text": chunk
            })

    # Save chunks
    output_file = os.path.join(OUTPUT_DIR, "bourse_chunks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Created {len(all_chunks)} chunks from {len(grouped_docs)} PDFs")
    print(f"Saved to {output_file}")
