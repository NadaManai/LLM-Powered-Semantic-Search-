import os
from PyPDF2 import PdfReader
from pathlib import Path
import re

PDF_DIR = "data/raw/pdfs"
OUTPUT_DIR = "data/clean_text"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_paragraph(text: str) -> str:
    """Cleans a paragraph from unwanted characters and hyphenation."""
    text = re.sub(r'-\n', '', text)       
    text = re.sub(r'\s+', ' ', text)      # normalize whitespace
    return text.strip()

def load_and_clean_pdfs(pdf_dir: str):
    """Load PDFs and split paragraphs into French and English based on position."""
    documents = []

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            reader = PdfReader(filepath)

            raw_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    raw_text += page_text + "\n\n"

            # split text into paragraphs
            paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
            for idx, p in enumerate(paragraphs):
                cleaned = clean_paragraph(p)
                if len(cleaned) < 20:
                    continue

                lang = "fr" if idx % 2 == 0 else "en"
                documents.append({
                    "source": filename,
                    "lang": lang,
                    "text": cleaned
                })

    return documents

if __name__ == "__main__":
    docs = load_and_clean_pdfs(PDF_DIR)
    print(f"Loaded {len(docs)} paragraphs from {len(set(d['source'] for d in docs))} PDFs\n")

    print("--- SAMPLE PARAGRAPHS ---\n")
    for d in docs[:10]:
        print(f"[{d['lang']}] {d['text'][:300]}...\n")
