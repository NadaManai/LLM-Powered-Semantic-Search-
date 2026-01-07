import os
from PyPDF2 import PdfReader

PDF_DIR = "data/raw/pdfs"

documents = []

for filename in os.listdir(PDF_DIR):
    if filename.endswith(".pdf"):
        filepath = os.path.join(PDF_DIR, filename)
        reader = PdfReader(filepath)

        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

        documents.append({
            "source": filename,
            "text": full_text
        })

print(f"Loaded {len(documents)} documents")
print(documents[0]["text"][:1000])  
