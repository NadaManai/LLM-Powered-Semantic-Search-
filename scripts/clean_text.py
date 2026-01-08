import os
import pdfplumber
import re

PDF_DIR = "data/raw/pdfs"
OUTPUT_DIR = "data/clean_text"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_paragraph(text: str) -> str:
    """Cleans a paragraph from unwanted characters and hyphenation."""
    text = re.sub(r'-\n', '', text)        # fix hyphenation
    text = re.sub(r'\s+', ' ', text)       # normalize whitespace
    text = re.sub(r'\s([.,;:%])', r'\1', text)  # fix space before punctuation
    return text.strip()

def load_and_clean_pdfs(pdf_dir: str):
    """Load PDFs and separate French/English based on text color (if available)."""
    documents = []

    for filename in os.listdir(pdf_dir):
        if not filename.endswith(".pdf"):
            continue

        filepath = os.path.join(pdf_dir, filename)
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                for char in page.chars:
                    # make the diffrence between the two languages based on color
                    color = char.get('non_stroking_color', (0, 0, 0))
                    if sum(color) < 1:
                        lang = 'fr'
                    else:
                        lang = 'en'
                    char['lang'] = lang

                lines_dict = {}
                for char in page.chars:
                    y = round(char['top'])
                    lines_dict.setdefault(y, []).append(char)
                
                lines = []
                for y in sorted(lines_dict.keys(), reverse=True): 
                    line_chars = lines_dict[y]
                    line_text = ''.join(c['text'] for c in line_chars)
                    fr_count = sum(1 for c in line_chars if c['lang']=='fr')
                    en_count = sum(1 for c in line_chars if c['lang']=='en')
                    line_lang = 'fr' if fr_count >= en_count else 'en'
                    line_text = clean_paragraph(line_text)
                    if len(line_text) > 20:
                        lines.append({
                            'text': line_text,
                            'lang': line_lang
                        })
                
                documents.extend([{'source': filename, **line} for line in lines])

    return documents


if __name__ == "__main__":
    docs = load_and_clean_pdfs(PDF_DIR)
    print(f"Loaded {len(docs)} lines from {len(set(d['source'] for d in docs))} PDFs\n")
    
    print("--- SAMPLE LINES ---\n")
    for d in docs[:10]:
        print(f"[{d['lang']}] {d['text'][:300]}...\n")
