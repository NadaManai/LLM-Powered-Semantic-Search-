from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
import json
import faiss
import numpy as np

# ----------------------------
# load FAISS index + metadata
# ----------------------------
OUTPUT_DIR = "data/faiss_index"
TOP_K = 5

index = faiss.read_index(os.path.join(OUTPUT_DIR, "bourse_index.faiss"))
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ----------------------------
# Load sentence transformer
# ----------------------------
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ----------------------------
# Load 8-bit Falcon 7B
# ----------------------------
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",       
    load_in_8bit=True,        # 8-bit quantization
    torch_dtype=torch.float16,
    trust_remote_code=True  
)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,       
    temperature=0.2,
)

# ----------------------------
# functions
# ----------------------------
def search(query, top_k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "score": float(score),
            "text": metadata[idx]["text"],
            "source": metadata[idx]["source"]
        })
    return results

def build_prompt(question, contexts):
    context_text = "\n\n".join([f"- {c['text']}" for c in contexts])
    prompt = f"""
Tu es un assistant expert en finance et marchés boursiers.

Réponds à la question en te basant UNIQUEMENT sur les informations ci-dessous.
Si l'information n'est pas présente, dis clairement que tu ne sais pas.

Contexte :
{context_text}

Question :
{question}

Réponse :
"""
    return prompt.strip()

def answer_question(question):
    contexts = search(question)
    prompt = build_prompt(question, contexts)
    response = llm(prompt)[0]["generated_text"]
    return response, contexts

# ----------------------------
# test
# ----------------------------
question = "quelles sont les meilleures entreprises en 2024?"
answer, sources = answer_question(question)

print("Answer:\n", answer)
print("\nSources:")
for s in sources:
    print(f"- {s['source']} (score: {s['score']:.3f})")
