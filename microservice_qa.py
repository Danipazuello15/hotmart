import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 📌 Inicializa a API
app = FastAPI(title="Q&A Microservice")

# 📌 Configura Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "hotmart_knowledge"
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# 📌 Carrega o modelo de embeddings e o modelo de LLM (Flan-T5)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# 📌 Define o formato da entrada
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def answer_question(request: QuestionRequest):
    """
    Recebe uma pergunta, busca os chunks relevantes no Qdrant,
    e gera uma resposta usando o modelo Flan-T5.
    """
    question = request.question

    # 🧩 1. Gera embedding da pergunta
    question_embedding = embedding_model.encode(question).tolist()

    # 🔍 2. Busca os Top-k chunks mais relevantes no Qdrant
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=question_embedding,
        limit=3  # Top-3 resultados
    )

    # 📝 3. Monta o contexto com os chunks encontrados
    context = "\n".join([hit.payload["text"] for hit in search_result])

    # 🔗 4. Prepara a entrada para o modelo Flan-T5
    prompt = f"Contexto: {context}\n\nPergunta: {question}\nResposta:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    
    # 🧠 5. Gera a resposta com o modelo Flan-T5
    output = flan_t5_model.generate(**inputs, max_length=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # 📤 6. Retorna a resposta gerada
    return {"question": question, "answer": response, "context_used": context}
