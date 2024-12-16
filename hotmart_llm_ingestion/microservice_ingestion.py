import os
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Batch

app = FastAPI(title="Microserviço de Ingestão")

# Nome do host e porta do Qdrant (definimos no docker-compose)
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Nome da collection no Qdrant (onde vamos salvar os vetores)
COLLECTION_NAME = "hotmart_knowledge"

# Inicializa o cliente do Qdrant (ele vai conectar no contêiner do Qdrant)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Carrega o modelo de embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.on_event("startup")
def startup_event():
    """
    Ao iniciar o servidor FastAPI, recriamos a 'collection' no Qdrant.
    Isso garante que ela exista e esteja limpa para receber dados.
    """
    vector_dim = 384  # Dimensão do modelo all-MiniLM-L6-v2
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' recriada com sucesso!")

def scrape_hotmart_blog(url: str) -> str:
    """
    Faz requisição HTTP e extrai o texto cru da página.
    """
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    # Remove script/style
    for tag in soup(["script", "style"]):
        tag.decompose()

    # Extrai texto
    text = soup.get_text(separator=" ")
    return text

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 20):
    """
    Quebra o texto em pedaços de 'chunk_size' palavras cada, 
    com sobreposição 'overlap' para evitar perder contexto.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap
        if start < 0:
            start = 0

    return chunks

@app.post("/ingest")
def ingest_content():
    """
    Endpoint que inicia o processo de ingestão:
      - Baixa o texto da página
      - Limpa, chunkar
      - Gera embeddings
      - Salva no Qdrant
    """
    url = "https://hotmart.com/pt-br/blog/como-funciona-hotmart"
    raw_text = scrape_hotmart_blog(url)

    # Limpeza simples (remover espaços e quebras múltiplas):
    raw_text = " ".join(raw_text.split())

    # Chunkar
    chunks = chunk_text(raw_text, chunk_size=200, overlap=20)

    # Gera embeddings de cada chunk
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)

    # Constrói payload para Qdrant
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        points.append((i, embedding.tolist(), {"text": chunk}))

    # Upsert no Qdrant
    ids, vectors, payloads = zip(*points)
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=Batch(ids=ids, vectors=vectors, payloads=payloads)
    )

    return JSONResponse(
        content={"message": "Ingestão concluída!", "num_chunks": len(chunks)}
    )
