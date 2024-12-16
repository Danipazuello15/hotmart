# Projeto Hotmart LLM

Este projeto utiliza **FastAPI**, **Qdrant** e **Sentence Transformers** para implementar dois microserviços:
- **Ingestão de Dados**: Coleta, processa e armazena texto em forma de embeddings no Qdrant.
- **Perguntas e Respostas (QA)**: Recupera embeddings relevantes e gera respostas baseadas no modelo Flan-T5.

---

## **Tecnologias Utilizadas**
- **FastAPI**: Framework para construir APIs.
- **Docker** e **Docker Compose**: Gerenciamento dos containers.
- **Qdrant**: Banco de dados vetorial.
- **Sentence Transformers**: Geração de embeddings.
- **Hugging Face Transformers**: Modelo de LLM (Flan-T5).
- **BeautifulSoup4**: Scraping de conteúdo web.

---

## **Como Rodar o Projeto**

1. **Clone o Repositório**:
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd hotmart
