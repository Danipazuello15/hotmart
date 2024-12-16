#!/bin/bash

echo "### Testando Ingestão de Dados no Endpoint /ingest ###"
curl -X POST http://localhost:8000/ingest
echo -e "\n"

echo "### Testando Perguntas e Respostas no Endpoint /ask ###"
curl -X POST http://localhost:8001/ask \
-H "Content-Type: application/json" \
-d '{"question": "O que é a Hotmart?"}'
echo -e "\n"

echo "### Testes Concluídos ###"
