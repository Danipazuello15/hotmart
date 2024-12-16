# Base Image
FROM python:3.11-slim

# Set Work Directory
WORKDIR /app

# Install Dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy Ingestion Service
COPY hotmart_llm_ingestion/microservice_ingestion.py /app/

# Copy QA Service
COPY microservice_qa.py /app/

# Default Command for Testing
CMD ["uvicorn", "microservice_ingestion:app", "--host", "0.0.0.0", "--port", "8000"]

