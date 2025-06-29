# Streamlit RAG Compliance Demo Dockerfile
# Build with:  docker build -t vxdf-compliance-demo use_case/
# Run with:   docker run --env-file .env -p 8501:8501 vxdf-compliance-demo

FROM python:3.10-slim

# Install OS deps (for pdfplumber -> pdfminer.six -> cryptography etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements first for caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

EXPOSE 8501

# Streamlit launches on 8501 by default
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
