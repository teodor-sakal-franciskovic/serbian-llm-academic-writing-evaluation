FROM python:3.13-slim

# Install system dependencies for pymupdf4llm (MuPDF)
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    poppler-utils \
    mupdf \
    mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
