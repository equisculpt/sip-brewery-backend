# Dockerfile for Institutional-Grade Finance Engine + API
FROM python:3.11-slim

WORKDIR /app

# System dependencies for Playwright, sentence-transformers, and bs4
RUN apt-get update && \
    apt-get install -y wget curl git build-essential libglib2.0-0 libnss3 libgconf-2-4 libfontconfig1 libxss1 libasound2 libxtst6 libxrandr2 libgtk-3-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m playwright install --with-deps

# Copy code
COPY . .

EXPOSE 8081

CMD ["uvicorn", "asi_finance_engine_api:app", "--host", "0.0.0.0", "--port", "8081"]
