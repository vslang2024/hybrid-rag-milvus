# Chat server image. Talks to Milvus Standalone via MILVUS_URI (set in
# docker-compose.yml); nothing Milvus-related is installed in this image
# beyond the pymilvus client.
FROM python:3.11-slim

WORKDIR /app

# Install deps first so code changes don't bust the pip cache layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY static/ static/
COPY data/ data/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
