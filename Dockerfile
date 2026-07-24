FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Defaults to fully offline mock backends so `docker run` works with zero
# configuration. Override at runtime, e.g.:
#   docker run --env-file .env myimage "your question"
ENV LLM_PROVIDER=mock
ENV EMBEDDING_PROVIDER=mock

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["What does the Send API in LangGraph do?"]
