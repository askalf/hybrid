# hybrid — local-first LLM router. Stdlib only, so the image is python:slim + five files.
FROM python:3.12-slim
WORKDIR /app
COPY hybrid.py solver.py verify.py equations.py server.py ./

# 0.0.0.0 binds inside the container's namespace; publish the port deliberately
# (compose maps it to loopback) and set HYBRID_API_KEY if you expose it wider.
ENV HYBRID_HOST=0.0.0.0 \
    PORT=8080 \
    OLLAMA_URL=http://ollama:11434/api/generate
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=4)"

# stdout is the JSONL decision log; the banner goes to stderr
CMD ["python", "server.py"]
