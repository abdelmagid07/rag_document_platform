# Build dependencies
FROM python:3.11-slim as builder
WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime Image
FROM python:3.11-slim
WORKDIR /app

# Copy deps
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy src code
COPY src/ ./src/

# ENVS
ENV PORT=8080
ENV HOST=0.0.0.0
EXPOSE 8080

# Command to run 
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]