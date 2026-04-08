# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY --chown=user . .

# HF Spaces exposes port 7860
EXPOSE 7860

# Launch FastAPI — root "/" auto-redirects to "/docs"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
