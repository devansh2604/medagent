# MedAgent — container image for Hugging Face Spaces (Docker SDK)
#
# Spaces runs the container as UID 1000 and expects the app on port 7860.
# Everything the app writes (chroma/, lab_store.json, session_history.json)
# must therefore be owned by that user.

FROM python:3.11-slim

# opencv-python-headless still needs a couple of shared libraries, and
# onnxruntime needs libgomp for its thread pool.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces executes as UID 1000; create that user so file ownership lines up.
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Dependencies first, so code changes do not invalidate the layer.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Bake the ONNX embedding model into the image. Without this the first
# semantic search on a fresh container downloads 79MB before it can answer.
RUN python -c "\
import chromadb; \
c = chromadb.EphemeralClient().get_or_create_collection('warmup'); \
c.add(ids=['1'], documents=['warm up the embedding model']); \
print('embedding model cached into image')"

COPY --chown=user . ./

# Demo defaults. OPENAI_API_KEY is NOT set here — add it as a Space secret
# so the key never lives in the image or the repository.
ENV PORT=7860 \
    DEMO_MODE=true \
    RATE_LIMIT_PER_HOUR=30 \
    SEED_ON_BOOT=0

EXPOSE 7860

# One worker keeps the in-memory rate limiter exact. Spaces gives 2 vCPUs,
# so four threads are comfortable here. The long timeout allows multi-step
# agent loops that make several OpenAI calls.
CMD ["gunicorn", "server:app", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "180", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
