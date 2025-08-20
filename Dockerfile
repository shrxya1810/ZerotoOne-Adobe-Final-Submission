# Multi-stage Docker build for PDF Research Canvas Application
FROM node:20-alpine AS frontend-builder

# Set working directory for frontend
WORKDIR /app/frontend

# Copy frontend package files
COPY Connect-the-Dots-clean-main/Connect-the-Dots-clean-main/package*.json ./

# Clear npm cache and install frontend dependencies
RUN npm cache clean --force && npm install

# Copy frontend source code
COPY Connect-the-Dots-clean-main/Connect-the-Dots-clean-main/ ./

# Set build-time environment variables for Vite
ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

# Don't build frontend - we'll run dev server

# Final production stage
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS production

# Install system dependencies and Node.js for serving frontend
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy backend requirements and install packages
COPY unified-doc-intelligence/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY unified-doc-intelligence/ ./backend/

# Copy frontend source code for dev server
COPY --from=frontend-builder /app/frontend ./frontend/

# Create data and temp directories
RUN mkdir -p /app/backend/data/faiss_data \
    && mkdir -p /app/backend/data/temp \
    && mkdir -p /app/backend/data/uploads

# Set environment variables
ENV PYTHONPATH="/app/backend"
ENV NODE_ENV=development

# Set environment variables for evaluation (can be overridden at runtime with -e)
ENV ADOBE_EMBED_API_KEY=""
ENV LLM_PROVIDER="gemini"
ENV GOOGLE_APPLICATION_CREDENTIALS=""
ENV GEMINI_MODEL="gemini-2.5-flash"
ENV GEMINI_API_KEY=""
ENV TTS_PROVIDER="azure"
ENV AZURE_TTS_KEY=""
ENV AZURE_TTS_ENDPOINT=""

# Map to legacy variable names for frontend compatibility  
# Note: This mapping doesn't work in Docker ENV, handled in startup script instead

# Create simple startup script
RUN echo '#!/bin/bash' > /app/start.sh && \
    echo 'set -e' >> /app/start.sh && \
    echo 'echo "=== API Configuration Check ==="' >> /app/start.sh && \
    echo 'echo "Adobe Embed API: ${ADOBE_EMBED_API_KEY:+✅ Present}${ADOBE_EMBED_API_KEY:-❌ Missing}"' >> /app/start.sh && \
    echo 'echo "Gemini API: ${GEMINI_API_KEY:+✅ Present}${GEMINI_API_KEY:-❌ Missing}"' >> /app/start.sh && \
    echo 'echo "Azure TTS Key: ${AZURE_TTS_KEY:+✅ Present}${AZURE_TTS_KEY:-❌ Missing}"' >> /app/start.sh && \
    echo 'echo "Azure TTS Endpoint: ${AZURE_TTS_ENDPOINT:+✅ Present}${AZURE_TTS_ENDPOINT:-❌ Missing}"' >> /app/start.sh && \
    echo 'echo "================================"' >> /app/start.sh && \
    echo 'echo "Setting frontend environment variables..."' >> /app/start.sh && \
    echo 'export VITE_ADOBE_CLIENT_ID="$ADOBE_EMBED_API_KEY"' >> /app/start.sh && \
    echo 'echo "VITE_ADOBE_CLIENT_ID set to: $VITE_ADOBE_CLIENT_ID"' >> /app/start.sh && \
    echo 'echo "Starting FastAPI backend..."' >> /app/start.sh && \
    echo 'cd /app/backend && python start_backend.py &' >> /app/start.sh && \
    echo 'BACKEND_PID=$!' >> /app/start.sh && \
    echo 'echo "Starting frontend dev server..."' >> /app/start.sh && \
    echo 'cd /app/frontend && VITE_ADOBE_CLIENT_ID="$ADOBE_EMBED_API_KEY" npm cache clean --force && npm install && npm run dev &' >> /app/start.sh && \
    echo 'FRONTEND_PID=$!' >> /app/start.sh && \
    echo 'wait $BACKEND_PID $FRONTEND_PID' >> /app/start.sh

RUN chmod +x /app/start.sh

# No need for serve - using npm run dev

# Expose ports for frontend (8080) and backend (8000)
EXPOSE 8080 8000

# Health check - check both frontend and backend
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ && curl -f http://localhost:8000/health || exit 1

# Start both frontend and backend
CMD ["/app/start.sh"]