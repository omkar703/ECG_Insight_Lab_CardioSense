# ==============================================================================
# CardioSense / ECG Insight Lab — Production Container Dockerfile
# Full-Stack Streamlit & Scientific Python Backend
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8501 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install minimal OS dependencies for numerical computing and image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and verified model artifacts
COPY ecg_lab/ /app/ecg_lab/
COPY artifacts/ /app/artifacts/
COPY sample_data/ /app/sample_data/
COPY hardware/ /app/hardware/
COPY docs/ /app/docs/
COPY scripts/ /app/scripts/
COPY app.py /app/app.py
COPY README.md /app/README.md

# Create persistent mount points for full datasets
RUN mkdir -p /app/data/ptb-xl /app/data/mitdb /app/data/cache

# Expose standard Streamlit port
EXPOSE 8501

# Health check to ensure service readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start full-stack web application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
