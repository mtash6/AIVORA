# ==============================================================================
# AIVORA OS - Docker Build Configuration
# ==============================================================================

# 1. Base Image: Use official lightweight Python 3.12
FROM python:3.12-slim

# 2. Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Install OS-level dependencies
# ffmpeg is required for Whisper audio transcription
# build-essential & git are required for compiling certain Python modules
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Set container working directory
WORKDIR /app

# 5. Copy and install Python dependencies first (optimizes Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy application code into container
COPY . .

# 7. Create output folders for persistent local databases, backups, and temporary files
RUN mkdir -p outputs/backups outputs/temp

# 8. Expose Streamlit default port
EXPOSE 8501

# 9. Health check to monitor Streamlit container status
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 10. Startup command to launch AIVORA OS
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]