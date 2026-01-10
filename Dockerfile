FROM python:3.10-slim

# Install system dependencies (updated for newer Debian)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/videos/input data/videos/output \
    data/calibration/images data/results/tracked_points \
    data/results/poses data/results/bbox_3d

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "scripts/process_video.py", "--help"]