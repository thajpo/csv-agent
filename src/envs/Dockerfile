FROM python:3.11-slim

# Install system dependencies often needed for data science packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-install data science packages
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    statsmodels

# Command to keep container running
CMD ["tail", "-f", "/dev/null"]

