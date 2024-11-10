# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set environment variables to ensure proper handling of dependencies and performance
ENV PYTHONUNBUFFERED 1
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app/

# Install required Python packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download NLTK stopwords
RUN python -m nltk.downloader stopwords
RUN python -c "import nltk; nltk.download('wordnet')"  # Download WordNet corpus

# Expose the FastAPI app port
EXPOSE 8001

# Command to run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4", "--loop", "asyncio"]
