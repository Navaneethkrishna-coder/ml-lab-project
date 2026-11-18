
# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and trained model
COPY app.py .
COPY model.pkl .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

