FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for OpenCV / YOLO
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Expose the port your app will run on
EXPOSE 8080

# Run your app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
