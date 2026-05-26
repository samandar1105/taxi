# Use an official lightweight Python image
FROM python:3.10-slim

# Set a working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your model artifacts and code into the container
COPY final_model.pkl .
COPY final_feature_names.pkl .
COPY inference.py .
COPY app.py .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the web server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]