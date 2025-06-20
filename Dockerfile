# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements file
COPY src/requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code and model directory structure
COPY src/ ./src/
COPY model/ ./model/

# Expose the port the app runs on
EXPOSE 5000

# Set the model directory environment variable
ENV MODEL_DIR=./model

# Command to run the application
CMD ["python", "-u", "src/app.py"]
