# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set a working directory inside the container
WORKDIR /app

# Copy requirements.txt (or poetry/conda equivalents) to the working directory
COPY requirements.txt /app/

# Install any needed packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code into the working directory
COPY . /app/

# Expose the port your app runs on (as defined in main.py, it’s 8001 by default)
EXPOSE 8001

# Command to run the app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]