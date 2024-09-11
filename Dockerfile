# Use an official Python runtime as a parent image
FROM python:3.12

LABEL maintainer="Khizar Ahmad"
# Set the working directory in the container
WORKDIR /code
# Install system dependencies required for potential Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy the current directory contents into the container at /code
COPY . /code/

# Configuration to avoid creating virtual environments inside the Docker container
RUN poetry config virtualenvs.create false

# Install dependencies including development ones
RUN poetry install


EXPOSE  8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# Run the app. CMD can be overridden when starting the container

ENTRYPOINT ["streamlit", "run", "rag/app.py", "--server.port=8501", "--server.address=0.0.0.0"]