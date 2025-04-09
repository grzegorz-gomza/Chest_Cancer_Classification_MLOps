# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim-bookworm

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install aws
RUN apt update -y && apt install awscli -y

# Install pip requirements
WORKDIR /app
COPY . /app

RUN mkdir -p /export/models && \
    curl -L -o /export/models/file.keras https://github.com/grzegorz-gomza/Chest_Cancer_Classification_MLOps/raw/refs/heads/main/export/model/model.keras?download=

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install wheel
RUN python -m pip install -r requirements.txt --no-cache-dir

CMD ["python3", "app.py"]

EXPOSE 1988
