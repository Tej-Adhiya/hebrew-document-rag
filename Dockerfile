
FROM python:3.12-slim
WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install  -r requirements.txt

EXPOSE 7860

CMD ["python", "app.py"]
