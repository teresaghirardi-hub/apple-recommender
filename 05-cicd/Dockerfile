FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python 02-experiment-tracking/train.py

EXPOSE 8000

CMD ["uvicorn", "03-deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]