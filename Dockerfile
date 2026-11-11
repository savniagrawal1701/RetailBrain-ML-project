FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Re-train inside the image so the saved models always match the installed library versions (~30 s).
RUN python train.py

EXPOSE 8000
CMD ["sh", "-c", "gunicorn app:app --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-8000}"]
