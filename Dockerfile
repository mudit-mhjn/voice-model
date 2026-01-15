# ---------- builder ----------
FROM python:3.10-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---------- runtime ----------
FROM python:3.10-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=builder /install /usr/local
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0","--port", "8000","--app-dir", "/app"]
