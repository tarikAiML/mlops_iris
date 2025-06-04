# ------------ BASE PYTHON ----------------
FROM python:3.11-slim

# ------------ ENV VARS ----------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ------------ DEPENDENCIES SYSTEME ------------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ------------ WORKDIR ------------
WORKDIR /app

# ------------ INSTALL DEPENDENCIES ------------
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# ------------ COPY PROJECT FILES ------------
COPY . .

# ------------ DEFAULT CMD ------------
CMD ["bash"]
