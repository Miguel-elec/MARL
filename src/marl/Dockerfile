FROM python:3.10

WORKDIR /app

# DEPENDENCIAS DEL SISTEMA PARA SUMO
RUN apt-get update && apt-get install -y \
    sumo \
    sumo-tools \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*
    
# Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV SUMO_HOME=/usr/share/sumo
ENV PATH=$PATH:/usr/share/sumo/bin

CMD ["python", "main.py"]