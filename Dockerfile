# Faz 1 — TrOCR el yazısı tanıma container'ı (CPU)
# 4 çekirdek / 8 GB RAM VPS için ayarlandı; sürümler tekrarlanabilirlik
# için sabitlendi (benchmarking.md'ye imaj sürümü not edilecek).

FROM python:3.11-slim

# Görüntü işleme için gerekli sistem kütüphaneleri
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU (CUDA'sız — imaj ~1.5 GB yerine ~700 MB kalır)
RUN pip install --no-cache-dir \
        torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
        transformers==4.44.2 \
        pillow==10.4.0 \
        sentencepiece==0.2.0

WORKDIR /app
COPY trocr_calistir.py /app/

# Model ağırlıkları volume'a önbelleklenir (her kurulumda tekrar inmesin)
ENV HF_HOME=/models
# 4 çekirdeği tam kullan
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

ENTRYPOINT ["python3", "/app/trocr_calistir.py"]
