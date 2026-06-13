# ==========================================
# ICHeritage Backend - ULTIMATE STANDALONE & SPEED OPTIMIZED
# Combined with requirements-cpu.txt and .dockerignore
# ==========================================

# ============ Stage 1: Builder ============
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Tận dụng file cấu hình CPU của bạn
COPY requirements-cpu.txt .
RUN pip install --upgrade pip && \
    # Cài thêm lõi openvino để kích hoạt tính năng tăng tốc dự đoán CPU
    pip install openvino openvino-telemetry && \
    pip install -r requirements-cpu.txt

# Hút mỡ triệt để Virtual Environment
RUN find /opt/venv -type d -name "tests" -exec rm -rf {} + && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + && \
    find /opt/venv -name "*.pyc" -delete

# ============ Stage 2: Runtime ============
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Nhận venv siêu sạch từ stage builder
COPY --from=builder /opt/venv /opt/venv

# Copy mã nguồn (Nhờ .dockerignore, các file rác và file media nặng sẽ bị tự động bỏ qua)
COPY app.py config.py constraintsDB.py history_store.py models.py services.py ./

# Nhúng trực tiếp mô hình đã xuất dạng OpenVINO vào Image để chạy độc lập không cần Volume trên Hosting
COPY weight/best_openvino_model/ /app/weight/best_openvino_model/

# Khởi tạo sẵn cấu trúc thư mục uploads sạch
RUN mkdir -p uploads/artifacts && chmod -R 755 /app

RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/api/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--threads", "2", "--timeout", "300", "app:app"]