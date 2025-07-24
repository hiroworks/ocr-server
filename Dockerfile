FROM python:3.11-bookworm

# OS依存パッケージ（ZBarが必要）
RUN apt-get update && apt-get install -y libzbar0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# ファイルをコピー
COPY . /app

# pipのアップグレードと依存関係インストール
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install paddleocr

# 起動コマンド
CMD ["python", "ocr_api_server.py"]
