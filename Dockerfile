FROM python:3.11-bookworm


# libzbar を含む必要なシステム依存をインストール
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean

# 作業ディレクトリの設定
WORKDIR /app

# ソースコードをコンテナにコピー
COPY . .

# 依存関係をインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリ起動
CMD ["python", "ocr_api_server.py"]
