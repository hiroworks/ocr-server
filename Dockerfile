FROM python:3.11-bookworm

# 必要なOSパッケージをインストール（ZBar, libGLなど）
RUN apt-get update && apt-get install -y \
    libzbar0 \
    libgl1 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを作成
WORKDIR /app

# プロジェクトファイルをコピー
COPY . /app

# pipと依存関係のインストール
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install paddleocr

# アプリケーション起動
CMD ["python", "ocr_api_server.py"]
