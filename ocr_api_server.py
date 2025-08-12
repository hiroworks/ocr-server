# ocr_api_server.py

# --- ライブラリの読み込み ---
import os
import cv2                                      # OpenCV。画像を読み込み加工
import re
import json
import time
import numpy as np
import threading
import requests
import sqlite3
from flask import Flask, request, jsonify       # APIサーバーを動かすためのWebフレームワーク
from pyzbar import pyzbar                       # バーコード読み取り用ライブラリ
from paddleocr import PaddleOCR                 # 文字認識のライブラリ（日本語対応）
from dotenv import load_dotenv                  # 秘密情報（APIキーなど）の安全な読み込み
from flask_cors import CORS                     # フロントエンドとの通信を許可
from geopy.distance import geodesic             # 距離計算（お店が近いか判定）

# --- 環境変数とディレクトリ準備 ---
load_dotenv()
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- .env 読み込み ---
load_dotenv()
RAKUTEN_APP_ID = os.getenv("RAKUTEN_APP_ID")

# カレントディレクトリの取得
cwd = os.getcwd()
print("=== カレントディレクトリ ===")
print(cwd)
print("=== /app の内容 ===")
print(os.listdir("/app"))
print("=== det ===")
print(os.listdir("/app/ch_PP-OCRv3_det_infer"))
print(os.listdir("./ch_PP-OCRv3_det_infer"))
print("=== rec ===")
print(os.listdir("/app/japan_PP-OCRv3_rec_infer"))
print(os.listdir("./japan_PP-OCRv3_rec_infer"))
print("=== cls ===")
print(os.listdir("/app/ch_ppocr_mobile_v2.0_cls_infer"))
print(os.listdir("./ch_ppocr_mobile_v2.0_cls_infer"))
# カレントディレクトリの中身を一覧表示
print("=== カレントディレクトリの内容 ===")
for item in os.listdir(cwd):
    print(item)


# --- DB(SQLite)準備 ---
# SQLite DB初期化
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "price_records.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# テーブルがなければ作成
cursor.execute('''
    CREATE TABLE IF NOT EXISTS prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_name TEXT,
        price INTEGER,
        shop_name TEXT,
        lat REAL,
        lon REAL,
        jan  TEXT,
        image_url  TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# --- Flask 初期化 外部（スマホアプリ）からの通信を許可 ---
t_flask = time.time()
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print(f"Flask初期化時間 : {time.time() - t_flask:.2f}秒")
print("[DEBUG] 使用中のDBファイル:", DB_PATH)


# === GeoJSON(店舗データ)ファイル読み込み（お店データ） ===
geojson = {"type": "FeatureCollection", "features": []}
geojson_path = os.path.join("data", "shops.geojson")

try:
    with open(geojson_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    if not raw.get("features") or not isinstance(raw["features"], list):
        raise ValueError("GeoJSONのfeaturesが存在しないか、配列ではありません。")

    for index, feature in enumerate(raw["features"]):
        try:
            geometry = feature.get("geometry")
            if geometry["type"] != "Point":
                continue
            coords = geometry.get("coordinates")
            if not (isinstance(coords, list) and len(coords) == 2 and all(isinstance(c, (int, float)) for c in coords)):
                continue
            geojson["features"].append(feature)
        except Exception as inner_err:
            print(f"⚠️ Feature index {index} をスキップ: {inner_err}")

    print(f"✅ 有効なPoint型店舗データ数: {len(geojson['features'])}")

except Exception as e:
    print("❌ shops.geojsonの読み込みエラー:", e)


# --- OCR 初期化 ---
print("🔍 OCR初期化開始")
t_ocr_init = time.time()
ocr = PaddleOCR(
    det_model_dir='/app/ch_PP-OCRv3_det_infer',   # 検出モデルフォルダ
    rec_model_dir='/app/japan_PP-OCRv3_rec_infer',   # 認識モデルフォルダ
    cls_model_dir='/app/ch_ppocr_mobile_v2.0_cls_infer',   # 角度分類モデル（use_angle_cls=True の場合でも指定可）
    use_angle_cls=True,
    lang='japan'
)
#        ocr = PaddleOCR(use_angle_cls=True, lang='japan')
print("🔍 OCR初期化完了")
print(f"OCR初期化時間: {time.time() - t_ocr_init:.2f}秒")


"""
# OCR初期化
print("🔍 OCR初期化開始")
t_ocr_init2 = time.time()
ocr = PaddleOCR(use_angle_cls=True, lang='japan')
print(f"OCR初期化時間: {time.time() - t_ocr_init2:.2f}秒")
"""

# バーコード処理（別スレッドで即実行）
def detect_barcode(img):
    print("🔍 バーコード検出開始")
    try:
        barcodes = pyzbar.decode(img)
        for barcode in barcodes:
            if barcode.type == 'EAN13':
                jan_code = barcode.data.decode('utf-8')
                print(f"📦 JANコード検出(by pyzbar): {jan_code}")
#                result = search_rakuten_product(jan_code, [])
                return jan_code  # ←JANコード認識結果を返す
#                return result  # ←楽天API結果を返す
        print("📦 JANコードは検出されませんでした。")
        return None
    except Exception as e:
        print(f"バーコード検出エラー: {e}")
        return None


"""
# --- OCR 初期化 ---
print("🔍 OCR初期化開始")
t_ocr_init = time.time()
ocr = PaddleOCR(use_angle_cls=True, lang='japan')
print(f"OCR初期化時間: {time.time() - t_ocr_init:.2f}秒")
"""

# OCR処理（別スレッドで即実行）
def run_ocr_logic(filename):
    try:
        # --- OCR 初期化 ---
#        print("🔍 OCR初期化開始")
#        t_ocr_init = time.time()
#        ocr = PaddleOCR(
#            det_model_dir='/app/ch_PP-OCRv3_det_infer',   # 検出モデルフォルダ
#            rec_model_dir='/app/ch_PP-OCRv3_rec_infer',   # 認識モデルフォルダ
#            cls_model_dir='/app/ch_PP-OCRv3_cls_infer',   # 角度分類モデル（use_angle_cls=True の場合でも指定可）
#            use_angle_cls=True,
#            lang='japan'
#        )
#        ocr = PaddleOCR(use_angle_cls=True, lang='japan')
#        print(f"OCR初期化時間: {time.time() - t_ocr_init:.2f}秒")

        img_path = os.path.join(UPLOAD_FOLDER, filename)
        output_dir = OUTPUT_FOLDER

        # 画像読み込み
        t_ocr_read = time.time()
        print("読み込み")  # リクエストが到達したことを確認
        image = cv2.imread(img_path)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        print(f"読み込み完了: {time.time() - t_ocr_read:.2f}秒")

        # グレースケール + シャープ化
        print("グレースケール＆シャープ化")  # リクエストが到達したことを確認
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpened = cv2.GaussianBlur(gray, (0, 0), 3)
        sharpened = cv2.addWeighted(gray, 1.5, sharpened, -0.5, 0)
        preprocessed_path = os.path.join(output_dir, "image_processed.jpg")
        cv2.imwrite(preprocessed_path, sharpened)

        # 赤色マスク
        def create_red_mask(hsv_img):
            print("赤色マスク")  # リクエストが到達したことを確認
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 70, 50])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
            return cv2.bitwise_or(mask1, mask2)

        # 価格抽出用の正規表現
        def split_text_price(text):
            print("価格抽出用の正規表現")  # リクエストが到達したことを確認
            pattern = re.compile(r'\d{2,4}(?:\.\d{1,2})?円')
            return pattern.findall(text)

        # ポリゴン分割
        def split_polygon_horizontally(poly):
            print("ポリゴン分割")  # リクエストが到達したことを確認
            poly = np.array(poly, dtype=np.float32)
            xs = poly[:, 0]
            sorted_xs = np.sort(xs)
            gaps = np.diff(sorted_xs)
            max_gap = np.max(gaps) if len(gaps) > 0 else 0
            img_w = image.shape[1]
            gap_threshold = img_w * 0.1
            if max_gap < gap_threshold:
                return [poly.astype(np.int32)]
            gap_idx = np.argmax(gaps)
            split_x = sorted_xs[gap_idx] + gaps[gap_idx] / 2
            left_points, right_points = [], []
            for p in poly:
                (left_points if p[0] <= split_x else right_points).append(p)
            def bounding_rect(points):
                points = np.array(points)
                x, y, w, h = cv2.boundingRect(points.astype(np.int32))
                return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
            rects = []
            if len(left_points) >= 2:
                rects.append(bounding_rect(left_points))
            if len(right_points) >= 2:
                rects.append(bounding_rect(right_points))
            if not rects:
                rects.append(poly.astype(np.int32))
            return rects

        # OCR開始
        t_ocr_start = time.time()
        print("OCR開始")  # リクエストが到達したことを確認
        paddle_text_lines = []
        image_with_polys = image.copy()
        image_with_split_polys = image.copy()
        image_with_polys_on_preprocessed = cv2.cvtColor(sharpened.copy(), cv2.COLOR_GRAY2BGR)
        candidate_regions = []

#        t_ocr_init2 = time.time()
#        ocr = PaddleOCR(use_angle_cls=True, lang='japan')
#        print(f"OCR初期化時間: {time.time() - t_ocr_init2:.2f}秒")
        result = ocr.predict(preprocessed_path)

        red_mask = create_red_mask(image_hsv)
        max_height = 1

        print("OCR処理１")  # リクエストが到達したことを確認
        for ocr_result in result:
            if 'rec_polys' in ocr_result:
                for poly in ocr_result['rec_polys']:
                    height = np.linalg.norm(np.array(poly[0]) - np.array(poly[3]))
                    if height > max_height:
                        max_height = height

        print("OCR処理２")  # リクエストが到達したことを確認
        all_text_polys = []
        for ocr_result in result:
            if 'rec_polys' in ocr_result and 'rec_texts' in ocr_result and 'rec_scores' in ocr_result:
                polys = ocr_result['rec_polys']
                texts = ocr_result['rec_texts']
                scores = ocr_result['rec_scores']
                for poly, txt, score in zip(polys, texts, scores):
                    poly_np = np.array(poly, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(poly_np)
                    all_text_polys.append({'text': txt, 'poly': poly_np, 'bbox': (x, y, w, h)})
                    cv2.rectangle(image_with_polys, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.rectangle(image_with_polys_on_preprocessed, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    split_texts = split_text_price(txt)
                    split_polys = split_polygon_horizontally(poly_np)
                    for spoly in split_polys:
                        sx, sy, sw, sh = cv2.boundingRect(spoly)
                        cv2.rectangle(image_with_split_polys, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
                    height_score = h / max_height
                    numeric_score = 1.0 if re.fullmatch(r"\d+", txt) else 0.0
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [poly_np], 255)
                    region = cv2.bitwise_and(image, image, mask=mask)
                    mean_color = cv2.mean(region, mask=mask)[:3]
                    is_red = mean_color[2] > 150 and mean_color[1] < 100 and mean_color[0] < 100
                    red_score = 1.0 if is_red else 0.0
                    nearby_score = 0.0
                    for other in all_text_polys:
                        if other['text'] in txt:
                            continue
                        ox, oy, ow, oh = other['bbox']
                        center_dist = np.linalg.norm(np.array([x + w/2, y + h/2]) - np.array([ox + ow/2, oy + oh/2]))
                        same_line = abs(y - oy) < h * 0.5
                        if center_dist < 150 and same_line and re.search(r"(円|本体|税込|価格)", other['text']):
                            nearby_score = 1.0
                            break
                    total_score = height_score * 3 + numeric_score * 3 + red_score * 2 + nearby_score * 2
                    candidate_regions.append({
                        'score': total_score,
                        'text': txt,
                        'poly': poly_np,
                        'bbox': (x, y, w, h)
                    })
                    paddle_text_lines.append(txt)

        # --- 構造化情報の抽出 ---
        def extract_structured_info(lines, fallback_base_price=""):
            print("構造化情報の抽出")  # リクエストが到達したことを確認
            text_block = "\n".join(lines)
            product_name = ""
            for line in lines:
                if re.match(r"^[\d\s\W_]+$", line):
                    continue
                if len(line) >= 5 and not re.search(r"\d", line):
                    product_name = line
                    break
            jan_match = re.search(r"\b(\d{13})\b", text_block)
            jan_code = jan_match.group(1) if jan_match else ""
            base_price_match = re.search(r"(\d{2,4})\s*円\s*[\n ]*本体", text_block)
            base_price = base_price_match.group(1) if base_price_match else fallback_base_price
            tax_price_match = re.search(r"(\d{2,4}(?:\.\d{1,2})?)\s*円", text_block)
            tax_price = tax_price_match.group(1) if tax_price_match else ""
            if tax_price == base_price:
                tax_price = ""
            expiry_match = re.search(r"\b(\d{1,2}/\d{1,2})\b", text_block)
            expiry = expiry_match.group(1) if expiry_match else ""
            return {
                "商品名": product_name,
                "JANコード": jan_code,
                "本体価格": base_price,
                "税込価格": tax_price,
                "特売期限": expiry
            }

        # --- 本体価格の補完（再OCRで高スコアな数字） ---
        print("本体価格の補完")  # リクエストが到達したことを確認
        best_numeric = sorted(
            [c for c in candidate_regions if re.fullmatch(r"\d{2,4}", c['text'])],
            key=lambda x: -x['score']
        )
        fallback_base_price = best_numeric[0]['text'] if best_numeric else ""

        # --- 構造化出力・保存 ---
        print("構造化出力・保存")  # リクエストが到達したことを確認
        structured = extract_structured_info(paddle_text_lines, fallback_base_price)
        json_path = os.path.join(output_dir, "structured_output.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, ensure_ascii=False, indent=2)
        print(json.dumps(structured, ensure_ascii=False, indent=2))

        print(f"OCR認識完了: {time.time() - t_ocr_start:.2f}秒")
        print("認識結果(by PaddleOCR)",(structured))
        print("JANコード(by PaddleOCR)",(structured.get("JANコード")))
        print("商品名(by PaddleOCR)", structured.get("商品名", []))

#        search_rakuten_product(structured.get("JANコード"), structured.get("商品名", []))



        # --- OCR出力結果の表示（従来形式） ---
        print("\n③ 再OCR結果（1/10サイズ画像・白背景）:")
    
        if best_numeric:
            print(f"  '{best_numeric[0]['text']}' (score: {best_numeric[0]['score']:.2f})")
        else:
            print("  (なし)")
        print("\n🔤 OCRで検出された全文テキスト:")
        for line in paddle_text_lines:
            print("   ", line)

        # デバッグ画像保存
        cv2.imwrite(os.path.join(output_dir, "debug_poly_drawn.jpg"), image_with_polys)
        cv2.imwrite(os.path.join(output_dir, "debug_split_poly_drawn.jpg"), image_with_split_polys)
        cv2.imwrite(os.path.join(output_dir, "debug_poly_drawn_preprocessed.jpg"), image_with_polys_on_preprocessed)
        cv2.imwrite(os.path.join(output_dir, "gray.jpg"), gray)

        return structured

##        return jsonify(structured)
        # --- 楽天商品検索 ---
#       result_data = search_rakuten_product(
#            structured.get("JANコード"),
#            structured.get("商品名", [])
#       )
##        result_data = search_rakuten_product(jan_code, product_name)


#        print("✅ JSON返却:", json.dumps({
#            "source": "ocr",
#            structured
#            "structured": structured,
#            "result": result_data or {"検索結果": [], "検索種別": "なし", "キーワード": ""}    # 楽天検索結果
#        }, ensure_ascii=False, indent=2))
#        return {
#            "source": "ocr",
#            structured
#            "structured": structured,
#            "result": result_data    # 楽天検索結果
#        }
  
#        return jsonify({
#            "source": "ocr",
#            "structured": structured,
#            "result": result_data
#        })


    except Exception as e:
        return jsonify({'error': str(e)}), 500





# --- 接続確認用エンドポイント ---
@app.route('/', methods=['GET'])
def index():
    return 'OCR API Server is running.'


# グローバルでOCR結果を保持（簡易な例）
ocr_cache = {}




# --- アップロード処理エンドポイント ---
# --- バーコード認識、OCR認識、楽天API検索の処理 ---
t_upload = time.time()
from shutil import copyfile  # ← ファイルコピーに使う　             ■■■ これだけ後で消す！ ■■■■
@app.route('/upload', methods=['POST'])
def upload_image():
    print("UPLOAD実行")  # リクエストが到達したことを確認
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400


# 👇 固定ファイル名にすり替え（例: sample.jpg）　                   ■■■ ここからcopyfileまで後で消す！ ■■■■
    filename = 'sample.jpg'
    filepath = os.path.join('sample.jpg')  # sample.jpgの保存場所
    dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # 👇 sample.jpg をアップロード先にコピー（すり替え）
    copyfile(filepath, dest_path)
    print(f"画像保存完了: {filepath}")
    print(f"画像アップロード時間: {time.time() - t_upload:.2f}秒")


#    file = request.files['file']
#    filename = file.filename
#    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#    file.save(filepath)


    # 🔁 バーコードスレッドを開始
    t_barcode_start = time.time()
    image = cv2.imread(filepath)

    # --- ① バーコード認識（先行実行）
    t_barcode_start = time.time()
    barcode_result = detect_barcode(image)
    print(f"📦 バーコード認識結果: {barcode_result}")
    print(f"📦 バーコード認識: {time.time() - t_barcode_start:.2f}秒")

    # --- ③ 楽天API検索（JANコード優先、なければ商品名で）
#    rakuten_result = search_rakuten_product(
#       barcode_result,
#        []
#   )

    # --- JANコードによる楽天検索
    if barcode_result:
        rakuten_result = search_rakuten_product(
            barcode_result,
#            barcode_result.get("キーワード"),  # JANコード
            []  # 商品名は空
        )



    # --- バーコード認識、OCR認識、楽天API検索の結果を返す ---
    if barcode_result:
        return jsonify({
            "filename": filename,
            "source": "barcode",
            "recognition": barcode_result,
            "rakuten": rakuten_result,
        })
    else:
        return jsonify({
            "filename": filename,
            "source": "barcode",
            "recognition": None
        })
    
    return jsonify({'message': 'Image uploaded successfully', 'filename': filename})



# --- 🔍 楽天「商品価格ナビ製品検索API」呼び出し ---
def search_rakuten_product(jan_code, product_names):
    endpoint = "https://app.rakuten.co.jp/services/api/Product/Search/20170426"
    headers = {
        "User-Agent": "ocr-app-client"
    }

    # 優先：JANコードによる検索
    if jan_code:
        print(f"\n🔎 JANコード '{jan_code}' で楽天API検索中...")
        params = {
            "format": "json",
            "applicationId": RAKUTEN_APP_ID,
            "keyword": jan_code
        }
        try:
            res = requests.get(endpoint, params=params, headers=headers)
            res.raise_for_status()
            data = res.json()
            items = data.get("Products", [])
            if items:
                products = []
                print(f"✅ JANコード検索で {len(items)} 件ヒット")
                for item in items[:7]:
                    p = item["Product"]
                    print(f"- 商品名: {p['productName']}")
                    print(f"  メーカー: {p.get('makerName')}")
                    print(f"  メーカー正式名: {p.get('makerNameFormal')}")
                    print(f"  キャプション: {p.get('productCaption')}")
                    print(f"  商品画像: {p.get('mediumImageUrl')}")
                    print(f"  最安価格: {p.get('minPrice')}円")
                    print(f"  URL: {p.get('productUrlMobile')}")

                    products.append({
                        "商品名": p['productName'],
                        "メーカー": p.get('makerName'),
                        "メーカー正式名": p.get('makerNameFormal'),
                        "キャプション": p.get('productCaption'),
                        "商品画像": p.get('mediumImageUrl'),
                        "最安価格": p.get('minPrice'),
                        "URL": p.get('productUrlMobile')
                    })
                return {"検索結果": products, "検索種別": "JAN", "キーワード": jan_code}
            else:
                print("❌ JANコードでは該当商品なし")
                return None  # 検索失敗    

        except Exception as e:
            print(f"APIエラー（JAN検索）: {e}")
            return None  # 👈 明示
        
    # 次点：商品名候補で検索
    if product_names:
        print(f"\n🔎 商品名 '{product_names}' で楽天API検索中...")
        params = {
            "format": "json",
            "applicationId": RAKUTEN_APP_ID,
            "keyword": product_names
        }
    try:
        res = requests.get(endpoint, params=params, headers=headers)
        res.raise_for_status()
        data = res.json()
        items = data.get("Products", [])
        if items:
            print(f"✅ '{product_names}' で {len(items)} 件ヒット")
            for item in items[:7]:
                p = item["Product"]
                print(f"- 商品名: {p['productName']}")
                print(f"  メーカー: {p.get('makerName')}")
                print(f"  メーカー正式名: {p.get('makerNameFormal')}")
                print(f"  キャプション: {p.get('productCaption')}")
                print(f"  商品画像: {p.get('mediumImageUrl')}")
                print(f"  最安価格: {p.get('minPrice')}円")
                print(f"  URL: {p.get('productUrlMobile')}")

                products.append({
                    "商品名": p['productName'],
                    "メーカー": p.get('makerName'),
                    "メーカー正式名": p.get('makerNameFormal'),
                    "キャプション": p.get('productCaption'),
                    "商品画像": p.get('mediumImageUrl'),
                    "最安価格": p.get('minPrice'),
                    "URL": p.get('productUrlMobile')
                })
            return {"検索結果": products, "検索種別": "JAN", "キーワード": jan_code}
        else:
            print("❌ 商品名では該当商品なし")
            return None  # 検索失敗    
    except Exception as e:
        print(f"APIエラー（商品名検索）: {e}")
        print("❌ すべての検索候補で該当商品なし")
        return None  # 👈 明示
    


# --- OCR処理エンドポイント ---
@app.route('/ocr', methods=['POST'])
def run_ocr():
    print("OCRリクエスト")  # リクエストが到達したことを確認
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({'error': 'Missing filename'}), 400
    filename = data['filename']
    ocr_result = run_ocr_logic(filename)  # ✅ OCR処理を共通関数で実行
    print(f"filename: {filename}")
    print(f"oce認識結果1: {ocr_result}")

#    return jsonify(ocr_result)

    # --- 楽天API検索
    # OCR認識後の楽天API検索処理
    if ocr_result:
        # タプルだった場合は辞書部分を取り出す
        if isinstance(ocr_result, tuple):
            ocr_result = ocr_result[0]
        if hasattr(ocr_result, "get_json"):
            ocr_result = ocr_result.get_json()
        print(f"oce認識結果2: {ocr_result}")
        print("JANコード(by PaddleOCR)",(ocr_result.get("JANコード")))
        print("商品名(by PaddleOCR)", (ocr_result.get("商品名", [])))

        # 安全に取り出す
#        recognition = ocr_result.get("recognition", {})
        jan_code = ocr_result.get("JANコード")
        product_name = ocr_result.get("商品名", [])
        print(f"JANコード: {jan_code}, 商品名: {product_name}")

        rakuten_result = search_rakuten_product(jan_code, product_name)
#            ocr_result.get("JANコード"),       # JANコード
#            ocr_result.get("商品名"),          # 商品名
#        )

    # --- バーコード認識、OCR認識、楽天API検索の結果を返す ---
    if ocr_result:
        return jsonify({
            "filename": filename,
            "source": "ocr",
            "recognition": ocr_result,
            "rakuten": rakuten_result,
        })
    else:
        return jsonify({
            "filename": filename,
            "source": "ocr",
            "recognition": None
        })



# === 近隣店舗検索 ===
@app.route('/api/nearby-shops', methods=['POST'])
def nearby_shops():
    data = request.json
    print(f"{data}")
    latitude = data.get('lat')
    longitude = data.get('lon')
#   latitude = data.get('latitude')
#   longitude = data.get('longitude')
    print(f"📦 緯度: {latitude}")
    print(f"📦 経度: {longitude}")

    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return jsonify({"error": "緯度または経度が不正です。"}), 400

    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        return jsonify({"error": "緯度または経度が不正です。"}), 400

    radius_km = 2.0
    nearby = []

    for index, feature in enumerate(geojson["features"]):
        try:
            lng, lat = feature["geometry"]["coordinates"]
#            print("feature latitude:",latitude)
#            print("feature longitude:", longitude)
#            print("feature lat:",lat)
#            print("feature lng:",lng)
            distance_km = geodesic((latitude, longitude), (lat, lng)).kilometers
            if distance_km <= radius_km:
                nearby.append({
                    "id": index,
                    "name": feature.get("properties", {}).get("name", "名称不明"),
                    "brand": feature.get("properties", {}).get("brand"),
                    "category": feature.get("properties", {}).get("shop"),
                    "coordinates": feature.get("geometry", {}).get("coordinates", []),
                    "distance_km": round(distance_km, 3)
                })
        except Exception as feature_err:
            print(f"❌ Feature処理エラー（index={index}）: {feature_err}")

        # ✅ 現在地に最も近い順（＝distance_km の昇順）に並び替える
        nearby.sort(key=lambda s: s["distance_km"])

    print(f"📦 ヒット店舗数: {len(nearby)}")
    if nearby:
        print("🔍 最初の店舗例:", json.dumps(nearby[0], ensure_ascii=False, indent=2))

    return jsonify({
        "count": len(nearby),
        "units": "kilometers",
        "origin": {"latitude": latitude, "longitude": longitude},
        "shops": nearby
    })



# === 値札情報DB登録 ===
@app.route('/api/register-price', methods=['POST'])
def register_price():

    cursor.execute("PRAGMA table_info(prices);")
    cols = cursor.fetchall()
    print("[DEBUG] テーブル構造:", cols)

    data = request.json
    product_name = data.get('product_name')
    price = data.get('price')
    shop_name = data.get('shop_name')
    lat = data.get('lat')
    lon = data.get('lon')
    jan = data.get('jan')
    image_url = data.get('image_url')
    print(f"📦 register-priceの処理中・・・")
    print(f"{data}")
    if not all([product_name, price, shop_name, lat, lon, jan, image_url]):
        return jsonify({"error": "全ての項目が必要です"}), 400

    try:
        cursor.execute('''
            INSERT INTO prices (product_name, price, shop_name, lat, lon, jan, image_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (product_name, price, shop_name, lat, lon, jan, image_url))
        conn.commit()
        return jsonify({"message": "価格情報を登録しました"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# === 価格ランキング作成 ===
@app.route('/api/price-ranking', methods=['POST'])
def price_ranking():
    payload = request.json        # jan OR product_name,  lat, lon
    jan = payload.get("jan")
    pname = payload.get("product_name")
    lat0, lon0 = payload["lat"], payload["lon"]
    print(f"📦price-rankingの処理中・・・")
    print(f"{payload}")

    # 同一商品：JAN があれば優先、なければ商品名で LIKE 検索
    if jan:
        cursor.execute("SELECT * FROM prices WHERE jan=?", (jan,))
    else:
        cursor.execute("SELECT * FROM prices WHERE product_name LIKE ?", ('%'+pname+'%',))
    rows = cursor.fetchall()
    print("[DEBUG] price-ranking rows:", rows)

    # 30 km 以内フィルタ & 価格昇順
    results = []
    for r in rows:
        id, p_name, price, shop, lat, lon, jan, img, created_at = r

        # ---------- ここからデバッグ ----------
        try:
            print(
                "[DEBUG] geodesic inputs:",
                f"lat0={lat0} ({type(lat0)})",
                f"lon0={lon0} ({type(lon0)})",
                f"lat={lat} ({type(lat)})",
                f"lon={lon} ({type(lon)})",
            )

            # 必要なら float へキャストして確認
            lat0_f = float(lat0)
            lon0_f = float(lon0)
            lat_f  = float(lat)
            lon_f  = float(lon)

            # 範囲外ならスキップ
            if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
                print("[WARN] 緯度経度が範囲外のためスキップ:", lat_f, lon_f)
                continue

            # 距離計算
            dist = geodesic((lat0_f, lon0_f), (lat_f, lon_f)).kilometers
            print(f"距離計算：{dist}")

        except ValueError as e:
            print("[ERROR] geodesic ValueError:", e)
            continue
        # ---------- ここまでデバッグ ----------
        if dist <= 2:
            results.append({
                "商品名": p_name,
                "商品画像": img,
                "店舗名": shop,
                "価格": price,
                "緯度": lat,
                "経度": lon,
                "距離_km": round(dist,2)
            })
    results.sort(key=lambda x: x["価格"])
    print(f"価格によるソート：{results}")
    return jsonify({"ranking": results[:5]})



if __name__ == '__main__':
    # 開発用：ローカル確認用のFlaskサーバー
    # 注意：本番では gunicorn など WSGI サーバーを使って起動してください
    # 例: gunicorn -w 4 ocr_api_server:app
    app.run(host='0.0.0.0', port=8000, debug=True)

