# %%
import base64
import datetime
import json
import os
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
import requests
import yaml
from PIL import Image

# %%


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_image_from_base64(base64_image):
    return Image.open(BytesIO(base64.b64decode(base64_image)))


def show_image_from_base64(base64_image):
    get_image_from_base64(base64_image).show()


def base64_url_to_base64_image(base64_url):
    # data:image/{format};base64,{data} 形式から base64 データを抽出
    if ";base64," in base64_url:
        return base64_url.split(";base64,", 1)[1]
    return base64_url  # すでに base64 データの場合はそのまま返す


def save_base64_url_to_file(base64_url, output_path):
    base64_image = base64_url_to_base64_image(base64_url)
    image_data = base64.b64decode(base64_image)
    
    # 画像フォーマットを自動判別
    image = Image.open(BytesIO(image_data))
    
    # 出力パスの拡張子を画像フォーマットに合わせる
    output_path = Path(output_path)
    format_extension = image.format.lower() if image.format else 'png'
    if format_extension == 'jpeg':
        format_extension = 'jpg'
    output_path = output_path.with_suffix(f'.{format_extension}')
    
    # 画像を保存
    image.save(output_path)
    return output_path


def image_generation_request(messages, model, openrouter_api_key=None):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key or os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "modalities": ["image", "text"]
    }

    response = requests.post(url, headers=headers,
                             json=payload, timeout=(10, 300))
    return response

def save_response_images(output_base_folder, response_data, prompt_info_data):
    """レスポンスの画像をファイルに保存し、保存したパスリストを返す
    
    Args:
        output_base_folder: 出力先ベースフォルダ
        response_data: パース済みのレスポンスJSON
        prompt_info_data: プロンプト情報
    
    Returns:
        tuple: (output_folder_path, saved_image_paths)
            - output_folder_path: 保存先フォルダパス
            - saved_image_paths: 保存した画像ファイルのパスリスト
    """
    images = response_data.get("choices", [])[0].get(
        "message", {}).get("images", [])

    now = datetime.datetime.now()
    yyyymmdd_hy = now.strftime("%Y-%m-%d")
    yyyymmddhhmmss = now.strftime("%Y%m%d%H%M%S")

    id = response_data.get("id", "unknown_id")

    today_folder = output_base_folder / yyyymmdd_hy
    today_folder.mkdir(parents=True, exist_ok=True)
    output_folder_path = today_folder
    output_json_path = output_folder_path / f"{yyyymmddhhmmss}_{id}_response.json"

    output_folder_path.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(
        response_data, indent=2), encoding="utf-8")

    saved_image_paths = []
    for idx, image_info in enumerate(images):
        base64_response = image_info["image_url"]["url"]
        output_image_path = output_folder_path / f"{yyyymmddhhmmss}_{id}_{idx}"
        saved_path = save_base64_url_to_file(base64_response, output_image_path)
        print(f"Saved image to {saved_path}")
        saved_image_paths.append(saved_path)

    # prompt_info.yamlを保存
    prompt_info_output_path = output_folder_path / f"{yyyymmddhhmmss}_{id}_prompt_info.yaml"
    prompt_info_output_path.write_text(yaml.dump(prompt_info_data, allow_unicode=True), encoding="utf-8")

    return output_folder_path, saved_image_paths

def unified_image_preview_request(prompt_text, image_paths, model, openrouter_api_key):
    """画像生成リクエストを送信する統合関数
    
    Args:
        prompt_text: プロンプトテキスト
        image_paths: 入力画像のパスリスト
        model: 使用するモデル名
        openrouter_api_key: OpenRouter APIキー
    
    Returns:
        APIレスポンス
    """
    text_content = {"type": "text", "text": prompt_text}

    image_contents = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image_to_base64(path)}"
            }
        }
        for path in image_paths
    ]

    messages = [
        {"role": "user",
                "content": [
                    text_content,
                    *image_contents
                ]
        }
    ]

    response = image_generation_request(messages, model=model, openrouter_api_key=openrouter_api_key)
    return response

def gemini_pro_3_image_preview_request(prompt_text, image_paths, openrouter_api_key):
    """Gemini Pro 3を使用した画像生成リクエスト"""
    return unified_image_preview_request(prompt_text, image_paths, "google/gemini-3-pro-image-preview", openrouter_api_key)

def flux_2_pro_image_preview_request(prompt_text, image_paths, openrouter_api_key):
    """Flux 2 Proを使用した画像生成リクエスト"""
    return unified_image_preview_request(prompt_text, image_paths, "black-forest-labs/flux.2-pro", openrouter_api_key)

def speedream_4_5_image_preview_request(prompt_text, image_paths, openrouter_api_key):
    """Speedream 4.5を使用した画像生成リクエスト"""
    return unified_image_preview_request(prompt_text, image_paths, "bytedance-seed/seedream-4.5", openrouter_api_key)

def flux_klein_image_preview_request(prompt_text, image_paths, openrouter_api_key):
    """Flux Kleinを使用した画像生成リクエスト"""
    return unified_image_preview_request(prompt_text, image_paths, "black-forest-labs/flux.2-klein-4b", openrouter_api_key)

def main():
    load_dotenv()

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OUTPUT_BASE_FOLDER = os.getenv("OUTPUT_BASE_FOLDER")
    OUTPUT_BASE_FOLDER = Path(OUTPUT_BASE_FOLDER)

    prompt_info_path = Path("prompt_info.yaml")
    with prompt_info_path.open("r", encoding="utf-8") as f:
        prompt_info = yaml.safe_load(f)
        prompt_text = prompt_info.get("text", "")
        image_paths = prompt_info.get("image_paths", [])

    image_paths = [path.strip('"') for path in image_paths]

    response = gemini_pro_3_image_preview_request(prompt_text, image_paths, OPENROUTER_API_KEY)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    response_data = response.json()
    save_response_images(OUTPUT_BASE_FOLDER, response_data, prompt_info)

if __name__ == "__main__":
    main()