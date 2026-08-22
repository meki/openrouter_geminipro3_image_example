# %%
import base64
import datetime
import json
import mimetypes
import os
from urllib.parse import unquote_to_bytes
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
import requests
import yaml
from PIL import Image

# %%

SUPPORTED_IMAGE_MODELS = [
    "google/gemini-3.1-flash-image-preview",
    "google/gemini-3.1-flash-lite-image",
    "google/gemini-3-pro-image-preview",
    "recraft/recraft-v4.1-pro-vector",
    "recraft/recraft-v4.1-vector",
    "openai/gpt-5.4-image-2",
    "openai/gpt-image-2",
    "krea/krea-2-large",
    "qwen/qwen-image-3",
    "qwen/qwen-image-3-pro",
]

MODEL_MODALITIES = {
    "recraft/recraft-v4.1-pro-vector": ["image"],
    "recraft/recraft-v4.1-vector": ["image"],
}

MODALITIES_SUPPORTED_PREFIXES = ("google/", "openai/")
TEXT_ONLY_STRING_CONTENT_MODELS = {
    "recraft/recraft-v4.1-pro-vector",
    "recraft/recraft-v4.1-vector",
}

# chat/completions ではなく専用の /api/v1/images エンドポイントを使うモデル
IMAGES_API_MODELS = {
    "openai/gpt-image-2",
    "krea/krea-2-large",
    "qwen/qwen-image-3",
    "qwen/qwen-image-3-pro",
}


def get_model_modalities(model):
    if model in MODEL_MODALITIES:
        return MODEL_MODALITIES[model]
    if any(model.startswith(prefix) for prefix in MODALITIES_SUPPORTED_PREFIXES):
        return ["image", "text"]
    return None


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def image_path_to_data_url(image_path):
    media_type, _ = mimetypes.guess_type(str(image_path))
    if not media_type:
        media_type = "image/jpeg"
    return f"data:{media_type};base64,{encode_image_to_base64(image_path)}"


def get_image_from_base64(base64_image):
    return Image.open(BytesIO(base64.b64decode(base64_image)))


def show_image_from_base64(base64_image):
    get_image_from_base64(base64_image).show()


def base64_url_to_base64_image(base64_url):
    # data:image/{format};base64,{data} 形式から base64 データを抽出
    if ";base64," in base64_url:
        return base64_url.split(";base64,", 1)[1]
    return base64_url  # すでに base64 データの場合はそのまま返す


def image_extension_from_media_type(media_type):
    media_type = media_type.lower()
    if media_type == "image/svg+xml":
        return "svg"
    if media_type == "image/jpeg":
        return "jpg"
    if media_type.startswith("image/"):
        return media_type.split("/", 1)[1]
    return None


def decode_image_data_url(image_url):
    if image_url.startswith("data:") and "," in image_url:
        header, encoded_data = image_url.split(",", 1)
        media_type = header[5:].split(";", 1)[0]
        extension = image_extension_from_media_type(media_type)

        if ";base64" in header.lower():
            return base64.b64decode(encoded_data), extension
        return unquote_to_bytes(encoded_data), extension

    return base64.b64decode(base64_url_to_base64_image(image_url)), None


def save_base64_url_to_file(base64_url, output_path):
    image_data, data_url_extension = decode_image_data_url(base64_url)
    output_path = Path(output_path)

    if data_url_extension == "svg":
        output_path = output_path.with_suffix(".svg")
        output_path.write_bytes(image_data)
        return output_path
    
    # 画像フォーマットを自動判別
    image = Image.open(BytesIO(image_data))
    
    # 出力パスの拡張子を画像フォーマットに合わせる
    format_extension = image.format.lower() if image.format else data_url_extension or 'png'
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
    }

    modalities = get_model_modalities(model)
    if modalities:
        payload["modalities"] = modalities

    response = requests.post(url, headers=headers,
                             json=payload, timeout=(10, 300))
    return response

def image_api_request(prompt_text, image_paths, model, openrouter_api_key=None):
    """専用の Images API (/api/v1/images) を使用した画像生成リクエスト"""
    url = "https://openrouter.ai/api/v1/images"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key or os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "prompt": prompt_text,
    }

    if image_paths:
        payload["input_references"] = [
            {
                "type": "image_url",
                "image_url": {"url": image_path_to_data_url(path)}
            }
            for path in image_paths
        ]

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
    choices = response_data.get("choices") or []
    if choices:
        # chat/completions 形式: choices[0].message.images[].image_url.url
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
        raw_images = message.get("images") or []
        image_base64_list = [
            (image_info.get("image_url", {}) or {}).get("url")
            for image_info in raw_images
            if isinstance(image_info, dict)
        ]
    else:
        # 専用 Images API 形式: data[].b64_json
        data_items = response_data.get("data") or []
        image_base64_list = [
            item.get("b64_json")
            for item in data_items
            if isinstance(item, dict)
        ]
    image_base64_list = [image for image in image_base64_list if image]

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
    # 最初の画像のみを保存（複数あっても全て同じ画像のため）
    if image_base64_list:
        base64_response = image_base64_list[0]
        output_image_path = output_folder_path / f"{yyyymmddhhmmss}_{id}_0"
        saved_path = save_base64_url_to_file(base64_response, output_image_path)
        print(f"Saved image to {saved_path}")
        saved_image_paths.append(saved_path)

    # prompt_info.yaml/jsonを保存
    prompt_info_output_path = output_folder_path / f"{yyyymmddhhmmss}_{id}_prompt_info.yaml"
    prompt_info_output_path.write_text(yaml.dump(prompt_info_data, allow_unicode=True), encoding="utf-8")

    prompt_info_json_output_path = output_folder_path / f"{yyyymmddhhmmss}_{id}_prompt_info.json"
    prompt_info_json_output_path.write_text(
        json.dumps(prompt_info_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return output_folder_path, saved_image_paths

def unified_image_preview_request(prompt_text, image_paths, model, openrouter_api_key):
    """画像生成リクエストを送信する統合関数
    
    Args:
        prompt_text: プロンプトテキスト
        image_paths: 入力画像のパスリスト（空のリストも可）
        model: 使用するモデル名
        openrouter_api_key: OpenRouter APIキー
    
    Returns:
        APIレスポンス
    """
    if model in IMAGES_API_MODELS:
        return image_api_request(prompt_text, image_paths, model, openrouter_api_key)

    text_content = {"type": "text", "text": prompt_text}

    # 画像がある場合のみ画像コンテンツを追加
    image_contents = [
        {
            "type": "image_url",
            "image_url": {
                "url": image_path_to_data_url(path)
            }
        }
        for path in image_paths
    ] if image_paths else []

    if image_contents:
        content = [text_content, *image_contents]
    elif model in TEXT_ONLY_STRING_CONTENT_MODELS:
        content = prompt_text
    else:
        content = [text_content]

    messages = [
        {"role": "user", "content": content}
    ]

    response = image_generation_request(messages, model=model, openrouter_api_key=openrouter_api_key)
    return response

def gemini_pro_3_1_image_preview_request(prompt_text, image_paths, openrouter_api_key):
    """Gemini Pro 3.1を使用した画像生成リクエスト"""
    return unified_image_preview_request(prompt_text, image_paths, "google/gemini-3.1-flash-image-preview", openrouter_api_key)

def gemini_pro_3_image_preview_request(prompt_text, image_paths, openrouter_api_key):
    """Gemini Pro 3を使用した画像生成リクエスト"""
    return unified_image_preview_request(prompt_text, image_paths, "google/gemini-3-pro-image-preview", openrouter_api_key)

def recraft_v4_1_pro_vector_request(prompt_text, image_paths, openrouter_api_key):
    """Recraft V4.1 Pro Vectorを使用した画像生成リクエスト"""
    return unified_image_preview_request(prompt_text, image_paths, "recraft/recraft-v4.1-pro-vector", openrouter_api_key)

def recraft_v4_1_vector_request(prompt_text, image_paths, openrouter_api_key):
    """Recraft V4.1 Vectorを使用した画像生成リクエスト"""
    return unified_image_preview_request(prompt_text, image_paths, "recraft/recraft-v4.1-vector", openrouter_api_key)

def gpt_5_4_image_2_request(prompt_text, image_paths, openrouter_api_key):
    """GPT-5.4 Image 2を使用した画像生成リクエスト"""
    return unified_image_preview_request(prompt_text, image_paths, "openai/gpt-5.4-image-2", openrouter_api_key)

def gpt_image_2_request(prompt_text, image_paths, openrouter_api_key):
    """GPT-image-2を使用した画像生成リクエスト"""
    return unified_image_preview_request(prompt_text, image_paths, "openai/gpt-image-2", openrouter_api_key)

def request_image_preview(prompt_text, image_paths, model, openrouter_api_key):
    """対応済みモデル名を使って画像生成リクエストを送信する"""
    if model not in SUPPORTED_IMAGE_MODELS:
        raise ValueError(f"Unsupported model: {model}")
    return unified_image_preview_request(prompt_text, image_paths, model, openrouter_api_key)

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
        model = prompt_info.get("model", "google/gemini-3-pro-image-preview")

    image_paths = [path.strip('"') for path in image_paths]

    try:
        response = request_image_preview(prompt_text, image_paths, model, OPENROUTER_API_KEY)
    except ValueError as error:
        print(error)
        print(f"Supported models: {', '.join(SUPPORTED_IMAGE_MODELS)}")
        return

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    response_data = response.json()
    save_response_images(OUTPUT_BASE_FOLDER, response_data, prompt_info)

if __name__ == "__main__":
    main()