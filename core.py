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
    Path(output_path).write_bytes(base64.b64decode(base64_image))


def _gemini_pro_3_image_preview_request(messages, openrouter_api_key=None):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openrouter_api_key or os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    gemini_3_pro_image_preview = "google/gemini-3-pro-image-preview"
    payload = {
        "model": gemini_3_pro_image_preview,
        "messages": messages,
        "modalities": ["image", "text"]
    }

    response = requests.post(url, headers=headers,
                             json=payload, timeout=(10, 300))
    return response

def save_response_images(output_base_folder, response):
    response_data = response.json()
    images = response_data.get("choices", [])[0].get(
        "message", {}).get("images", [])

    now = datetime.datetime.now()
    yyyymmdd_hy = now.strftime("%Y-%m-%d")
    yyyymmddhhmmss = now.strftime("%Y%m%d%H%M%S")

    id = response_data.get("id", "unknown_id")

    today_folder = output_base_folder / yyyymmdd_hy
    today_folder.mkdir(parents=True, exist_ok=True)
    output_folder_path = today_folder / f"{yyyymmddhhmmss}_{id}"
    output_json_path = output_folder_path / "response.json"

    output_folder_path.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(
        response_data, indent=2), encoding="utf-8")

    for idx, image_info in enumerate(images):
        base64_response = image_info["image_url"]["url"]
        output_image_path = output_folder_path / f"output_{id}_{idx}.jpg"
        save_base64_url_to_file(base64_response, output_image_path)
        print(f"Saved image to {output_image_path}")
    
    return output_folder_path

def gemini_pro_3_image_preview_request(prompt_text, image_paths, openrouter_api_key):
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

    response = _gemini_pro_3_image_preview_request(messages, openrouter_api_key=openrouter_api_key)
    return response

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

    save_response_images(OUTPUT_BASE_FOLDER, response)

if __name__ == "__main__":
    main()