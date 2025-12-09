"""
ユーティリティ関数群
設定ファイル管理、履歴管理、お気に入り管理など、UI以外の処理を担当
"""

import os
import json
import tempfile
from pathlib import Path
from PIL import Image


def get_settings_path():
    """設定ファイルのパスを取得"""
    settings_folder = os.getenv("SETTING_FOLDER_PATH", ".")
    settings_folder = Path(settings_folder)
    settings_folder.mkdir(parents=True, exist_ok=True)
    return settings_folder / "settings.json"


def load_settings():
    """設定ファイルを読み込む"""
    settings_path = get_settings_path()
    if settings_path.exists():
        try:
            with settings_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"image_path_history": [], "favorite_image_paths": []}
    return {"image_path_history": [], "favorite_image_paths": []}


def save_settings(settings):
    """設定ファイルを保存"""
    settings_path = get_settings_path()
    try:
        with settings_path.open("w", encoding="utf-8") as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save settings: {e}")


def add_to_history(path):
    """画像パスを履歴に追加"""
    if not path or path.strip() == "":
        return
    
    path = path.strip('"')
    if not Path(path).exists():
        return
    
    settings = load_settings()
    history = settings.get("image_path_history", [])
    
    # 既存の場合は削除
    if path in history:
        history.remove(path)
    
    # 先頭に追加
    history.insert(0, path)
    
    # 最大件数をsettingsから取得（デフォルト300件）
    max_history = settings.get("max_history_count", 300)
    history = history[:max_history]
    
    settings["image_path_history"] = history
    save_settings(settings)


def get_history_choices():
    """履歴からドロップダウンの選択肢を取得"""
    settings = load_settings()
    history = settings.get("image_path_history", [])
    # 存在するパスのみを返す
    return [p for p in history if Path(p).exists()]


def add_to_favorites(path):
    """画像パスをお気に入りに追加"""
    if not path or path.strip() == "":
        return
    
    path = path.strip('"')
    if not Path(path).exists():
        return
    
    settings = load_settings()
    favorites = settings.get("favorite_image_paths", [])
    
    # 既にお気に入りに入っている場合は何もしない
    if path not in favorites:
        favorites.append(path)
        settings["favorite_image_paths"] = favorites
        save_settings(settings)


def remove_from_favorites(path):
    """画像パスをお気に入りから削除"""
    if not path or path.strip() == "":
        return
    
    path = path.strip('"')
    settings = load_settings()
    favorites = settings.get("favorite_image_paths", [])
    
    if path in favorites:
        favorites.remove(path)
        settings["favorite_image_paths"] = favorites
        save_settings(settings)


def is_favorite(path):
    """画像パスがお気に入りに入っているか確認"""
    if not path or path.strip() == "":
        return False
    
    path = path.strip('"')
    settings = load_settings()
    favorites = settings.get("favorite_image_paths", [])
    return path in favorites


def get_favorites_choices():
    """お気に入りからドロップダウンの選択肢を取得"""
    settings = load_settings()
    favorites = settings.get("favorite_image_paths", [])
    # 存在するパスのみを返す
    return [p for p in favorites if Path(p).exists()]


def get_history_gallery(filter_mode="all"):
    """履歴画像をギャラリー用のタプルリストで取得
    
    Args:
        filter_mode: "all" (全て), "favorites" (お気に入りのみ)
    
    Returns:
        tuple: (gallery_items, displayed_paths)
            - gallery_items: ギャラリー表示用の(PIL Image, caption)リスト
            - displayed_paths: 実際に表示されている画像のパスリスト
    """
    settings = load_settings()
    max_gallery_display = settings.get("max_gallery_display", 50)  # デフォルト50件
    
    if filter_mode == "favorites":
        history_paths = get_favorites_choices()
    else:
        history_paths = get_history_choices()
    
    gallery_items = []
    displayed_paths = []  # 実際に表示されているパスのリスト
    favorites_set = set(settings.get("favorite_image_paths", []))
    
    for path in history_paths[:max_gallery_display]:
        try:
            if Path(path).exists():
                # (PIL Image, caption)のタプルで返す
                img = Image.open(path)
                # お気に入りの場合は★マークを付ける
                star = "★ " if path in favorites_set else ""
                caption = star + Path(path).name
                gallery_items.append((img, caption))
                displayed_paths.append(path)  # 表示されたパスを記録
        except Exception:
            continue
    return gallery_items, displayed_paths


def load_image_preview(path):
    """画像パスからプレビューを読み込む"""
    if not path or path.strip() == "":
        return None

    path = path.strip('"')

    try:
        if Path(path).exists():
            return Image.open(path)
        return None
    except Exception:
        return None


def check_image_path(path):
    """画像パスが存在するかチェック"""
    if not path or path.strip() or path.strip('"') == "":
        return ""
    if not Path(path).exists():
        return f"⚠️ パスが存在しません: {path}"
    return ""


def handle_image_upload(image):
    """アップロードされた画像を一時ファイルとして保存し、パスを返す"""
    if image is None:
        return "", None
    
    try:
        # 画像がファイルパスの場合
        if isinstance(image, str):
            return image, Image.open(image)
        
        # PIL Imageの場合、一時ファイルに保存
        # 環境変数から一時ディレクトリのベースを取得
        temp_base = os.getenv("TEMP_IMAGE_DIR", tempfile.gettempdir())
        temp_dir = Path(temp_base) / "gradio_images"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        import time
        timestamp = int(time.time() * 1000)
        temp_path = temp_dir / f"uploaded_{timestamp}.png"
        
        image.save(temp_path)
        return str(temp_path), image
    except Exception:
        return "", None
