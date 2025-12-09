import os
import tempfile
import json
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import gradio as gr
import yaml
from core import gemini_pro_3_image_preview_request, flux_2_pro_image_preview_request, save_response_images, get_image_from_base64, base64_url_to_base64_image


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
            return {"image_path_history": []}
    return {"image_path_history": []}


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


def get_history_gallery():
    """履歴画像をギャラリー用のタプルリストで取得"""
    settings = load_settings()
    max_gallery_display = settings.get("max_gallery_display", 50)  # デフォルト50件
    
    history_paths = get_history_choices()
    gallery_items = []
    for path in history_paths[:max_gallery_display]:
        try:
            if Path(path).exists():
                # (PIL Image, caption)のタプルで返す
                img = Image.open(path)
                # パスの最後の部分をキャプションに
                caption = Path(path).name
                gallery_items.append((img, caption))
        except Exception:
            continue
    return gallery_items


def select_from_gallery(evt: gr.SelectData):
    """ギャラリーから画像を選択したときの処理"""
    history_paths = get_history_choices()
    if evt.index < len(history_paths):
        selected_path = history_paths[evt.index]
        try:
            preview = Image.open(selected_path) if Path(selected_path).exists() else None
            return selected_path, preview
        except Exception:
            return selected_path, None
    return "", None


def show_image_row(current_count):
    """画像フォームの表示数を増やす"""
    new_count = min(current_count + 1, 10)  # 最大10個まで
    updates = []
    for i in range(10):
        updates.append(gr.Row(visible=(i < new_count)))
    updates.append(new_count)
    return updates


def hide_image_row(current_count):
    """画像フォームの表示数を減らす"""
    new_count = max(current_count - 1, 1)  # 最低1個は表示
    updates = []
    for i in range(10):
        updates.append(gr.Row(visible=(i < new_count)))
    updates.append(new_count)
    return updates


def check_image_path(path):
    """画像パスが存在するかチェック"""
    if not path or path.strip() or path.strip('"') == "":
        return ""
    if not Path(path).exists():
        return f"⚠️ パスが存在しません: {path}"
    return ""


def load_prompt_info(file):
    """
prompt_info.yamlを読み込んでフォームに流し込む"""
    if file is None:
        # ファイルがない場合は全て空で返す
        empty_paths = [""] * 10
        empty_previews = [None] * 10
        empty_rows = [gr.Row(visible=(i < 1)) for i in range(10)]
        return "", *empty_paths, *empty_previews, *empty_rows, 1

    try:
        file_path = Path(file.name) if hasattr(file, 'name') else Path(file)

        with file_path.open("r", encoding="utf-8") as f:
            prompt_info = yaml.safe_load(f)

        prompt_text = prompt_info.get("text", "")
        image_paths = prompt_info.get("image_paths", [])
        
        # 画像数を取得し、最大10個まで制限
        num_images = min(len(image_paths), 10)
        
        # 10個のパスとプレビューを準備
        paths = []
        previews = []
        
        for i in range(10):
            if i < len(image_paths):
                path = image_paths[i]
                paths.append(path)
                # 画像プレビューを読み込み
                preview = Image.open(path) if path and Path(path).exists() else None
                previews.append(preview)
            else:
                paths.append("")
                previews.append(None)
        
        # Rowの表示設定（画像数分表示する）
        row_updates = [gr.Row(visible=(i < num_images)) for i in range(10)]
        
        return prompt_text, *paths, *previews, *row_updates, num_images

    except Exception:
        # エラー時は全て空で返す
        empty_paths = [""] * 10
        empty_previews = [None] * 10
        empty_rows = [gr.Row(visible=(i < 1)) for i in range(10)]
        return "", *empty_paths, *empty_previews, *empty_rows, 1


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


def run_request(output_folder, api_key, model, prompt, *image_paths):
    """リクエストを実行して結果を返す"""
    # 空のパスをフィルタリング
    valid_image_paths = [p for p in image_paths if p and p.strip() != ""]

    valid_image_paths = [p.strip('"') for p in valid_image_paths]

    if not valid_image_paths:
        history = get_history_choices()
        return "エラー: 少なくとも1つの画像パスを指定してください", None, *([gr.Dropdown(choices=history)] * 10)

    # パスの存在確認
    for path in valid_image_paths:
        if not Path(path).exists():
            history = get_history_choices()
            return f"エラー: 画像パスが存在しません: {path}", None, *([gr.Dropdown(choices=history)] * 10)

    if not prompt or prompt.strip() == "":
        history = get_history_choices()
        return "エラー: プロンプトを入力してください", None, *([gr.Dropdown(choices=history)] * 10)

    if not api_key or api_key.strip() == "":
        history = get_history_choices()
        return "エラー: OpenRouter API Keyを入力してください", None, *([gr.Dropdown(choices=history)] * 10)
    
    # 画像パスを履歴に追加
    for path in valid_image_paths:
        add_to_history(path)

    try:
        # モデルに応じてリクエスト実行
        if model == "google/gemini-3-pro-image-preview":
            response = gemini_pro_3_image_preview_request(
                prompt, valid_image_paths, api_key)
        else:  # black-forest-labs/flux.2-pro
            response = flux_2_pro_image_preview_request(
                prompt, valid_image_paths, api_key)

        if response.status_code != 200:
            history = get_history_choices()
            return f"エラー: {response.status_code}\n{response.text}", None, *([gr.Dropdown(choices=history)] * 10)

        prompt_info_data = {
            "text": prompt,
            "image_paths": valid_image_paths
        }

        # 結果を保存
        save_response_images(Path(output_folder), response, prompt_info_data)

        # レスポンスから結果テキストを取得
        response_data = response.json()
        result_text = response_data.get("choices", [])[0].get(
            "message", {}).get("content", "")
        images = response_data.get("choices", [])[0].get(
            "message", {}).get("images", [])

        # 画像が0枚の場合はfinish_reasonを表示
        if len(images) == 0:
            native_finish_reason = response_data.get(
                "choices", [])[0].get("native_finish_reason", "不明")
            result = f"⚠️ 画像生成失敗\n\n結果:\n{result_text}\n\n"
            result += f"生成された画像数: {len(images)}\n"
            result += f"Finish Reason: {native_finish_reason}\n"
            result += f"保存先: {output_folder}"
        else:
            result = f"✅ 成功!\n\n結果:\n{result_text}\n\n"
            result += f"生成された画像数: {len(images)}\n"
            result += f"保存先: {output_folder}"

        # 画像をPIL形式に変換
        pil_images = []
        if images:
            for image_info in images:
                base64_url = image_info["image_url"]["url"]
                base64_data = base64_url_to_base64_image(base64_url)
                pil_image = get_image_from_base64(base64_data)
                pil_images.append(pil_image)
        
        # 更新された履歴を取得
        updated_history = get_history_choices()
        return result, pil_images if pil_images else None, *([gr.Dropdown(choices=updated_history)] * 10)

    except Exception as e:
        history = get_history_choices()
        return f"エラーが発生しました: {str(e)}", None, *([gr.Dropdown(choices=history)] * 10)


def create_ui():
    load_dotenv()

    default_output_folder = os.getenv("OUTPUT_BASE_FOLDER", "")
    default_api_key = os.getenv("OPENROUTER_API_KEY", "")

    with gr.Blocks(title="Open Router Image Generator") as app:
        gr.Markdown("# Open Router Image Generator")

        with gr.Row():
            output_folder = gr.Textbox(
                label="Output Folder Base",
                value=default_output_folder,
                placeholder="結果出力フォルダパス"
            )

        with gr.Row():
            api_key = gr.Textbox(
                label="OpenRouter API Key",
                value=default_api_key,
                type="password",
                placeholder="API Key"
            )
        
        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=[
                    "google/gemini-3-pro-image-preview",
                    "black-forest-labs/flux.2-pro"
                ],
                value="google/gemini-3-pro-image-preview"
            )

        # prompt_info.yamlアップロード用
        with gr.Row():
            prompt_info_file = gr.File(
                label="prompt_info.yamlをアップロード",
                file_types=[".yaml", ".yml"],
                type="filepath",
                height=150
            )

        with gr.Row():
            prompt = gr.Textbox(
                label="Prompt",
                lines=5,
                placeholder="プロンプトを入力してください"
            )

        gr.Markdown("### Image Paths")
        
        visible_count = gr.State(value=1)  # 現在表示されているフォームの数

        # 画像パス入力フィールド (最大10個作成、デフォルト1個表示)
        image_path_inputs = []
        image_path_warnings = []
        image_previews = []
        image_uploads = []
        image_rows = []
        
        # 履歴を取得
        history_choices = get_history_choices()
        
        # 履歴ギャラリーのリストを保持
        history_galleries = []

        for i in range(10):
            with gr.Row(visible=(i < 1)) as row:
                image_rows.append(row)
                with gr.Column(scale=3):
                    image_path = gr.Dropdown(
                        label=f"Image Path {i+1}",
                        choices=history_choices,
                        allow_custom_value=True,
                        value="",
                        interactive=True
                    )
                    image_path_inputs.append(image_path)
                    
                    # 履歴から画像を選択するギャラリー
                    with gr.Accordion(f"履歴から画像を選択 {i+1}", open=False):
                        history_gallery = gr.Gallery(
                            label="画像履歴",
                            value=get_history_gallery(),
                            columns=5,
                            rows=2,
                            height=300,
                            object_fit="contain",
                            show_label=False
                        )
                        history_galleries.append(history_gallery)
                    
                    # 画像アップロード用
                    image_upload = gr.Image(
                        label=f"画像をアップロード/貼り付け {i+1}",
                        type="pil",
                        height=200,
                        sources=["upload", "clipboard"]
                    )
                    image_uploads.append(image_upload)

                    warning = gr.Markdown(
                        value="", elem_classes=["warning-text"])
                    image_path_warnings.append(warning)

                with gr.Column(scale=1):
                    preview = gr.Image(
                        label=f"Preview {i+1}",
                        height=100,
                        show_label=False,
                        interactive=False
                    )
                    image_previews.append(preview)

            # パス入力時のチェックとプレビュー更新
            image_path.change(
                fn=check_image_path,
                inputs=[image_path],
                outputs=[warning]
            )
            image_path.change(
                fn=load_image_preview,
                inputs=[image_path],
                outputs=[preview]
            )
            
            # 画像アップロード時の処理
            image_upload.change(
                fn=handle_image_upload,
                inputs=[image_upload],
                outputs=[image_path, preview]
            )
            
            # 履歴ギャラリーから画像を選択した時の処理
            history_gallery.select(
                fn=select_from_gallery,
                outputs=[image_path, preview]
            )
        
        # 画像フォーム追加・削除ボタン
        with gr.Row():
            add_image_btn = gr.Button("➕ 画像フォームを追加", size="sm")
            remove_image_btn = gr.Button("➖ 画像フォームを削除", size="sm")
        
        # 画像フォーム追加ボタンのイベント
        add_image_btn.click(
            fn=show_image_row,
            inputs=[visible_count],
            outputs=[*image_rows, visible_count]
        )
        
        # 画像フォーム削除ボタンのイベント
        remove_image_btn.click(
            fn=hide_image_row,
            inputs=[visible_count],
            outputs=[*image_rows, visible_count]
        )

        with gr.Row():
            run_btn = gr.Button("Run", variant="primary")

        with gr.Row():
            result_output = gr.Textbox(
                label="結果",
                lines=3,
                max_lines=20,
                interactive=False
            )

        with gr.Row():
            image_gallery = gr.Gallery(
                label="生成された画像",
                show_label=True,
                columns=3,
                height="auto"
            )

        # Runボタンのクリックイベント
        run_btn.click(
            fn=run_request,
            inputs=[output_folder, api_key, model_dropdown, prompt, *image_path_inputs],
            outputs=[result_output, image_gallery, *image_path_inputs]
        )

        # prompt_info.yamlアップロード時のイベント
        prompt_info_file.change(
            fn=load_prompt_info,
            inputs=[prompt_info_file],
            outputs=[prompt, *image_path_inputs, *image_previews, *image_rows, visible_count]
        )

        # カスタムCSS
        app.css = """
        .warning-text p {
            color: red;
            font-weight: bold;
            margin: 0;
            padding: 0;
        }
        """

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_port=7861)
