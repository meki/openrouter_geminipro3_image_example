import os
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
import gradio as gr
import yaml
from core import gemini_pro_3_image_preview_request, flux_2_pro_image_preview_request, speedream_4_5_image_preview_request, save_response_images, get_image_from_base64, base64_url_to_base64_image
from utility import (
    add_to_history,
    get_history_choices,
    add_to_favorites,
    remove_from_favorites,
    is_favorite,
    get_history_gallery,
    load_image_preview,
    check_image_path,
    handle_image_upload
)


def select_from_gallery(evt: gr.SelectData, displayed_paths):
    """ギャラリーから画像を選択したときの処理
    
    Args:
        evt: GradioのSelectDataイベント
        displayed_paths: ギャラリーに表示されている画像パスのリスト
    """
    if evt.index < len(displayed_paths):
        selected_path = displayed_paths[evt.index]
        try:
            preview = Image.open(selected_path) if Path(selected_path).exists() else None
            return selected_path, preview
        except Exception:
            return selected_path, None
    return "", None


def update_gallery_display(filter_mode):
    """ギャラリーの表示を更新"""
    gallery_items, displayed_paths = get_history_gallery(filter_mode)
    return gallery_items, displayed_paths


def toggle_favorite(current_path, filter_mode):
    """お気に入りの追加/削除を切り替え"""
    if not current_path or current_path.strip() == "":
        gallery_items, displayed_paths = get_history_gallery(filter_mode)
        return gallery_items, displayed_paths, "画像パスを選択してください"
    
    current_path = current_path.strip('"')
    
    if is_favorite(current_path):
        remove_from_favorites(current_path)
        message = f"お気に入りから削除しました: {Path(current_path).name}"
    else:
        add_to_favorites(current_path)
        message = f"お気に入りに追加しました: {Path(current_path).name}"
    
    # ギャラリーを更新して返す
    gallery_items, displayed_paths = get_history_gallery(filter_mode)
    return gallery_items, displayed_paths, message


def show_image_row(current_count):
    """画像フォームの表示数を増やす"""
    new_count = min(current_count + 1, 10)  # 最大10個まで
    updates = []
    for i in range(10):
        updates.append(gr.update(visible=(i < new_count)))
    updates.append(new_count)
    return updates


def hide_image_row(current_count):
    """画像フォームの表示数を減らす"""
    new_count = max(current_count - 1, 1)  # 最低1個は表示
    updates = []
    for i in range(10):
        updates.append(gr.update(visible=(i < new_count)))
    updates.append(new_count)
    return updates


def load_prompt_info(file):
    """
prompt_info.yamlを読み込んでフォームに流し込む"""
    if file is None:
        # ファイルがない場合は全て空で返す
        empty_paths = [""] * 10
        empty_previews = [None] * 10
        empty_rows = [gr.update(visible=(i < 1)) for i in range(10)]
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
                # ダブルクォートで囲まれている場合は除去
                if isinstance(path, str):
                    path = path.strip('"')
                paths.append(path)
                # 画像プレビューを読み込み
                preview = Image.open(path) if path and Path(path).exists() else None
                previews.append(preview)
            else:
                paths.append("")
                previews.append(None)
        
        # Rowの表示設定（画像数分表示する）
        row_updates = [gr.update(visible=(i < num_images)) for i in range(10)]
        
        return prompt_text, *paths, *previews, *row_updates, num_images

    except Exception:
        # エラー時は全て空で返す
        empty_paths = [""] * 10
        empty_previews = [None] * 10
        empty_rows = [gr.update(visible=(i < 1)) for i in range(10)]
        return "", *empty_paths, *empty_previews, *empty_rows, 1


def run_request(output_folder, api_key, model, prompt, *args):
    """リクエストを実行して結果を返す
    
    Args:
        output_folder: 出力フォルダ
        api_key: APIキー
        model: モデル名
        prompt: プロンプト
        *args: image_path_inputs (10個) + filter_radios (10個)
    
    Returns:
        result_text, image_gallery, *dropdown_updates (10), *gallery_updates (10), *state_updates (10)
    """
    # 引数を分解
    image_paths = args[:10]  # 最初の10個が画像パス
    filter_modes = args[10:20] if len(args) >= 20 else ["全て"] * 10  # 次の10個がフィルターモード
    
    # 空のパスをフィルタリング
    valid_image_paths = [p for p in image_paths if p and p.strip() != ""]
    valid_image_paths = [p.strip('"') for p in valid_image_paths]

    # エラー時のデフォルト返り値を作成する関数
    def create_error_response(message):
        # エラー時も履歴更新は省略
        dropdown_updates = [gr.Dropdown()] * 10
        gallery_updates = [gr.Gallery()] * 10
        state_updates = [gr.State()] * 10
        return message, None, *dropdown_updates, *gallery_updates, *state_updates

    if not valid_image_paths:
        return create_error_response("エラー: 少なくとも1つの画像パスを指定してください")

    # パスの存在確認
    for path in valid_image_paths:
        if not Path(path).exists():
            return create_error_response(f"エラー: 画像パスが存在しません: {path}")

    if not prompt or prompt.strip() == "":
        return create_error_response("エラー: プロンプトを入力してください")

    if not api_key or api_key.strip() == "":
        return create_error_response("エラー: OpenRouter API Keyを入力してください")
    
    # 画像パスを履歴に追加
    for path in valid_image_paths:
        add_to_history(path)

    try:
        # モデルに応じてリクエスト実行
        if model == "google/gemini-3-pro-image-preview":
            response = gemini_pro_3_image_preview_request(
                prompt, valid_image_paths, api_key)
        elif model == "black-forest-labs/flux.2-pro":
            response = flux_2_pro_image_preview_request(
                prompt, valid_image_paths, api_key)
        elif model == "bytedance-seed/seedream-4.5":
            response = speedream_4_5_image_preview_request(
                prompt, valid_image_paths, api_key)

        if response.status_code != 200:
            return create_error_response(f"エラー: {response.status_code}\n{response.text}")

        response_data = response.json()
        
        prompt_info_data = {
            "text": prompt,
            "image_paths": valid_image_paths
        }

        output_folder_path, saved_image_paths = save_response_images(
            Path(output_folder), response_data, prompt_info_data
        )

        # レスポンスから結果テキストを取得
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
            result += f"保存先: {output_folder_path}"
        else:
            result = f"✅ 成功!\n\n結果:\n{result_text}\n\n"
            result += f"生成された画像数: {len(images)}\n"
            result += f"保存先: {output_folder_path}"

        # 保存済みファイルからPIL画像を読み込み（base64デコードの重複処理を回避）
        pil_images = []
        for saved_path in saved_image_paths:
            try:
                pil_image = Image.open(saved_path)
                pil_images.append(pil_image)
            except Exception as e:
                print(f"Failed to load saved image {saved_path}: {e}")
        
        # ドロップダウンとギャラリーの更新は省略（次回のユーザー操作時に自動更新される）
        # これによりファイル保存完了後の待ち時間を最小化
        dropdown_updates = [gr.Dropdown()] * 10  # 更新しない
        gallery_updates = [gr.Gallery()] * 10  # 更新しない
        state_updates = [gr.State()] * 10  # 更新しない
        
        return result, pil_images if pil_images else None, *dropdown_updates, *gallery_updates, *state_updates

    except Exception as e:
        return create_error_response(f"エラーが発生しました: {str(e)}")


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
                    "black-forest-labs/flux.2-pro",
                    "bytedance-seed/seedream-4.5"
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
        gallery_path_states = []  # 各ギャラリーの表示パスリストを保持するState
        filter_radios = []
        favorite_buttons = []
        favorite_messages = []

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
                        with gr.Row():
                            filter_radio = gr.Radio(
                                choices=["全て", "お気に入りのみ"],
                                value="全て",
                                label="表示フィルター",
                                scale=2
                            )
                            filter_radios.append(filter_radio)
                            
                            favorite_btn = gr.Button(
                                "★ お気に入り切替",
                                size="sm",
                                scale=1
                            )
                            favorite_buttons.append(favorite_btn)
                        
                        favorite_msg = gr.Markdown(value="", elem_classes=["favorite-message"])
                        favorite_messages.append(favorite_msg)
                        
                        # ギャラリーの表示パスリストを保持するState
                        gallery_items, displayed_paths = get_history_gallery("all")
                        gallery_path_state = gr.State(value=displayed_paths)
                        gallery_path_states.append(gallery_path_state)
                        
                        history_gallery = gr.Gallery(
                            label="画像履歴",
                            value=gallery_items,
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
            
            # フィルター切り替え時にギャラリーを更新
            filter_radio.change(
                fn=lambda mode: update_gallery_display("favorites" if mode == "お気に入りのみ" else "all"),
                inputs=[filter_radio],
                outputs=[history_gallery, gallery_path_state]
            )
            
            # お気に入りボタンのクリック処理
            favorite_btn.click(
                fn=lambda path, mode: toggle_favorite(path, "favorites" if mode == "お気に入りのみ" else "all"),
                inputs=[image_path, filter_radio],
                outputs=[history_gallery, gallery_path_state, favorite_msg]
            )
            
            # 履歴ギャラリーから画像を選択した時の処理
            history_gallery.select(
                fn=select_from_gallery,
                inputs=[gallery_path_state],
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
            inputs=[output_folder, api_key, model_dropdown, prompt, *image_path_inputs, *filter_radios],
            outputs=[result_output, image_gallery, *image_path_inputs, *history_galleries, *gallery_path_states]
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
        .favorite-message p {
            color: #2563eb;
            font-weight: bold;
            margin: 5px 0;
            padding: 0;
        }
        """

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_port=7861)
