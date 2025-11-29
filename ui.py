import os
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
import yaml
from core import gemini_pro_3_image_preview_request, save_response_images, get_image_from_base64, base64_url_to_base64_image


def check_image_path(path):
    """画像パスが存在するかチェック"""
    if not path or path.strip() or path.strip('"') == "":
        return ""
    if not Path(path).exists():
        return f"⚠️ パスが存在しません: {path}"
    return ""


def run_request(output_folder, api_key, prompt, *image_paths):
    """リクエストを実行して結果を返す"""
    # 空のパスをフィルタリング
    valid_image_paths = [p for p in image_paths if p and p.strip() != ""]

    valid_image_paths = [p.strip('"') for p in valid_image_paths]
    
    if not valid_image_paths:
        return "エラー: 少なくとも1つの画像パスを指定してください", None
    
    # パスの存在確認
    for path in valid_image_paths:
        if not Path(path).exists():
            return f"エラー: 画像パスが存在しません: {path}", None
    
    if not prompt or prompt.strip() == "":
        return "エラー: プロンプトを入力してください", None
    
    if not api_key or api_key.strip() == "":
        return "エラー: OpenRouter API Keyを入力してください", None
    
    try:
        # リクエスト実行
        response = gemini_pro_3_image_preview_request(prompt, valid_image_paths, api_key)
        
        if response.status_code != 200:
            return f"エラー: {response.status_code}\n{response.text}", None
        
        # 結果を保存
        output_folder_path = save_response_images(Path(output_folder), response)
        
        # レスポンスから結果テキストを取得
        response_data = response.json()
        result_text = response_data.get("choices", [])[0].get("message", {}).get("content", "")
        images = response_data.get("choices", [])[0].get("message", {}).get("images", [])
        
        # prompt_info.yamlを保存
        prompt_info_output_path = output_folder_path / "prompt_info.yaml"
        
        prompt_info_data = {
            "text": prompt,
            "image_paths": valid_image_paths
        }
        
        prompt_info_output_path.write_text(yaml.dump(prompt_info_data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        
        # 画像が0枚の場合はfinish_reasonを表示
        if len(images) == 0:
            native_finish_reason = response_data.get("choices", [])[0].get("native_finish_reason", "不明")
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
        
        return result, pil_images if pil_images else None
        
    except Exception as e:
        return f"エラーが発生しました: {str(e)}", None


def create_ui():
    load_dotenv()
    
    default_output_folder = os.getenv("OUTPUT_BASE_FOLDER", "")
    default_api_key = os.getenv("OPENROUTER_API_KEY", "")
    
    with gr.Blocks(title="Gemini Pro 3 Image Preview") as app:
        gr.Markdown("# Gemini Pro 3 Image Preview")
        
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
            prompt = gr.Textbox(
                label="Prompt",
                lines=5,
                placeholder="プロンプトを入力してください"
            )
        
        gr.Markdown("### Image Paths")
        
        # 画像パス入力フィールド (デフォルト3個)
        image_path_inputs = []
        image_path_warnings = []
        
        for i in range(3):
            with gr.Row():
                image_path = gr.Textbox(
                    label=f"Image Path {i+1}",
                    placeholder="画像パスを入力"
                )
                image_path_inputs.append(image_path)
            
            warning = gr.Markdown(value="", elem_classes=["warning-text"])
            image_path_warnings.append(warning)
            
            # パス入力時のチェック
            image_path.change(
                fn=check_image_path,
                inputs=[image_path],
                outputs=[warning]
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
            inputs=[output_folder, api_key, prompt, *image_path_inputs],
            outputs=[result_output, image_gallery]
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
