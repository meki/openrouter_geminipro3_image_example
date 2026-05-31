# OpenRouter Image Editing Example

This project demonstrates how to use OpenRouter image generation and editing models, including `recraft/recraft-v4.1-pro-vector`, `recraft/recraft-v4.1-vector`, `x-ai/grok-imagine-image-quality`, `openai/gpt-5.4-image-2`, `recraft/recraft-v4-pro`, and `recraft/recraft-v4`.

## Setup

uv Create a virtual environment and install dependencies:

```bash
uv venv
uv sync
```

```bash
cp .env.example .env
# Edit the .env file to add your OpenRouter API key and desired output folder path.
```

## Usage

Set `model`, `text`, and `image_paths` in `prompt_info.yaml` as needed. Then run the main script:

```bash
uv run python core.py
```