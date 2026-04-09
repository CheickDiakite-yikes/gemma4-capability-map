from __future__ import annotations

import argparse
import json

import requests

from gemma4_capability_map.models.runtime_utils import detect_hf_token_source, load_hf_auth_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    args = parser.parse_args()

    token = load_hf_auth_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    api_url = f"https://huggingface.co/api/models/{args.model}"
    model_info = requests.get(api_url, headers=headers, timeout=30).json()
    siblings = {item["rfilename"] for item in model_info.get("siblings", [])}
    config_head = requests.head(
        f"https://huggingface.co/{args.model}/resolve/main/config.json",
        headers=headers,
        allow_redirects=True,
        timeout=30,
    )
    model_head = requests.head(
        f"https://huggingface.co/{args.model}/resolve/main/model.safetensors",
        headers=headers,
        allow_redirects=True,
        timeout=30,
    )
    output = {
        "model": args.model,
        "auth_token_present": bool(token),
        "auth_token_source": detect_hf_token_source(),
        "gated": model_info.get("gated"),
        "pipeline_tag": model_info.get("pipeline_tag"),
        "auto_model": model_info.get("transformersInfo", {}).get("auto_model"),
        "processor": model_info.get("transformersInfo", {}).get("processor"),
        "siblings": sorted(siblings),
        "config_status": config_head.status_code,
        "model_status": model_head.status_code,
        "model_content_length": int(model_head.headers.get("Content-Length", "0")),
        "model_size_gb": round(int(model_head.headers.get("Content-Length", "0")) / (1024 ** 3), 2),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
