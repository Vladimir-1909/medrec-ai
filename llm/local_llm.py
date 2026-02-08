from __future__ import annotations

import os
from typing import Dict, Any
from dotenv import load_dotenv

for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(key, None)

load_dotenv()


def get_llm_config() -> Dict[str, Any]:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
    api_key = os.getenv("LLM_API_KEY", "lm-studio")
    model_name = os.getenv("MODEL_NAME", "medgemma-1.5-4b-it")

    return {
        "config_list": [{
            "model": model_name,
            "base_url": base_url.rstrip("/"),
            "api_key": api_key or "not-needed",
            "api_type": "openai",
            "temperature": 0.3,
            "max_tokens": 2048,
            "timeout": 120,
        }],
        "temperature": 0.3,
        "max_tokens": 2048,
        "timeout": 120,
        "cache_seed": None,
    }
