import os
from typing import Any, Dict
from .logger import get_logger

logger = get_logger(__name__)

def safe_load_text(text: str) -> str:
    if text is None:
        return ""
    return " ".join(text.split())

def save_json(path: str, data: Dict[str, Any]):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved JSON to {path}")
