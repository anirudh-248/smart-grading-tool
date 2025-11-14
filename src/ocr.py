import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image
import cv2
import numpy as np
from typing import Dict
from .logger import get_logger
from .config import Config

logger = get_logger(__name__)
cfg = Config()

def preprocess_image_for_ocr(image_path: str) -> np.ndarray:
    """Read image and apply common preprocessing for OCR"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 2)
    return thresh

def image_to_text(image_path: str, lang: str = None) -> str:
    try:
        lang = lang or cfg.get('ocr', {}).get('lang', 'eng')
        logger.info(f"Running OCR on {image_path} with lang={lang}")
        proc = preprocess_image_for_ocr(image_path)
        pil = Image.fromarray(proc)
        text = pytesseract.image_to_string(pil, lang=lang)
        text = text.strip()
        logger.debug(f"OCR output (first 200 chars): {text[:200]}")
        return text
    except Exception as e:
        logger.exception("OCR failed")
        raise
