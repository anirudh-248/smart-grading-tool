import io
import cv2
from PIL import Image
from google.cloud import vision
from google.oauth2 import service_account
from .logger import get_logger
from .config import Config


logger = get_logger(__name__)
cfg = Config()

_VISION_CLIENT = None


def get_vision_client():
    global _VISION_CLIENT
    if _VISION_CLIENT is None:
        creds_path = cfg.get("ocr", {}).get("google_credentials")
        if not creds_path:
            raise ValueError("Google Vision credentials path not set in config.yaml")

        logger.info(f"Loading Google Vision credentials from: {creds_path}")

        credentials = service_account.Credentials.from_service_account_file(creds_path)
        _VISION_CLIENT = vision.ImageAnnotatorClient(credentials=credentials)

    return _VISION_CLIENT


def preprocess_image_for_ocr(image_path: str) -> bytes:
    """
    Preprocess image using OpenCV and return PNG bytes for Vision API.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

    pil_img = Image.fromarray(thresh)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def image_to_text(image_path: str, lang: str = None) -> str:
    try:
        logger.info(f"Running Google Vision OCR on {image_path}")

        image_bytes = preprocess_image_for_ocr(image_path)

        client = get_vision_client()
        vision_img = vision.Image(content=image_bytes)

        response = client.document_text_detection(image=vision_img)

        if response.error.message:
            raise RuntimeError(f"Google Vision OCR error: {response.error.message}")

        if not response.full_text_annotation:
            logger.warning("Google Vision returned no text.")
            return ""

        text = response.full_text_annotation.text.strip()
        logger.debug(f"OCR output (first 200 chars): {text[:200]}")
        return text

    except Exception:
        logger.exception("OCR failed using Google Vision")
        raise
