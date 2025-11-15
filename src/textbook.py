from rake_nltk import Rake
from typing import List
from .logger import get_logger
from .config import Config
import nltk


logger = get_logger(__name__)
cfg = Config()


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading punkt")
    nltk.download('punkt')


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading stopwords")
    nltk.download('stopwords')


def extract_keywords(text: str, num_keywords: int = None) -> List[str]:
    if not text:
        return []
    num_keywords = num_keywords or cfg.get('keyword', {}).get('num_keywords', 15)
    r = Rake()
    r.extract_keywords_from_text(text)
    ranked = r.get_ranked_phrases()
    logger.debug(f"Extracted {len(ranked)} keywords/phrases")
    return ranked[:num_keywords]
