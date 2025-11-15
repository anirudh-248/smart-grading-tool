from sentence_transformers import SentenceTransformer, util
from typing import List
from .config import Config
from .logger import get_logger


logger = get_logger(__name__)
cfg = Config()

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        model_name = cfg.get('similarity', {}).get('model_name', 'all-MiniLM-L6-v2')
        logger.info(f"Loading sentence-transformers model: {model_name}")
        _MODEL = SentenceTransformer(model_name)
    return _MODEL


def similarity_score(model_answer: str, student_answer: str) -> float:
    try:
        model = _get_model()
        if (not model_answer) or (not student_answer):
            return 0.0
        emb_model = model.encode(model_answer, convert_to_tensor=True)
        emb_student = model.encode(student_answer, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(emb_model, emb_student).item()
        sim = max(0.0, min(1.0, float(sim)))
        logger.debug(f"Similarity: {sim}")
        return sim
    except Exception as e:
        logger.exception("Semantic similarity failed")
        return 0.0

def batch_similarity(model_answers: List[str], student_answers: List[str]) -> List[float]:
    return [similarity_score(m, s) for m, s in zip(model_answers, student_answers)]
