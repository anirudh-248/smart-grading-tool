import traceback
from typing import Dict, Any, List
from .logger import get_logger
from .ocr import image_to_text
from .textbook import extract_keywords
from .semantic import similarity_score
from .quality import quality_score
from .rubric import validate_rubric, apply_rubric_to_answer
from .config import Config
from .utils import safe_load_text
import re

logger = get_logger(__name__)
cfg = Config()

def split_answers_from_ocr(ocr_text: str) -> List[str]:
    """
    Heuristic to split the OCR text into answers per question.
    Best-effort: assumes answers are labeled like 'Q1.' or '1.' or 'Question 1'.
    Falls back to splitting on double newlines.
    Returns list of answer strings (ordered).
    """
    if not ocr_text:
        return []
    # Normalize
    text = ocr_text.replace('\r\n', '\n')
    # Try to split by question markers
    pattern = re.compile(r'(?:^|\n)\s*(?:Q|Question)?\s*([0-9]{1,3})\s*[:\.\)]', flags=re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if matches and len(matches) > 1:
        answers = []
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            answers.append(text[start:end].strip())
        logger.info(f"Detected {len(answers)} answers by question markers")
        return answers
    # fallback: split by two or more newlines
    parts = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    logger.info(f"Fallback splitting produced {len(parts)} answer chunks")
    return parts

def evaluate_script(image_path: str, model_answers: Dict[int, str], rubric: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main unified evaluation function.

    - image_path: path to scanned student script image (or image with multiple answers).
    - model_answers: dict mapping question_id -> model answer string.
    - rubric: rubric dictionary (see README or expected format).

    Returns dict with required structure.
    """
    try:
        # validate rubric
        if not validate_rubric(rubric):
            logger.warning("Invalid rubric format; proceeding but rubric scores may be zeroed.")
        # 1. OCR
        ocr_text = image_to_text(image_path)
        ocr_text = safe_load_text(ocr_text)
        # 2. Split answers heuristically
        student_chunks = split_answers_from_ocr(ocr_text)
        # Map student_chunks to question ids from rubric if possible, else by order
        question_entries = rubric.get('questions', [])
        results = []
        total_score = 0.0

        # Precompute textbook keywords from model answers (optional)
        combined_textbook = " ".join([a for a in model_answers.values() if a])
        textbook_keywords = extract_keywords(combined_textbook)

        # iterate rubric-specified questions
        for idx, q in enumerate(question_entries):
            qid = q.get('question_id')
            max_marks = float(q.get('max_marks', 0))
            # attempt to find corresponding student answer by label
            student_answer = ""
            # try to find by leading "Q{qid}" in OCR text
            pattern = re.compile(r'(?:^|\n)\s*(?:Q|Question)?\s*' + re.escape(str(qid)) + r'\s*[:\.\)]', flags=re.IGNORECASE)
            m = pattern.search(ocr_text)
            if m:
                # find end position
                # find next question marker after this
                next_match = pattern.search(ocr_text, m.end())
                start = m.end()
                end = next_match.start() if next_match else len(ocr_text)
                student_answer = ocr_text[start:end].strip()
            else:
                # fallback to by order mapping
                if idx < len(student_chunks):
                    student_answer = student_chunks[idx]
                else:
                    student_answer = ""

            # compute similarity
            model_answer = model_answers.get(qid, "")
            sim = similarity_score(model_answer, student_answer)

            # compute quality
            qual = quality_score(student_answer, max_score=1.0)

            # rubric score
            rubric_score = apply_rubric_to_answer(student_answer, q) if validate_rubric(rubric) else 0.0

            # combine to final marks according to weights in config
            weights = cfg.get('weights', {'similarity': 0.6, 'quality': 0.3, 'rubric': 0.1})
            sim_w = weights.get('similarity', 0.6)
            qual_w = weights.get('quality', 0.3)
            rub_w = weights.get('rubric', 0.1)

            # normalize rubric_score to 0..1 relative to max_marks
            rubric_norm = (rubric_score / max_marks) if max_marks > 0 else 0.0

            combined_norm = sim_w * sim + qual_w * qual + rub_w * rubric_norm
            final_marks = combined_norm * max_marks

            # feedback generation (simple)
            feedback_lines = []
            if sim < 0.4:
                feedback_lines.append("Answer differs substantially from expected answer; consider revising core concepts.")
            elif sim < 0.7:
                feedback_lines.append("Answer captures some key points but misses important details.")
            else:
                feedback_lines.append("Answer closely matches the expected answer.")

            if qual < 0.5:
                feedback_lines.append("Improve clarity and sentence structure; check grammar and coherence.")
            else:
                feedback_lines.append("Good clarity and coherence.")

            # rubric-specific feedback
            missing_expected = []
            for kw in q.get('expected_keywords', []):
                if kw.lower() not in (student_answer or "").lower():
                    missing_expected.append(kw)
            if missing_expected:
                feedback_lines.append(f"Missing expected keywords/concepts: {', '.join(missing_expected)}")

            feedback = " ".join(feedback_lines)

            results.append({
                "question_id": qid,
                "student_answer": student_answer,
                "similarity_score": round(float(sim), 4),
                "quality_score": round(float(qual), 4),
                "rubric_score": round(float(rubric_score), 4),
                "final_marks": round(float(final_marks), 4),
                "max_marks": max_marks,
                "feedback": feedback
            })
            total_score += final_marks

        out = {
            "questions": results,
            "total_score": round(float(total_score), 4)
        }
        return out

    except Exception as e:
        logger.exception("evaluate_script failed: " + str(e))
        traceback.print_exc()
        raise
