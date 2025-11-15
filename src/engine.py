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
    Robust answer splitter:
    - Matches only VALID question markers using strict regex.
    - Removes duplicate/overlapping markers.
    - Splits cleanly even when OCR inserts extra markers.
    """
    logger.info("Starting robust answer splitting...")

    if not ocr_text:
        logger.warning("OCR text is empty.")
        return []

    # Normalize: collapse whitespace, but keep newlines for readability
    text = ocr_text.replace("\r", "")
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())

    # Strict question marker pattern
    pattern = re.compile(
        r"(Q(?:uestion)?\s*\d+[\.\):])",      # e.g., Q1.  / Question 1:
        flags=re.IGNORECASE
    )

    # Find all markers
    raw_matches = list(pattern.finditer(text))
    if not raw_matches:
        logger.warning("No valid question markers found. Returning full text.")
        return [text]

    # Remove duplicate markers (e.g., OCR detecting both Q2. and 2.)
    markers = []
    last_end = -1
    for m in raw_matches:
        if m.start() <= last_end:
            continue
        markers.append(m)
        last_end = m.end()

    logger.info(f"Cleaned Markers: {[m.group() for m in markers]}")

    # Extract answer blocks
    answers = []
    for i in range(len(markers)):
        start = markers[i].end()
        end = markers[i+1].start() if i+1 < len(markers) else len(text)
        chunk = text[start:end].strip()
        answers.append(chunk)

    logger.info(f"Extracted {len(answers)} answers after cleaning duplicates.")
    return answers


def evaluate_script(
    student_answers: Dict[int, str], 
    model_answers: Dict[int, str], 
    rubric: Dict[str, Any]
) -> Dict[str, Any]:

    logger.info("========== Starting Script Evaluation (PDF Mode) ==========")

    try:
        if not validate_rubric(rubric):
            logger.warning("Invalid rubric format detected.")

        # DEBUG PRINT INPUTS
        logger.info("Received student answers and model answers directly (no OCR).")

        # Combine textbook content for keyword extraction
        logger.info("Extracting keywords from model answers...")
        combined_textbook = " ".join([a for a in model_answers.values() if a])
        textbook_keywords = extract_keywords(combined_textbook)
        logger.info(f"Extracted textbook keywords: {len(textbook_keywords)}")

        results = []
        total_score = 0.0

        logger.info("STEP 1: Beginning question-by-question evaluation...")

        for q in rubric.get("questions", []):
            qid = q.get("question_id")
            logger.info(f"--- Evaluating Question {qid} ---")

            # Student answer mapping
            student_answer = student_answers.get(qid, "")
            model_answer = model_answers.get(qid, "")

            logger.debug(f"Student Answer for Q{qid}: {student_answer}")

            # Similarity scoring
            logger.info(f"Q{qid}: Calculating semantic similarity...")
            sim = similarity_score(model_answer, student_answer)
            logger.info(f"Q{qid}: Similarity score = {sim}")

            # Quality scoring
            logger.info(f"Q{qid}: Calculating answer quality...")
            qual = quality_score(student_answer, max_score=1.0)
            logger.info(f"Q{qid}: Quality score = {qual}")

            # Rubric scoring
            logger.info(f"Q{qid}: Applying rubric scoring...")
            rubric_score = apply_rubric_to_answer(student_answer, q)
            logger.info(f"Q{qid}: Rubric score = {rubric_score}")

            # Combine weighted scores
            logger.info(f"Q{qid}: Combining weighted scores...")
            weights = cfg.get("weights", {"similarity": 0.6, "quality": 0.3, "rubric": 0.1})

            max_marks = float(q.get("max_marks", 0))
            rubric_norm = (rubric_score / max_marks) if max_marks > 0 else 0.0

            combined_norm = (
                weights["similarity"] * sim +
                weights["quality"] * qual +
                weights["rubric"] * rubric_norm
            )

            final_marks = combined_norm * max_marks
            logger.info(f"Q{qid}: Final marks = {final_marks}/{max_marks}")

            # Feedback generation
            feedback = []
            if sim < 0.4:
                feedback.append("Answer differs substantially from expected answer.")
            elif sim < 0.7:
                feedback.append("Answer partially matches expected answer.")
            else:
                feedback.append("Answer closely matches expected answer.")

            if qual < 0.5:
                feedback.append("Poor clarity/grammar detected.")
            else:
                feedback.append("Good clarity and coherence.")

            results.append({
                "question_id": qid,
                "student_answer": student_answer,
                "similarity_score": round(sim, 4),
                "quality_score": round(qual, 4),
                "rubric_score": round(rubric_score, 4),
                "final_marks": round(final_marks, 4),
                "max_marks": max_marks,
                "feedback": " ".join(feedback)
            })

            total_score += final_marks

        logger.info("STEP 2: Evaluation Completed.")
        logger.info(f"Total Score = {total_score}")

        return {
            "questions": results,
            "total_score": round(total_score, 4)
        }

    except Exception as e:
        logger.exception("FATAL ERROR during evaluate_script()")
        traceback.print_exc()
        raise
