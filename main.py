import pdfplumber
import re
import json
import unicodedata
from typing import Dict, Any
from src.engine import similarity_score, quality_score, apply_rubric_to_answer, validate_rubric, cfg
import logging


logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\(cid:\d+\)", "", text)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_pdf_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()


def parse_schema(schema_text: str):
    blocks = re.split(r"\bQ(\d+)\s*[:.)]", schema_text)
    model_answers = {}
    rubric = {"questions": []}
    for i in range(1, len(blocks), 2):
        qid = int(blocks[i])
        content = blocks[i+1].strip()
        match = re.search(r"Answer\s*:\s*(.*)", content, flags=re.DOTALL | re.IGNORECASE)
        if match:
            model_answer = clean_text(match.group(1).strip())
        else:
            model_answer = ""
        model_answers[qid] = model_answer
        rubric["questions"].append({
            "question_id": qid,
            "max_marks": 5,
            "expected_keywords": [],
            "penalties": {},
            "bonus": {}
        })
    return model_answers, rubric


def parse_student_answers(student_text: str) -> Dict[int, str]:
    qa = {}
    blocks = re.split(r"\bQ(\d+)\s*[:.)]", student_text)
    for i in range(1, len(blocks), 2):
        qid = int(blocks[i])
        raw_block = blocks[i+1].strip()
        match = re.search(r"Answer\s*:\s*(.*)", raw_block, flags=re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1)
        else:
            answer = raw_block
        qa[qid] = clean_text(answer)
    return qa


def evaluate_script(student_answers: Dict[int, str], model_answers: Dict[int, str], rubric: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("========== Starting Script Evaluation (PDF MODE) ==========")
    try:
        if not validate_rubric(rubric):
            logger.warning("Invalid rubric detected.")
        results = []
        total_score = 0.0
        for q in rubric.get("questions", []):
            qid = q.get("question_id")
            student_answer = student_answers.get(qid, "")
            model_answer = model_answers.get(qid, "")
            sim = similarity_score(model_answer, student_answer)
            qual = quality_score(student_answer, max_score=1.0)
            rubric_score = apply_rubric_to_answer(student_answer, q)
            weights = cfg.get("weights", {"similarity": 0.6, "quality": 0.3, "rubric": 0.1})
            max_marks = float(q.get("max_marks", 5))
            rubric_norm = (rubric_score / max_marks) if max_marks else 0.0
            combined = (
                weights["similarity"] * sim +
                weights["quality"] * qual +
                weights["rubric"] * rubric_norm
            )
            final_marks = combined * max_marks
            fb = []
            if sim < 0.4:
                fb.append("Answer differs substantially from expected answer.")
            elif sim < 0.7:
                fb.append("Answer partially matches expected answer.")
            else:
                fb.append("Answer closely matches expected answer.")
            if qual < 0.5:
                fb.append("Poor clarity/grammar detected.")
            else:
                fb.append("Good clarity and coherence.")
            results.append({
                "question_id": qid,
                "student_answer": student_answer,
                "similarity_score": round(sim, 4),
                "quality_score": round(qual, 4),
                "rubric_score": round(rubric_score, 4),
                "final_marks": round(final_marks, 4),
                "max_marks": max_marks,
                "feedback": " ".join(fb)
            })
            total_score += final_marks
        return {"questions": results, "total_score": round(total_score, 4)}
    except Exception:
        logger.exception("FATAL ERROR in evaluate_script()")
        raise


def run_evaluation(schema_pdf: str, student_pdf: str):
    print("\nExtracting Schema PDF...")
    schema_text = extract_pdf_text(schema_pdf)
    print("\nExtracting Student PDF...")
    student_text = extract_pdf_text(student_pdf)
    print("\nParsing Schema...")
    model_answers, rubric = parse_schema(schema_text)
    print("\nParsing Student Answers...")
    student_answers = parse_student_answers(student_text)
    print("\nRunning Evaluation...")
    result = evaluate_script(student_answers, model_answers, rubric)
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\nSaved result.json")
    return result


if __name__ == "__main__":
    SCHEMA_PDF = "./schema.pdf"
    STUDENT_PDF = "./answer.pdf"
    run_evaluation(SCHEMA_PDF, STUDENT_PDF)
