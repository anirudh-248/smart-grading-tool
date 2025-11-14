from typing import Dict, Any
from .logger import get_logger
import re

logger = get_logger(__name__)

def validate_rubric(rubric: Dict[str, Any]) -> bool:
    required_keys = ['questions']
    if not isinstance(rubric, dict):
        return False
    for k in required_keys:
        if k not in rubric:
            return False
    # simple validation of question entries
    for q in rubric.get('questions', []):
        if 'question_id' not in q or 'max_marks' not in q:
            return False
    return True

def apply_rubric_to_answer(student_answer: str, rubric_for_question: Dict[str, Any]) -> float:
    """
    rubric_for_question example:
    {
      "question_id": 1,
      "max_marks": 5,
      "expected_keywords": ["binary search", "midpoint"],
      "penalties": {"length_penalty": {"min_words": 5, "deduct_per_missing_word": 0.5}},
      "bonus": {"contains_example": 0.5}
    }
    """
    try:
        max_marks = float(rubric_for_question.get('max_marks', 0))
        if max_marks == 0:
            return 0.0
        ans = (student_answer or "").lower()
        score = 0.0
        # 1. keyword match
        expected = [k.lower() for k in rubric_for_question.get('expected_keywords', [])]
        if expected:
            found = 0
            for kw in expected:
                # simple substring match (could be improved with fuzzy matching)
                if kw in ans:
                    found += 1
            keyword_fraction = found / len(expected)
            score += keyword_fraction * max_marks * 0.7  # keywords contribute 70% to rubric marks
        else:
            # if no expected keywords, allocate some base
            score += max_marks * 0.2

        # 2. length / penalties
        penalties = rubric_for_question.get('penalties', {})
        words = ans.split()
        if 'length_penalty' in penalties:
            lp = penalties['length_penalty']
            min_words = lp.get('min_words', 0)
            deduct = lp.get('deduct_per_missing_word', 0.5)
            if len(words) < min_words:
                missing = max(0, min_words - len(words))
                penalty_marks = missing * deduct
                score -= penalty_marks
        # 3. bonus
        bonus_total = 0.0
        bonus = rubric_for_question.get('bonus', {})
        for k, v in bonus.items():
            # simple check: if keyword of bonus in answer, add bonus
            if isinstance(v, (int, float)) and k in ans:
                bonus_total += float(v)
        score += bonus_total

        # clamp to [0, max_marks]
        score = max(0.0, min(max_marks, float(score)))
        logger.debug(f"Rubric score before clamp: {score} for max {max_marks}")
        return score
    except Exception as e:
        logger.exception("apply_rubric_to_answer failed")
        return 0.0
