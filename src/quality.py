from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict
from .logger import get_logger
from language_tool_python import LanguageTool
import subprocess
import shutil

logger = get_logger(__name__)
_SENT_ANALYZER = SentimentIntensityAnalyzer()

def sentiment_score(text: str) -> Dict[str, float]:
    if not text:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}
    scores = _SENT_ANALYZER.polarity_scores(text)
    logger.debug(f"Sentiment scores: {scores}")
    return scores

def _language_tool_available() -> bool:
    # language_tool_python needs Java; check if javac/java exists
    java_path = shutil.which("java") or shutil.which("javac")
    return bool(java_path)

def grammar_issues_count(text: str) -> int:
    """
    Try to use LanguageTool if available; otherwise fallback to heuristic.
    Returns number of grammar issues detected.
    """
    if not text:
        return 0
    if _language_tool_available():
        try:
            tool = LanguageTool('en-US')
            matches = tool.check(text)
            # filter matches for grammar/typo types (severity may vary)
            issues = len(matches)
            logger.debug(f"LanguageTool issues: {issues}")
            return issues
        except Exception as e:
            logger.exception("LanguageTool check failed; falling back to heuristic")
    # fallback heuristic: count multiple consecutive punctuation or suspicious tokens
    import re
    sentences = re.split(r'[.!?]+', text)
    # basic heuristic: many very short fragments -> poor coherence
    short_fragments = sum(1 for s in sentences if len(s.strip().split()) < 3)
    # count occurrences of obvious grammar signals:
    repeated_punct = len(re.findall(r'[\!\?]{2,}|\.\.{2,}', text))
    issues = short_fragments + repeated_punct
    logger.debug(f"Fallback heuristic issues: {issues}")
    return issues

def quality_score(text: str, max_score: float = 1.0) -> float:
    """
    Combines sentiment neutrality (not strictly necessary) and coherence (grammar)
    into a quality score between 0 and max_score.
    Lower grammar issues -> higher score. This is intentionally simple and tunable.
    """
    if not text:
        return 0.0
    sentiment = sentiment_score(text)
    compound = sentiment.get("compound", 0.0)
    # grammar
    issues = grammar_issues_count(text)
    # map issues to penalty; simple mapping:
    penalty = min(1.0, issues * 0.05)  # each issue reduces by 0.05, cap at 1.0
    # sentiment: reward neutral to slightly positive answers (student answers shouldn't be extremely negative)
    sentiment_factor = (compound + 1) / 2  # maps [-1,1] -> [0,1]
    # combine: give more weight to grammar (coherence)
    score = max(0.0, min(max_score, (0.7 * (1 - penalty) + 0.3 * sentiment_factor) * max_score))
    logger.debug(f"Quality components -> compound: {compound}, issues: {issues}, score: {score}")
    return score
