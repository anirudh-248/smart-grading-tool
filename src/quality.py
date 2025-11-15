import re
import multiprocessing as mp
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict
from .logger import get_logger
from .config import Config


logger = get_logger(__name__)
cfg = Config()
_SENT_ANALYZER = SentimentIntensityAnalyzer()


def sentiment_score(text: str) -> Dict[str, float]:
    if not text:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 0.0}
    scores = _SENT_ANALYZER.polarity_scores(text)
    logger.debug(f"Sentiment scores: {scores}")
    return scores


def _language_tool_available() -> bool:
    return False


def _lt_worker(text: str, q: mp.Queue):
    try:
        from language_tool_python import LanguageTool
        tool = LanguageTool('en-US')
        matches = tool.check(text)
        q.put({"ok": len(matches)})
    except Exception as e:
        q.put({"error": str(e)})


def grammar_issues_count(text: str) -> int:
    if not text:
        return 0
    timeout_sec = cfg.get("quality", {}).get("languagetool_timeout_seconds", 6)
    if not _language_tool_available():
        logger.debug("Using heuristic grammar check (LanguageTool unavailable).")
        return _heuristic_grammar_issues(text)
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_lt_worker, args=(text, q))
    p.daemon = True
    start = time.time()
    try:
        p.start()
        p.join(timeout=timeout_sec)
        if p.is_alive():
            logger.warning("LanguageTool check timed out; terminating and using heuristic fallback.")
            p.terminate()
            p.join(1)
            return _heuristic_grammar_issues(text)
        if not q.empty():
            res = q.get()
            if "ok" in res:
                issues = int(res["ok"])
                logger.info(f"LanguageTool reported {issues} issues (took {time.time()-start:.2f}s).")
                return issues
            else:
                logger.warning(f"LanguageTool worker returned error: {res.get('error')}. Using heuristic fallback.")
                return _heuristic_grammar_issues(text)
        else:
            logger.warning("LanguageTool returned no result; using heuristic fallback.")
            return _heuristic_grammar_issues(text)
    except Exception:
        logger.exception("LanguageTool check failed unexpectedly; using heuristic fallback.")
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass
        return _heuristic_grammar_issues(text)


def _heuristic_grammar_issues(text: str) -> int:
    sentences = re.split(r'[.!?]+', text)
    short_fragments = sum(1 for s in sentences if len(s.strip().split()) < 3 and len(s.strip()) > 0)
    repeated_punct = len(re.findall(r'[\!\?]{2,}|\.\.{2,}', text))
    single_letter_tokens = sum(1 for tok in text.split() if len(tok) == 1)
    issues = short_fragments + repeated_punct + int(single_letter_tokens / 10)
    logger.debug(f"Heuristic grammar issues: short={short_fragments}, repeated={repeated_punct}, letters={single_letter_tokens}, total={issues}")
    return int(issues)


def quality_score(text: str, max_score: float = 1.0) -> float:
    if not text:
        return 0.0
    try:
        sentiment = sentiment_score(text)
        compound = sentiment.get("compound", 0.0)
        issues = grammar_issues_count(text)
        penalty = min(1.0, issues * 0.05)
        sentiment_factor = (compound + 1) / 2
        score = max(0.0, min(max_score, (0.7 * (1 - penalty) + 0.3 * sentiment_factor) * max_score))
        logger.info(f"Computed quality score: {score:.4f} (issues={issues}, compound={compound:.3f})")
        return score
    except Exception:
        logger.exception("Quality scoring failed; returning 0.0")
        return 0.0
