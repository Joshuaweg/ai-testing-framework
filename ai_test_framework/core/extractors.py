from __future__ import annotations
import re
from typing import Optional


def extract(text: str, method: str) -> Optional[str]:
    if method == "numeric":
        return _extract_numeric(text)
    if method == "yn":
        return _extract_yn(text)
    if method == "letter":
        return _extract_letter(text)
    if method == "exact":
        return text.strip().lower()
    raise ValueError(f"Unknown extraction method: {method!r}")


def _extract_numeric(text: str) -> Optional[str]:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    val = match.group()
    # strip trailing .0 from whole numbers
    if val.endswith(".0"):
        val = val[:-2]
    return val


def _extract_yn(text: str) -> Optional[str]:
    lower = text.lower()
    if re.search(r"\byes\b", lower):
        return "yes"
    if re.search(r"\bno\b", lower):
        return "no"
    return None


def _extract_letter(text: str) -> Optional[str]:
    match = re.search(r"\b([A-Da-d])\b", text)
    return match.group(1).upper() if match else None
