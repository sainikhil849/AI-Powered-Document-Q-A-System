"""Cheap, deterministic analysis of the question text (no LLM call).

Two things are detected:

* **Summary cue** - wording that asks about the document as a whole:
  "what is this pdf about", "summarize the document", "give me an overview",
  "main points", "explain this file", ... Such questions have no single
  passage that "contains the answer", so they need a different path
  (representative passages in reading order instead of similarity search).
* **Topic words** - the content words left after removing stop-words and
  generic words such as "pdf", "document", "explain", "summary". If a
  question with a summary cue still has topic words that occur in the
  document ("summarize the leave policies"), it is a focused question and
  goes through normal retrieval instead.

Rules are regular expressions on purpose: they are free, instant and testable.
The trade-off is that unusual phrasings fall back to the normal fact path.
"""

from __future__ import annotations

import re
from pathlib import PurePath

_TOKEN = re.compile(r"[a-z0-9]+")
DOC_NOUNS = r"(?:pdf|document|doc|file|text|paper|report|attachment|upload|handbook|manual|book)"

SUMMARY_CUES = [
    re.compile(r"\bsumm?ar(?:y|ies|ise|ize|ised|ized|ising|izing|isation|ization)\b"),
    re.compile(r"\b(?:overview|gist|synopsis|abstract|tl\s*;?\s*dr)\b"),
    re.compile(rf"\b{DOC_NOUNS}s?\s+(?:is\s+|all\s+)?about\b"),  # "what is this pdf about"
    re.compile(rf"\babout\s+(?:this|the|that|my|your)\s+{DOC_NOUNS}\b"),  # "tell me about the document"
    re.compile(r"\bwhat\s+is\s+(?:it|this|that)\s+(?:all\s+)?about\b"),
    re.compile(
        rf"\b(?:explain|describe|outline|introduce)\s+(?:me\s+)?(?:(?:this|the|that|my|your)\s+)?(?:(?:whole|entire|full)\s+)?{DOC_NOUNS}\b"
    ),
    re.compile(rf"\bwhat\s+(?:does|do)\s+(?:this|the|that)\s+{DOC_NOUNS}\s+(?:say|cover|contain|describe|discuss|explain|talk)\b"),
    re.compile(r"\b(?:main|key|important|major)\s+(?:points?|topics?|ideas?|takeaways?|sections?|themes?|highlights?)\b"),
    re.compile(r"\bwhat\s+(?:topics|sections|subjects)\s+(?:are|does|do|is)\b"),
]

STOPWORDS = frozenset(
    "a an and are as at be by can could did do does for from had has have how i if in into is it its "
    "many may me much my of on or our please shall should so than that the their them there these "
    "they this those to us was we were what when where which who whom why will with would you your".split()
)

# Words that describe the request itself rather than a topic inside the document.
GENERIC_WORDS = frozenset(
    "pdf pdfs document documents doc docs file files text paper papers report reports attachment "
    "upload uploaded content contents page pages summary summaries summarize summarise summarized "
    "summarised summarizing summarising overview gist synopsis abstract tl dr tldr explain explained "
    "explanation describe description outline introduce about main key important major point points "
    "topic topics idea ideas takeaway takeaways section sections theme themes highlight highlights "
    "subject subjects tell give show say says cover covers contain contains discuss discusses talk "
    "talks brief briefly short whole entire overall all everything something anything know understand "
    "it this that thing things kind sort handbook manual book guide guidebook brochure".split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def stem(token: str) -> str:
    """Very light stemming so "policies"/"policy" and "days"/"day" match."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def has_summary_cue(question: str) -> bool:
    text = question.lower()
    return any(pattern.search(text) for pattern in SUMMARY_CUES)


def topic_words(question: str) -> list[str]:
    """Content words of the question, in order, without stop-words and generic request words."""
    seen: list[str] = []
    for token in tokenize(question):
        if token in STOPWORDS or token in GENERIC_WORDS or token.isdigit() or len(token) < 2:
            continue
        if token not in seen:
            seen.append(token)
    return seen


def filename_stems(filename: str) -> set[str]:
    """Distinctive words of a filename: 'Employee_Handbook-2026.pdf' -> {'employee', 'handbook'}."""
    stem_part = PurePath(filename).stem
    return {stem(t) for t in tokenize(stem_part) if len(t) >= 3 and not t.isdigit() and t not in GENERIC_WORDS and t not in STOPWORDS}


def mentions_filename(question: str, filename: str) -> bool:
    """True when the question names the file: all words of the filename appear, in order.

    'employee_handbook.pdf' matches "summarize the employee handbook" but NOT
    "how many days do employees get" - a single shared word is not a mention.
    Filenames made only of generic words ('report_2025.pdf') never match.
    """
    name_tokens = [stem(t) for t in tokenize(PurePath(filename).stem) if not t.isdigit()]
    if not any(len(t) >= 3 and t not in GENERIC_WORDS and t not in STOPWORDS for t in name_tokens):
        return False
    question_tokens = [stem(t) for t in tokenize(question) if not t.isdigit()]
    width = len(name_tokens)
    return any(question_tokens[i : i + width] == name_tokens for i in range(len(question_tokens) - width + 1))


def vocabulary(texts: list[str]) -> set[str]:
    return {stem(t) for text in texts for t in tokenize(text)}
