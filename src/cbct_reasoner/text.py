"""Clinical text normalization and phrase segmentation.

The official metric stack works at two granularities: whole-report token overlap
(BLEU-4 / METEOR) and per-phrase logical entailment (RadFact). ``split_phrases``
mirrors the RadFact prompt contract - one verifiable finding per phrase, with
recommendations and scan-quality remarks separated out - so local calibration
optimizes the same units the grader scores.
"""

from __future__ import annotations

import re
import unicodedata

# Abbreviations whose trailing period never ends a sentence in these reports.
# Units are deliberately absent: "residual bone height is 6 mm." ends a sentence
# far more often than "mm." continues one, and guarding it merges two findings
# into a single phrase, which RadFact then scores as one all-or-nothing unit.
_ABBREVIATIONS = (
    "approx",
    "cf",
    "dr",
    "e.g",
    "et al",
    "fig",
    "i.e",
    "n",
    "no",
    "nr",
    "prof",
    "sig",
    "vs",
)
_ABBREVIATION_RE = re.compile(
    r"\b(" + "|".join(re.escape(item) for item in _ABBREVIATIONS) + r")\.", re.IGNORECASE
)
_DECIMAL_RE = re.compile(r"(?<=\d)[.,](?=\d)")
# A period only ends a sentence when the next token starts like one. These
# reports are translated from Italian dictation and consistently capitalize.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_SEMICOLON_RE = re.compile(r"\s*;\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•–—]|\(?\d{1,2}[.)])\s+")

_PLACEHOLDER_PERIOD = "\x01"
_PLACEHOLDER_DECIMAL = "\x02"

#: Phrases RadFact's parser drops (recommendations, speculation, scan quality).
#: Keeping them out of the candidate set protects logical precision.
NON_VERIFIABLE_RE = re.compile(
    r"\b(?:recommend\w*|advis\w*|suggest\s+(?:clinical|further)|follow[-\s]?up|"
    r"correlat\w*\s+clinical\w*|referral|should\s+be\s+(?:considered|evaluated)|"
    r"image\s+quality|scan\s+quality|motion\s+artifact\w*\s+limit\w*|"
    r"examination\s+(?:is\s+)?(?:technically\s+)?(?:adequate|suboptimal))\b",
    re.IGNORECASE,
)

#: Conjunctions that join two independently verifiable findings in one sentence.
#: "with" is deliberately excluded: it almost always introduces a dependent
#: modifier ("...the mandibular canals, with a predominantly lingual course"),
#: and splitting there strands a fragment that no reference can entail.
_SPLIT_CONJUNCTIONS = re.compile(
    r",\s+(?:and|while|whereas)\s+(?=(?:the|a|an|there|no|both|bilateral|left|right)\b)",
    re.IGNORECASE,
)

#: A clause needs a finite verb to stand alone as a checkable finding.
_CLAUSE_VERB_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|shows?|showed|appears?|presents?|"
    r"demonstrates?|reveals?|extends?|involves?|measures?|exhibits?|"
    r"noted|observed|identified|seen|visible|detected|documented|evident|"
    r"absent|present|confirmed)\b",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Collapse whitespace and unify Unicode punctuation without altering meaning."""
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFKC", text.replace("\x00", " "))
    cleaned = cleaned.translate(
        str.maketrans(
            {
                "‘": "'",
                "’": "'",
                "“": '"',
                "”": '"',
                "–": "-",
                "—": "-",
                " ": " ",
            }
        )
    )
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def split_sentences(text: str) -> list[str]:
    """Split a narrative report into sentences, protecting decimals and abbreviations."""
    normalized = normalize_text(text)
    if not normalized:
        return []
    guarded = _ABBREVIATION_RE.sub(lambda m: m.group(1) + _PLACEHOLDER_PERIOD, normalized)
    guarded = _DECIMAL_RE.sub(_PLACEHOLDER_DECIMAL, guarded)
    parts = [
        piece for clause in _SEMICOLON_RE.split(guarded) for piece in _SENTENCE_END_RE.split(clause)
    ]
    sentences = []
    for part in parts:
        restored = part.replace(_PLACEHOLDER_PERIOD, ".").replace(_PLACEHOLDER_DECIMAL, ".")
        restored = _BULLET_RE.sub("", restored).strip()
        if restored:
            sentences.append(restored)
    return sentences


def split_phrases(text: str, *, split_conjunctions: bool = True) -> list[str]:
    """Segment a report into RadFact-style verifiable phrases.

    Multi-finding sentences are broken apart because RadFact scores each finding
    independently: one unsupported clause otherwise drags an otherwise-correct
    sentence to ``not_entailment``.
    """
    phrases: list[str] = []
    for sentence in split_sentences(text):
        candidates = [sentence]
        if split_conjunctions:
            parts = [part.strip() for part in _SPLIT_CONJUNCTIONS.split(sentence)]
            # Only accept the split when every piece can stand on its own; a
            # stranded modifier is unverifiable in both scoring directions.
            if len(parts) > 1 and all(_CLAUSE_VERB_RE.search(part) for part in parts):
                candidates = parts
        for candidate in candidates:
            cleaned = candidate.strip(" ;,")
            if len(cleaned.split()) >= 2:
                phrases.append(ensure_terminal_period(cleaned))
    return phrases


def is_verifiable(phrase: str) -> bool:
    """True when RadFact's parser would retain this phrase as a CBCT finding."""
    return not NON_VERIFIABLE_RE.search(phrase)


def ensure_terminal_period(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped if stripped[-1] in ".!?" else stripped + "."


def canonicalize(phrase: str, *, mask_numbers: bool = True) -> str:
    """Aggressive lowercase form used to deduplicate near-identical sentences.

    With ``mask_numbers`` every number becomes ``#``, which groups phrasings
    ("tooth 36 is impacted" and "tooth 46 is impacted" become one form). Set it
    False to keep statements about different teeth apart - a distinction that
    matters because RadFact scores a wrong tooth number as not-entailed.
    """
    lowered = normalize_text(phrase).casefold()
    if mask_numbers:
        lowered = re.sub(r"\d+(?:\.\d+)?", "#", lowered)
    else:
        lowered = re.sub(r"(?<!\d)\d{1,3}(?:\.\d+)?", lambda m: f" n{m.group()} ", lowered)
    lowered = re.sub(r"[^a-z0-9#\s]", " ", lowered)
    return _WHITESPACE_RE.sub(" ", lowered).strip()


def capitalize_first(text: str) -> str:
    """Uppercase the leading letter, leaving acronyms and numbers untouched.

    Splitting a compound sentence on ", and ..." leaves the second clause
    lowercase; as a standalone report sentence it needs a capital.
    """
    stripped = text.lstrip()
    for position, character in enumerate(stripped):
        if character.isalpha():
            return stripped[:position] + character.upper() + stripped[position + 1 :]
        if character.isdigit():
            break
    return stripped


def join_report(sentences: list[str]) -> str:
    """Render selected sentences as one flowing narrative paragraph."""
    rendered = [
        capitalize_first(ensure_terminal_period(normalize_text(item))) for item in sentences
    ]
    return " ".join(item for item in rendered if item)


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))
