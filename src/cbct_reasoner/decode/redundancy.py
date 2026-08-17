"""Detect statements that assert the same clinical fact in different words.

The lexical surrogate rewards paraphrase, and the real metric does not.

Selection against the surrogate keeps picking sets like::

    Mandibular condyles are not included in the scan.
    Mandibular condyles: Excluded from the acquisition and not visible.

Under lexical matching those are two separate wins, because each one matches a
different phrasing somewhere in the reference corpus. Under RadFact, which asks
an LLM whether the reference entails the phrase, they are one fact asserted
twice:

* **Recall does not move.** Recall counts reference phrases entailed by the
  report. One reference phrase about the condyles is entailed whether the report
  says it once or three times.
* **Precision is neutral at best.** If the fact is true both phrasings are
  entailed, so the ratio is unchanged; if it is false, the report is charged
  twice for one mistake.

So paraphrase is a coin that pays only in the surrogate, and can only lose in the
metric that decides the shortlist. It is worse still in the double-blind clinical
comparison: a report that says the same thing twice in two registers is the most
obvious tell there is, and a reader does not need to know anything about the case
to notice it.

Two statements are treated as the same fact when they carry the same
``(concept, polarity)`` set, the same laterality, and the same tooth numbers -
the same signature RadFact's own entailment check keys on. Statements from which
no concept can be extracted are never grouped, because an empty signature is
absence of evidence rather than evidence of sameness.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from functools import lru_cache

from cbct_reasoner.ontology import extract_mentions, extract_teeth

#: A statement's clinical content: concepts with polarity, laterality, teeth.
AssertionKey = tuple[frozenset[tuple[str, str]], str, frozenset[int]]


def assertion_key(text: str) -> AssertionKey | None:
    """The fact a statement asserts, or None when nothing can be extracted."""
    mentions = extract_mentions(text)
    if not mentions:
        return None
    concepts = frozenset((mention.concept, mention.polarity) for mention in mentions)
    if not concepts:
        return None
    laterality = next(
        (m.laterality for m in mentions if m.laterality != "unspecified"), "unspecified"
    )
    teeth = frozenset(tooth for mention in mentions for tooth in mention.teeth)
    return concepts, laterality, teeth


def redundant_groups(texts: Sequence[str]) -> list[list[int]]:
    """Positions of statements sharing a fact, largest group first."""
    groups: dict[AssertionKey, list[int]] = {}
    for position, text in enumerate(texts):
        key = assertion_key(text)
        if key is None:
            continue
        groups.setdefault(key, []).append(position)
    duplicates = [group for group in groups.values() if len(group) > 1]
    return sorted(duplicates, key=lambda g: (-len(g), g[0]))


#: Words that describe the same property and cannot both be true of one
#: structure. Selection against the Final Score happily takes both, because each
#: earns credit on the subset of cases whose reference agrees with it - the
#: metric scores statements one at a time and never reads the report as a whole.
EXCLUSIVE_ATTRIBUTES: tuple[frozenset[str], ...] = (
    # Course of the mandibular canal relative to the cortical plates.
    frozenset({"lingual", "buccal", "vestibular", "palatal"}),
    # How much of a structure the volume caught.
    frozenset({"completely", "partially", "minimally"}),
    frozenset({"regular", "irregular"}),
)

#: Concepts that assert a tooth is not there.
_ABSENCE_CONCEPTS = frozenset({"missing_tooth", "edentulous"})

#: Edentulism of the whole arch, as opposed to a gap at named positions.
_COMPLETE_EDENTULISM_RE = re.compile(
    r"\b(?:complete|total)\s+edentul\w*|\bedentul\w*\s+(?:arch|mandible|maxilla)",
    re.IGNORECASE,
)


def _words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z]+", text.lower()))


@lru_cache(maxsize=200_000)
def conflicts(first: str, second: str) -> str | None:
    """Why these two statements cannot both be true, or None if they can.

    A report that contradicts itself is guaranteed to be wrong about something,
    so this costs RadFact precision as well as credibility. It costs credibility
    far more: a reader does not need to know anything about the case to see that
    "maxilla: not included" and "the maxilla is partially included" cannot both
    hold, and that teeth listed as absent cannot also carry fillings.
    """
    first_mentions = extract_mentions(first)
    second_mentions = extract_mentions(second)
    first_polarity = {m.concept: m.polarity for m in first_mentions}
    second_polarity = {m.concept: m.polarity for m in second_mentions}

    for concept, polarity in first_polarity.items():
        other = second_polarity.get(concept)
        if other is None:
            continue
        if {polarity, other} == {"present", "absent"}:
            return f"{concept}: {polarity} vs {other}"

    shared = set(first_polarity) & set(second_polarity)
    if shared:
        first_words, second_words = _words(first), _words(second)
        for group in EXCLUSIVE_ATTRIBUTES:
            left, right = first_words & group, second_words & group
            if left and right and left != right:
                return f"{sorted(shared)[0]}: {sorted(left)[0]} vs {sorted(right)[0]}"

    # Complete edentulism carries no tooth numbers to intersect, so it needs its
    # own rule: it rules out every specific dental finding, not just overlapping
    # ones. A mandible cannot be toothless and also carry occlusal composites.
    for edentulous, other in ((first, second), (second, first)):
        if not _COMPLETE_EDENTULISM_RE.search(edentulous):
            continue
        # Tooth numbers are read from the text rather than from the mentions:
        # only concepts flagged tooth_specific carry them, and "restoration" is
        # not one, so a sentence about composites on teeth 36 and 37 looked
        # toothless to this check.
        described = set(extract_teeth(other))
        if described and any(
            m.polarity == "present" and m.concept not in _ABSENCE_CONCEPTS
            for m in extract_mentions(other)
        ):
            return f"complete edentulism vs findings on teeth {sorted(described)}"

    # A tooth cannot be missing in one sentence and restored in the next.
    for absent, present in ((first_mentions, second_mentions), (second_mentions, first_mentions)):
        gone = {
            tooth
            for m in absent
            if m.concept in _ABSENCE_CONCEPTS and m.polarity == "present"
            for tooth in m.teeth
        }
        if not gone:
            continue
        claimed = {
            tooth
            for m in present
            if m.polarity == "present" and m.concept not in _ABSENCE_CONCEPTS
            for tooth in m.teeth
        }
        overlap = gone & claimed
        if overlap:
            return f"teeth {sorted(overlap)} are both absent and described"
    return None


#: Statements that begin with a reference to something a previous sentence
#: named. Lifted out of one patient's report, they refer to nothing.
_DANGLING_RE = re.compile(
    r"^\s*(?:it\b|they\b|the\s+(?:fifth|sixth|seventh|eighth)\s+tooth\b|"
    r"a\s+prosthetic\s+crown\s+is\s+present)",
    re.IGNORECASE,
)


#: Back-references to a tooth named in a sentence that is no longer there.
_LATTER_RE = re.compile(
    r"\b(?:this|the)\s+(?:latter|former|same|aforementioned|said)\b"
    r"|\bof\s+(?:this|that)\s+(?:tooth|element)\b",
    re.IGNORECASE,
)


def is_well_formed(text: str) -> bool:
    """Can this sentence stand alone in a report?

    Rejects the wreckage of splitting one patient's narrative into sentences:
    clauses opening with a reference to something the previous sentence named
    ("it has undergone endodontic treatment", "distal to the crown of this
    latter tooth"), and fragments carrying a bracket whose partner was left
    behind ("Endodontic material beyond the apex of the distal root)").

    These cost nothing to drop - they are unreadable wherever they land - so the
    check is always on, unlike the tooth-number filter below.
    """
    stripped = text.strip()
    if not stripped or _DANGLING_RE.match(stripped):
        return False
    if _LATTER_RE.search(stripped):
        return False
    return stripped.count("(") == stripped.count(")")


def is_generic(text: str) -> bool:
    """Can this statement be asserted without knowing which patient it is?

    Statements naming specific teeth cannot. Across this corpus they hold for a
    mean of 1.6% of cases, against 3.5% for statements that name none - so in a
    report served to every patient, a tooth number is wrong about 98% of the
    time. It still earns surrogate credit through token overlap, which is why
    selection reaches for it, and it is what turns the report into a patchwork
    that says teeth 38 and 48 are absent, impacted, and erupted at once.

    A radiologist who cannot resolve the specifics writes generically instead,
    and a generic statement that is usually true beats a specific one that is
    usually false on precision as well as on reading.
    """
    from cbct_reasoner.ontology import BARE_FDI_RE, TOOTH_NUMBER_RE

    if _DANGLING_RE.match(text):
        return False
    return not (TOOTH_NUMBER_RE.search(text) or BARE_FDI_RE.search(text))


def consistent_subset(texts: Sequence[str]) -> list[int]:
    """Positions of statements that neither contradict nor repeat an earlier one."""
    kept: list[int] = []
    keys: set[AssertionKey] = set()
    for position, text in enumerate(texts):
        key = assertion_key(text)
        if key is not None and key in keys:
            continue
        if any(conflicts(text, texts[earlier]) for earlier in kept):
            continue
        kept.append(position)
        if key is not None:
            keys.add(key)
    return kept


def deduplicate(
    texts: Sequence[str],
    *,
    prefer: Iterable[float] | None = None,
) -> list[int]:
    """Keep one statement per fact.

    ``prefer`` scores each statement; the highest scorer in a group survives.
    Without it the first statement in render order wins, which keeps the report
    reading in the order a radiologist writes it.
    """
    scores = list(prefer) if prefer is not None else [0.0] * len(texts)
    if len(scores) != len(texts):
        raise ValueError(f"prefer has {len(scores)} scores for {len(texts)} statements")

    dropped: set[int] = set()
    for group in redundant_groups(texts):
        keeper = max(group, key=lambda i: (scores[i], -i))
        dropped.update(position for position in group if position != keeper)
    return [position for position in range(len(texts)) if position not in dropped]
