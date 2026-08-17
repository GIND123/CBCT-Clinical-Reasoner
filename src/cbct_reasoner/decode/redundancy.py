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

from collections.abc import Iterable, Sequence

from cbct_reasoner.ontology import extract_mentions

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
