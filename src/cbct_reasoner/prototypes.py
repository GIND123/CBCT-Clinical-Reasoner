"""Sentence-prototype bank: the bridge from free text to a supervised label space.

ToothFairy4 reports are narrative but highly formulaic - the same findings recur
in a small number of phrasings. Clustering the training phrases yields a compact
vocabulary of reportable statements, which converts an intractable
image-to-free-text problem into multi-label classification plus deterministic
rendering. That matters for the score in three ways:

* **RadFact precision** - every emitted sentence is a phrasing a clinician
  actually wrote about this dataset, so entailment failures come from picking the
  wrong finding, never from fabricated language.
* **RadFact recall** - the bank's prevalence statistics tell you which findings
  references almost always mention, which are exactly the ones worth emitting.
* **METEOR** - each cluster's representative is chosen to maximize expected
  METEOR against the other members, so a correct prediction also lands the
  highest-scoring surface form.

Cluster assignment at inference uses a portable token-overlap matcher rather than
the training-time TF-IDF model, so the deployed bundle needs only numpy.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cbct_reasoner.config import PrototypeConfig
from cbct_reasoner.data.corpus import CorpusEntry
from cbct_reasoner.metrics.official import meteor_score_tokenized, tokenize
from cbct_reasoner.ontology import SECTION_INDEX, extract_mentions, section_of
from cbct_reasoner.text import canonicalize, is_verifiable, split_phrases

BANK_VERSION = 2

_STOPWORDS = frozenset(
    """
    a an and are as at be been both by for from has have in is it its of on or
    that the there these this to was were with within
    """.split()
)

#: Minimum token-overlap similarity for a phrase to be assigned to a prototype.
ASSIGN_THRESHOLD = 0.55


@dataclass(frozen=True, slots=True)
class Prototype:
    """One reportable statement type."""

    index: int
    text: str
    canonical: str
    section: str
    concepts: tuple[str, ...]
    tokens: frozenset[str]
    support: int
    prevalence: float
    variants: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "text": self.text,
            "canonical": self.canonical,
            "section": self.section,
            "concepts": list(self.concepts),
            "tokens": sorted(self.tokens),
            "support": self.support,
            "prevalence": self.prevalence,
            "variants": list(self.variants),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Prototype:
        return cls(
            index=int(payload["index"]),  # type: ignore[arg-type]
            text=str(payload["text"]),
            canonical=str(payload["canonical"]),
            section=str(payload["section"]),
            concepts=tuple(str(item) for item in payload["concepts"]),  # type: ignore[union-attr]
            tokens=frozenset(str(item) for item in payload["tokens"]),  # type: ignore[union-attr]
            support=int(payload["support"]),  # type: ignore[arg-type]
            prevalence=float(payload["prevalence"]),  # type: ignore[arg-type]
            variants=tuple(str(item) for item in payload["variants"]),  # type: ignore[union-attr]
        )

    @property
    def sort_key(self) -> tuple[int, float]:
        """Report position: section order first, then commonest statement first."""
        return (SECTION_INDEX.get(self.section, SECTION_INDEX["other"]), -self.prevalence)


@dataclass(frozen=True, slots=True)
class PrototypeBank:
    """The full label space plus the statistics the decoder calibrates against."""

    prototypes: tuple[Prototype, ...]
    num_cases: int

    def __len__(self) -> int:
        return len(self.prototypes)

    def __iter__(self):
        return iter(self.prototypes)

    def __getitem__(self, index: int) -> Prototype:
        return self.prototypes[index]

    @property
    def prevalence(self) -> np.ndarray:
        return np.asarray([p.prevalence for p in self.prototypes], dtype=np.float32)

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(p.text for p in self.prototypes)

    @property
    def render_order(self) -> tuple[int, ...]:
        return tuple(p.index for p in sorted(self.prototypes, key=lambda item: item.sort_key))

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bank_version": BANK_VERSION,
            "num_cases": self.num_cases,
            "prototypes": [prototype.to_dict() for prototype in self.prototypes],
        }
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> PrototypeBank:
        location = Path(path)
        if not location.is_file():
            raise FileNotFoundError(
                f"Prototype bank not found at {location}. Run `cbct-reasoner prototypes` first."
            )
        payload = json.loads(location.read_text(encoding="utf-8-sig"))
        if int(payload.get("bank_version", 0)) != BANK_VERSION:
            raise ValueError(
                f"Prototype bank version {payload.get('bank_version')} != "
                f"{BANK_VERSION}; rebuild it."
            )
        return cls(
            prototypes=tuple(Prototype.from_dict(item) for item in payload["prototypes"]),
            num_cases=int(payload["num_cases"]),
        )

    def assign(self, phrase: str, *, threshold: float = ASSIGN_THRESHOLD) -> int | None:
        """Map an arbitrary phrase onto its closest prototype, or ``None``."""
        best_index, best_score = None, threshold
        signature = _phrase_tokens(phrase)
        concepts = frozenset(mention.concept for mention in extract_mentions(phrase))
        if not signature:
            return None
        for prototype in self.prototypes:
            score = _similarity(signature, concepts, prototype)
            if score >= best_score:
                best_index, best_score = prototype.index, score
        return best_index

    def label_vector(
        self, phrases: Iterable[str], *, threshold: float = ASSIGN_THRESHOLD
    ) -> np.ndarray:
        vector = np.zeros(len(self.prototypes), dtype=np.float32)
        for phrase in phrases:
            index = self.assign(phrase, threshold=threshold)
            if index is not None:
                vector[index] = 1.0
        return vector


def _phrase_tokens(phrase: str) -> frozenset[str]:
    return frozenset(
        token
        for token in tokenize(canonicalize(phrase))
        if token.isalnum() and token not in _STOPWORDS
    )


def _similarity(tokens: frozenset[str], concepts: frozenset[str], prototype: Prototype) -> float:
    """Concept-gated Jaccard.

    Two sentences about different anatomy must never merge just because they
    share filler words, so a concept mismatch caps the similarity below the
    assignment threshold.
    """
    if not tokens or not prototype.tokens:
        return 0.0
    overlap = len(tokens & prototype.tokens)
    if overlap == 0:
        return 0.0
    jaccard = overlap / len(tokens | prototype.tokens)
    prototype_concepts = frozenset(prototype.concepts)
    if prototype_concepts and concepts and not (prototype_concepts & concepts):
        return jaccard * 0.4
    if prototype_concepts and concepts:
        agreement = len(prototype_concepts & concepts) / len(prototype_concepts | concepts)
        return 0.7 * jaccard + 0.3 * agreement
    return jaccard


def collect_phrases(
    entries: Iterable[CorpusEntry], *, max_words: int
) -> tuple[list[str], dict[str, set[str]]]:
    """Gather every verifiable phrase and the cases each canonical form occurs in."""
    representatives: dict[str, Counter[str]] = defaultdict(Counter)
    cases_by_canonical: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        seen: set[str] = set()
        for report in entry.reports:
            for phrase in split_phrases(report):
                if not is_verifiable(phrase):
                    continue
                if not 2 <= len(phrase.split()) <= max_words:
                    continue
                canonical = canonicalize(phrase)
                if not canonical:
                    continue
                representatives[canonical][phrase] += 1
                seen.add(canonical)
        for canonical in seen:
            cases_by_canonical[canonical].add(entry.case_id)

    unique = [counter.most_common(1)[0][0] for counter in representatives.values()]
    return unique, dict(cases_by_canonical)


def build_bank(entries: Sequence[CorpusEntry], config: PrototypeConfig) -> PrototypeBank:
    """Cluster the training phrases into a prototype bank."""
    entries = list(entries)
    if not entries:
        raise ValueError("Cannot build a prototype bank from an empty corpus")

    phrases, cases_by_canonical = collect_phrases(entries, max_words=config.max_sentence_words)
    if not phrases:
        raise ValueError("No verifiable phrases were extracted from the corpus")

    canonical_of = {phrase: canonicalize(phrase) for phrase in phrases}
    labels = _cluster(phrases, config)

    clusters: dict[int, list[str]] = defaultdict(list)
    for phrase, label in zip(phrases, labels, strict=True):
        clusters[int(label)].append(phrase)

    scored: list[tuple[int, str, list[str], set[str]]] = []
    for members in clusters.values():
        supporting_cases: set[str] = set()
        for member in members:
            supporting_cases |= cases_by_canonical.get(canonical_of[member], set())
        if len(supporting_cases) < config.min_support:
            continue
        scored.append((len(supporting_cases), _representative(members), members, supporting_cases))

    if not scored:
        # A tiny corpus can leave every cluster below min_support; fall back to
        # the most widely observed statements rather than returning nothing.
        scored = [
            (
                len(cases_by_canonical.get(canonical_of[members[0]], set())) or 1,
                _representative(members),
                members,
                set(),
            )
            for members in clusters.values()
        ]

    scored.sort(key=lambda item: (-item[0], item[1]))
    scored = scored[: config.max_prototypes]

    prototypes: list[Prototype] = []
    for index, (support, text, members, _) in enumerate(scored):
        concepts = tuple(sorted({mention.concept for mention in extract_mentions(text)}))
        prototypes.append(
            Prototype(
                index=index,
                text=text,
                canonical=canonicalize(text),
                section=section_of(text),
                concepts=concepts,
                tokens=_phrase_tokens(text),
                support=int(support),
                prevalence=float(support) / len(entries),
                variants=tuple(sorted(set(members))[:12]),
            )
        )
    return PrototypeBank(prototypes=tuple(prototypes), num_cases=len(entries))


def _cluster(phrases: Sequence[str], config: PrototypeConfig) -> np.ndarray:
    """Agglomerative clustering over TF-IDF cosine distance."""
    if len(phrases) == 1:
        return np.zeros(1, dtype=np.int64)
    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as error:  # pragma: no cover - environment specific
        raise RuntimeError(
            "scikit-learn is required to build the prototype bank; install '.[train]'"
        ) from error

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        preprocessor=canonicalize,
        token_pattern=r"[a-z#]+",
    )
    matrix = vectorizer.fit_transform(phrases)
    if matrix.shape[1] == 0:
        return np.zeros(len(phrases), dtype=np.int64)

    dense = np.asarray(matrix.todense(), dtype=np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    dense = dense / np.clip(norms, 1e-8, None)
    n_clusters = None if len(phrases) > 2 else 1
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
        distance_threshold=config.linkage_threshold if n_clusters is None else None,
    )
    return np.asarray(model.fit_predict(dense), dtype=np.int64)


def _representative(members: Sequence[str]) -> str:
    """Pick the cluster member with the highest mean METEOR against the others.

    This is the surface form the grader will reward most often when the finding
    is predicted correctly - a metric-aware choice, not simply the most frequent
    or the shortest string.
    """
    unique = sorted(set(members))
    if len(unique) == 1:
        return unique[0]
    tokenized = [tokenize(member) for member in unique]
    best_index, best_score = 0, -1.0
    for index, candidate in enumerate(tokenized):
        others = [other for position, other in enumerate(tokenized) if position != index]
        score = sum(meteor_score_tokenized(candidate, other) for other in others) / len(others)
        if score > best_score or (
            math.isclose(score, best_score) and len(candidate) > len(tokenized[best_index])
        ):
            best_index, best_score = index, score
    return unique[best_index]


def build_labels(
    entries: Sequence[CorpusEntry], bank: PrototypeBank, *, use_all_reports: bool = True
) -> np.ndarray:
    """Multi-hot targets, one row per case.

    ``use_all_reports`` unions the findings across every annotator for a case: a
    finding one clinician recorded is present in the scan even when the selected
    reference omits it, so the imaging model should be trained to see it.
    """
    matrix = np.zeros((len(entries), len(bank)), dtype=np.float32)
    for row, entry in enumerate(entries):
        phrases = entry.all_phrases if use_all_reports else entry.phrases
        matrix[row] = bank.label_vector(phrases)
    return matrix


def save_labels(path: str | Path, case_ids: Sequence[str], matrix: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, case_ids=np.asarray(list(case_ids), dtype=np.str_), labels=matrix.astype(np.float32)
    )
    return output


def load_labels(path: str | Path) -> tuple[list[str], np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return [str(v) for v in archive["case_ids"].tolist()], archive["labels"].astype(np.float32)
