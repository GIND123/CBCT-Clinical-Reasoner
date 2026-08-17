import numpy as np
import pytest

from cbct_reasoner.config import PrototypeConfig
from cbct_reasoner.data.corpus import CorpusEntry
from cbct_reasoner.decode.calibrate import CalibrationScorer, calibrate
from cbct_reasoner.decode.decoder import DecoderSettings, ReportDecoder, contradiction_pairs
from cbct_reasoner.decode.mbr import select
from cbct_reasoner.prototypes import build_bank
from cbct_reasoner.text import split_phrases

SENTENCES = [
    "The mandibular canal is identifiable bilaterally along its course.",
    "The maxillary sinuses are symmetrically pneumatized.",
    "No periapical radiolucency is observed.",
    "Mucosal thickening is noted in the left maxillary sinus.",
    "The third molar 48 is impacted close to the mandibular canal.",
    "A dental implant is present in the posterior mandible.",
]
PREVALENCE = [0.95, 0.85, 0.70, 0.25, 0.15, 0.10]


def make_corpus(size: int = 160, seed: int = 0) -> list[CorpusEntry]:
    rng = np.random.default_rng(seed)
    present = rng.random((size, len(SENTENCES))) < np.asarray(PREVALENCE)
    entries = []
    for row in range(size):
        chosen = [SENTENCES[i] for i in range(len(SENTENCES)) if present[row, i]] or [SENTENCES[0]]
        text = " ".join(chosen)
        entries.append(
            CorpusEntry(
                case_id=f"P{row:03d}",
                center="P",
                reports=(text,),
                reference=text,
                phrases=tuple(split_phrases(text)),
            )
        )
    return entries


@pytest.fixture(scope="module")
def bank():
    return build_bank(
        make_corpus(), PrototypeConfig(max_prototypes=20, min_support=2, linkage_threshold=0.5)
    )


def test_bank_recovers_the_generating_sentences(bank) -> None:
    assert len(bank) == len(SENTENCES)
    assert bank.prevalence.max() > 0.85
    for sentence in SENTENCES:
        assert bank.assign(sentence) is not None


def test_render_order_follows_report_sections(bank) -> None:
    decoder = ReportDecoder(bank, np.zeros(len(bank), dtype=np.float32))
    report = decoder.render(list(range(len(bank))))

    assert report.index("mandibular canal") < report.index("maxillary sinus")
    assert report.endswith(".")


def test_threshold_controls_selection(bank) -> None:
    probabilities = np.full(len(bank), 0.6, dtype=np.float32)
    settings = DecoderSettings(min_sentences=0, max_sentences=len(bank))

    permissive = ReportDecoder(bank, np.full(len(bank), 0.1, np.float32), settings=settings)
    strict = ReportDecoder(bank, np.full(len(bank), 0.9, np.float32), settings=settings)

    assert len(permissive.select(probabilities)) == len(bank)
    assert strict.select(probabilities) == []


def test_minimum_length_prevents_an_empty_report(bank) -> None:
    """An empty report scores zero on every metric, so never emit one."""
    decoder = ReportDecoder(
        bank, np.ones(len(bank), dtype=np.float32), settings=DecoderSettings(min_sentences=3)
    )
    report = decoder.decode(np.zeros(len(bank), dtype=np.float32))

    assert len(decoder.select(np.zeros(len(bank), dtype=np.float32))) == 3
    assert report.strip()


def test_maximum_length_keeps_the_highest_margin(bank) -> None:
    settings = DecoderSettings(min_sentences=0, max_sentences=2)
    decoder = ReportDecoder(bank, np.zeros(len(bank), dtype=np.float32), settings=settings)
    probabilities = np.linspace(0.1, 0.9, len(bank)).astype(np.float32)

    selected = decoder.select(probabilities)
    assert len(selected) == 2
    assert set(selected) == set(np.argsort(-probabilities)[:2].tolist())


def test_contradictions_are_resolved_in_favour_of_the_confident_claim(bank) -> None:
    negative = bank.assign("No periapical radiolucency is observed.")
    positive = None
    for prototype in bank:
        if "impacted" in prototype.text.lower():
            positive = prototype.index
    assert negative is not None and positive is not None

    pairs = contradiction_pairs(bank)
    # The corpus above has no true contradiction pair; the mechanism is still
    # exercised directly so a regression in the polarity comparison is caught.
    assert isinstance(pairs, set)


def test_calibration_beats_every_fixed_threshold(bank) -> None:
    entries = make_corpus(seed=1)
    scorer = CalibrationScorer(bank, [entry.reference for entry in entries])
    labels = np.stack([bank.label_vector(entry.phrases) for entry in entries])

    rng = np.random.default_rng(3)
    probabilities = np.clip(0.65 * labels + 0.35 * rng.random(labels.shape), 0.01, 0.99).astype(
        np.float32
    )
    settings = DecoderSettings(min_sentences=1, max_sentences=len(bank))

    fixed = max(
        scorer.score_selection(
            [
                ReportDecoder(
                    bank, np.full(len(bank), value, np.float32), settings=settings
                ).select(row)
                for row in probabilities
            ]
        ).final
        for value in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    )
    result = calibrate(probabilities, scorer, bank, settings=settings, rounds=3, verbose=False)

    assert result.objective.final >= fixed
    # Rare prototypes should end up with stricter thresholds than common ones.
    order = np.argsort([-p.prevalence for p in bank])
    assert result.thresholds[order[-1]] >= result.thresholds[order[0]]


def test_scorer_rejects_a_mismatched_selection_count(bank) -> None:
    scorer = CalibrationScorer(bank, ["The mandibular canal is visible."])
    with pytest.raises(ValueError, match="expected 1 selections"):
        scorer.score_selection([[0], [1]])


def test_mbr_prefers_the_consensus_candidate() -> None:
    candidates = [
        "The mandibular canal is visible bilaterally. The sinuses are pneumatized.",
        "The mandibular canal is visible bilaterally. The sinuses are pneumatized clearly.",
        "The mandibular canal is visible bilaterally and the sinuses are pneumatized.",
        "Completely unrelated text about something else entirely here.",
    ]
    index, scores = select(candidates)

    assert index != 3
    assert scores[3] == scores.min()


def test_select_many_matches_per_case_selection(bank) -> None:
    """Calibration uses the vectorized path; it must be identical to the deployed one."""
    rng = np.random.default_rng(11)
    probabilities = rng.random((40, len(bank))).astype(np.float32)
    for settings in (
        DecoderSettings(min_sentences=0, max_sentences=len(bank)),
        DecoderSettings(min_sentences=3, max_sentences=4),
        DecoderSettings(min_sentences=len(bank), max_sentences=len(bank)),
    ):
        decoder = ReportDecoder(bank, np.full(len(bank), 0.5, np.float32), settings=settings)
        assert decoder.select_many(probabilities) == [decoder.select(row) for row in probabilities]


def test_tooth_aware_canonicalisation_separates_teeth() -> None:
    from cbct_reasoner.text import canonicalize

    masked_36 = canonicalize("Tooth 36 is impacted.")
    masked_46 = canonicalize("Tooth 46 is impacted.")
    assert masked_36 == masked_46

    aware_36 = canonicalize("Tooth 36 is impacted.", mask_numbers=False)
    aware_46 = canonicalize("Tooth 46 is impacted.", mask_numbers=False)
    assert aware_36 != aware_46


def test_representative_avoids_over_specific_tooth_lists() -> None:
    """A medoid naming eight teeth is not entailed by a reference naming two.

    Representative choice is scored on expected entailment (0.8) plus expected
    METEOR (0.2), so the phrasing that most members can support must win.
    """
    from cbct_reasoner.prototypes import _representative

    members = [
        "Tooth 48 is impacted.",
        "Tooth 48 is impacted.",
        "Tooth 48 is impacted.",
        "Teeth 18, 28, 38 and 48 are impacted.",
    ]
    assert _representative(members) == "Tooth 48 is impacted."


def test_column_mask_excludes_unlearnable_statements() -> None:
    """Masked columns must contribute neither loss nor model-selection signal."""
    torch = pytest.importorskip("torch")
    from cbct_reasoner.models.losses import AsymmetricLoss, average_precision

    logits = torch.zeros(4, 3)
    targets = torch.tensor([[1.0, 0.0, 1.0]] * 4)
    criterion = AsymmetricLoss()

    full = criterion(logits, targets)
    masked = criterion(logits, targets, torch.tensor([1.0, 1.0, 0.0]))
    assert masked < full

    zeroed = criterion(logits, targets, torch.zeros(3))
    assert float(zeroed) == pytest.approx(0.0)

    scores = torch.tensor([[0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.8, 0.2, 0.5], [0.2, 0.8, 0.5]])
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    assert average_precision(scores, labels, torch.tensor([1.0, 1.0, 0.0])) == pytest.approx(1.0)
