import pytest

from cbct_reasoner.metrics import clinical_term_f1, corpus_bleu4, evaluate_pairs, meteor_lite

LONG_SENTENCE = "Impacted tooth 48 is close to the mandibular canal."


def test_identical_text_scores_one() -> None:
    assert corpus_bleu4([LONG_SENTENCE], [LONG_SENTENCE]) == pytest.approx(1.0)
    assert meteor_lite(LONG_SENTENCE, LONG_SENTENCE) == pytest.approx(0.9995)
    assert clinical_term_f1(LONG_SENTENCE, LONG_SENTENCE) == pytest.approx(1.0)


def test_short_identical_text_is_penalised_by_missing_4grams() -> None:
    """A three-token report has no 4-grams, so even a perfect match scores ~0.47.

    This is the grader's own behaviour (NLTK method1 smoothing), not a bug: it is
    why very short reports cannot win BLEU-4 regardless of correctness.
    """
    result = evaluate_pairs([("No lesion.", "No lesion."), ("Bone loss.", "Bone loss.")])

    assert result["num_cases"] == 2
    assert result["bleu_4"] == pytest.approx(0.05**0.25)
    assert result["meteor"] == pytest.approx(1.0 - 0.5 * (1 / 3) ** 3)


def test_evaluate_pairs_keeps_legacy_keys() -> None:
    result = evaluate_pairs([(LONG_SENTENCE, LONG_SENTENCE)])

    assert result["bleu_4_proxy"] == result["bleu_4"]
    assert result["meteor_lite"] == result["meteor"]
    assert result["captioning"] == pytest.approx((result["bleu_4"] + result["meteor"]) / 2)


def test_mismatched_pair_counts_fail() -> None:
    with pytest.raises(ValueError):
        corpus_bleu4(["one", "two"], ["one"])
