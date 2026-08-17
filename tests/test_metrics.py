import pytest

from cbct_reasoner.metrics import clinical_term_f1, corpus_bleu4, evaluate_pairs, meteor_lite


def test_identical_text_scores_one() -> None:
    text = "Impacted tooth 48 is close to the mandibular canal."

    assert corpus_bleu4([text], [text]) == pytest.approx(1.0)
    assert meteor_lite(text, text) == pytest.approx(0.9995)
    assert clinical_term_f1(text, text) == pytest.approx(1.0)


def test_evaluate_pairs_reports_case_count() -> None:
    result = evaluate_pairs([("No lesion.", "No lesion."), ("Bone loss.", "Bone loss.")])

    assert result["num_cases"] == 2
    assert result["bleu_4_proxy"] == pytest.approx(1.0)


def test_mismatched_pair_counts_fail() -> None:
    with pytest.raises(ValueError):
        corpus_bleu4(["one", "two"], ["one"])
