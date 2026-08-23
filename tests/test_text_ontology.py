from cbct_reasoner.ontology import (
    concept_profile,
    detect_laterality,
    extract_measurements,
    extract_mentions,
    extract_teeth,
    section_of,
)
from cbct_reasoner.text import (
    canonicalize,
    capitalize_first,
    is_verifiable,
    join_report,
    split_phrases,
    split_sentences,
)


def test_measurement_does_not_end_a_sentence() -> None:
    """Regression: guarding "mm." as an abbreviation merged two findings into one.

    RadFact scores a phrase all-or-nothing, so a merged phrase turns one correct
    finding plus one wrong finding into zero credit instead of a half.
    """
    text = (
        "Residual bone height above the sinus floor measures approximately 6 mm. "
        "The nasopalatine canal is of normal calibre."
    )
    assert split_sentences(text) == [
        "Residual bone height above the sinus floor measures approximately 6 mm.",
        "The nasopalatine canal is of normal calibre.",
    ]


def test_decimals_and_italian_tooth_notation_survive() -> None:
    text = "Tooth n. 48 is impacted. Mucosal thickening of 3.5 mm is present."
    sentences = split_sentences(text)

    assert sentences[0] == "Tooth n. 48 is impacted."
    assert "3.5 mm" in sentences[1]


def test_semicolons_and_conjunctions_split_into_separate_findings() -> None:
    text = "The left sinus is clear; the right sinus shows thickening."
    assert len(split_phrases(text)) == 2

    joined = "The mandibular canal is visible, and the mental foramen is identified."
    assert len(split_phrases(joined)) == 2


def test_recommendations_are_not_verifiable() -> None:
    assert not is_verifiable("Clinical correlation is recommended.")
    assert not is_verifiable("Follow-up imaging should be considered.")
    assert not is_verifiable("The image quality is suboptimal.")
    assert is_verifiable("The maxillary sinuses are pneumatized.")


def test_canonicalize_collapses_tooth_numbers() -> None:
    assert canonicalize("Tooth 36 is impacted.") == canonicalize("Tooth 46 is impacted.")


def test_capitalisation_and_joining() -> None:
    assert capitalize_first("the sinus is clear.") == "The sinus is clear."
    assert capitalize_first("48 is impacted.") == "48 is impacted."
    assert join_report(["the canal is visible", "no lesion"]) == (
        "The canal is visible. No lesion."
    )


def test_tooth_extraction_prefers_explicit_notation() -> None:
    assert extract_teeth("Impacted tooth 48 near the canal.") == (48,)
    assert extract_teeth("Teeth 16, 17 and 26 are restored.") == (16, 17, 26)
    assert extract_teeth("No teeth are numbered here.") == ()


def test_measurements_normalise_to_millimetres() -> None:
    assert extract_measurements("Residual height of 1.2 cm.") == (12.0,)
    assert extract_measurements("Thickening of 4,5 mm.") == (4.5,)


def test_laterality_detection() -> None:
    assert detect_laterality("The left maxillary sinus is opacified.") == "left"
    assert detect_laterality("Bilateral mucosal thickening.") == "bilateral"
    assert detect_laterality("Left and right condyles are flattened.") == "bilateral"
    assert detect_laterality("The canal is visible.") == "unspecified"


def test_negation_and_uncertainty_are_distinct_polarities() -> None:
    absent = extract_mentions("No periapical radiolucency is observed.")
    uncertain = extract_mentions("A possible periapical lesion is noted.")
    present = extract_mentions("Periapical radiolucency at the apex of tooth 46.")

    assert any(m.polarity == "absent" for m in absent)
    assert any(m.polarity == "uncertain" for m in uncertain)
    assert any(m.polarity == "present" for m in present)


def test_concept_profile_prefers_the_strongest_polarity() -> None:
    profile = concept_profile(
        ["No periapical lesion is seen.", "Periapical radiolucency at tooth 46."]
    )
    assert profile["periapical_lesion"] == "present"


def test_sections_order_the_report() -> None:
    assert section_of("The maxillary sinus shows mucosal thickening.") == "sinus"
    assert section_of("The mandibular canal is visible.") == "mandible"
    assert section_of("Cone-beam CT of the maxillofacial region.") == "technique"
