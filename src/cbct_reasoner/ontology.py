"""Maxillofacial CBCT finding ontology.

Concepts are the interpretable layer between the imaging encoder and the report
renderer. Each concept carries the regular expressions that recognise it in a
reference report, so a free-text corpus can be converted into supervised
multi-label targets without manual annotation.

Nothing here is a diagnostic rule set. It is a lexical mapping used for training
supervision, evaluation slicing, and report structure.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# Report sections in the order maxillofacial surgeons dictate them. Rendering in
# this order maximises contiguous n-gram overlap with references, which the
# METEOR chunk penalty and BLEU-4 both reward.
SECTIONS: tuple[str, ...] = (
    "technique",
    "dentition",
    "periapical",
    "periodontal",
    "mandible",
    "maxilla",
    "sinus",
    "tmj",
    "other",
    "impression",
)

#: FDI permanent quadrants/positions. Used for tooth-level heads and validation.
FDI_PERMANENT: tuple[int, ...] = tuple(
    quadrant * 10 + position for quadrant in (1, 2, 3, 4) for position in range(1, 9)
)
FDI_PRIMARY: tuple[int, ...] = tuple(
    quadrant * 10 + position for quadrant in (5, 6, 7, 8) for position in range(1, 6)
)
THIRD_MOLARS: tuple[int, ...] = (18, 28, 38, 48)

TOOTH_NUMBER_RE = re.compile(
    r"\b(?:tooth|teeth|element|elements|nn?\.?)\s*#?\s*((?:\d{2}[,\s/and]*)+)"
)
BARE_FDI_RE = re.compile(r"\b([1-4][1-8]|[5-8][1-5])\b")
MEASUREMENT_RE = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(mm|cm|millimet\w*|centimet\w*)\b", re.IGNORECASE
)
LEFT_RE = re.compile(r"\b(?:left|sinistr\w*|lt\.?)\b", re.IGNORECASE)
RIGHT_RE = re.compile(r"\b(?:right|dextr\w*|rt\.?)\b", re.IGNORECASE)
BILATERAL_RE = re.compile(r"\b(?:bilateral\w*|both\s+sides?|on\s+both)\b", re.IGNORECASE)
NEGATION_RE = re.compile(
    r"\b(?:no|not|without|absence\s+of|absent|free\s+of|negative\s+for|unremarkable|"
    r"neither|nor|denies|rules?\s+out|excluded)\b",
    re.IGNORECASE,
)
UNCERTAINTY_RE = re.compile(
    r"\b(?:possible|possibly|probable|probably|suspected|suspicious|apparent\w*|"
    r"cannot\s+be\s+excluded|may\s+be|likely|questionable|equivocal|borderline)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Concept:
    """One reportable observation with its lexical triggers."""

    key: str
    section: str
    label: str
    patterns: tuple[re.Pattern[str], ...]
    tooth_specific: bool = False
    lateralized: bool = False
    measurable: bool = False

    def matches(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self.patterns)


def _c(
    key: str,
    section: str,
    label: str,
    *patterns: str,
    tooth_specific: bool = False,
    lateralized: bool = False,
    measurable: bool = False,
) -> Concept:
    return Concept(
        key=key,
        section=section,
        label=label,
        patterns=tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns),
        tooth_specific=tooth_specific,
        lateralized=lateralized,
        measurable=measurable,
    )


CONCEPTS: tuple[Concept, ...] = (
    # --- technique -------------------------------------------------------
    _c(
        "cbct_study",
        "technique",
        "CBCT examination of the maxillofacial region",
        r"\bcone[-\s]?beam\b",
        r"\bCBCT\b",
        r"\bCT\s+scan\b",
    ),
    _c("field_of_view", "technique", "field of view", r"\bfield\s+of\s+view\b", r"\bFOV\b"),
    # --- dentition -------------------------------------------------------
    _c(
        "edentulous",
        "dentition",
        "edentulous region",
        r"\bedentul\w*\b",
        r"\bmissing\s+teeth\b",
        r"\btoothless\b",
        tooth_specific=True,
    ),
    _c(
        "missing_tooth",
        "dentition",
        "missing tooth",
        r"\b(?:absence|absent|missing|agenesis)\s+of\s+(?:the\s+)?(?:tooth|teeth|element)",
        r"\btooth\s+\d{2}\s+is\s+(?:missing|absent)\b",
        tooth_specific=True,
    ),
    _c(
        "impacted_tooth",
        "dentition",
        "impacted tooth",
        r"\bimpact(?:ed|ion)\b",
        r"\bunerupted\b",
        r"\bretained\s+(?:tooth|third\s+molar)\b",
        r"\binclu(?:so|ded)\b",
        tooth_specific=True,
    ),
    _c(
        "third_molar",
        "dentition",
        "third molar",
        r"\bthird\s+molars?\b",
        r"\bwisdom\s+(?:tooth|teeth)\b",
        r"\b(?:18|28|38|48)\b",
        tooth_specific=True,
    ),
    _c(
        "residual_root",
        "dentition",
        "residual root",
        r"\bresidual\s+roots?\b",
        r"\bretained\s+roots?\b",
        r"\broot\s+remnants?\b",
        tooth_specific=True,
    ),
    _c(
        "caries",
        "dentition",
        "carious lesion",
        r"\bcari(?:es|ous)\b",
        r"\bdecay\b",
        tooth_specific=True,
    ),
    _c(
        "restoration",
        "dentition",
        "coronal restoration",
        r"\brestorations?\b",
        r"\bfillings?\b",
        r"\bcrowns?\b",
        r"\bprosthe\w*\b",
        r"\bbridge\b",
        r"\bonlay\b",
        tooth_specific=True,
    ),
    _c(
        "endodontic",
        "dentition",
        "endodontic treatment",
        r"\bendodontic\w*\b",
        r"\broot\s+canal\s+(?:treat|fill|therap)\w*\b",
        r"\bobturat\w*\b",
        tooth_specific=True,
    ),
    _c(
        "dental_implant",
        "dentition",
        "dental implant",
        r"\bimplants?\b",
        r"\bfixtures?\b",
        tooth_specific=True,
    ),
    _c(
        "supernumerary",
        "dentition",
        "supernumerary tooth",
        r"\bsupernumerar\w*\b",
        r"\bmesiodens\b",
        tooth_specific=True,
    ),
    _c(
        "malposition",
        "dentition",
        "tooth malposition",
        r"\bmalpositio\w*\b",
        r"\bectopic\w*\b",
        r"\bdystopic\b",
        r"\brotat(?:ed|ion)\b",
        r"\btipp(?:ed|ing)\b",
        tooth_specific=True,
    ),
    # --- periapical ------------------------------------------------------
    _c(
        "periapical_lesion",
        "periapical",
        "periapical radiolucency",
        r"\bperiapical\s+(?:lesion|radiolucenc\w*|rarefaction|granulom\w*|process)\b",
        r"\bapical\s+(?:lesion|periodontitis|radiolucenc\w*)\b",
        tooth_specific=True,
    ),
    _c(
        "cyst",
        "periapical",
        "cystic lesion",
        r"\bcyst\w*\b",
        r"\bkeratocyst\w*\b",
        r"\bfollicular\s+lesion\b",
        measurable=True,
    ),
    _c(
        "root_resorption",
        "periapical",
        "root resorption",
        r"\broot\s+resorption\b",
        r"\bexternal\s+resorption\b",
        r"\binternal\s+resorption\b",
        tooth_specific=True,
    ),
    _c(
        "sclerosis",
        "periapical",
        "osseous sclerosis",
        r"\bsclero(?:sis|tic)\b",
        r"\bcondensing\s+osteitis\b",
        r"\bosteosclero\w*\b",
    ),
    # --- periodontal -----------------------------------------------------
    _c(
        "bone_loss",
        "periodontal",
        "periodontal bone loss",
        r"\bperiodontal\s+(?:bone\s+)?(?:loss|defect|disease)\b",
        r"\b(?:horizontal|vertical|angular)\s+bone\s+loss\b",
        r"\bcrestal\s+bone\s+loss\b",
        measurable=True,
    ),
    _c("furcation", "periodontal", "furcation involvement", r"\bfurcation\b"),
    _c("calculus", "periodontal", "calculus deposits", r"\bcalculus\b", r"\btartar\b"),
    # --- mandible --------------------------------------------------------
    _c(
        "mandibular_canal",
        "mandible",
        "mandibular canal",
        r"\bmandibular\s+canal\b",
        r"\binferior\s+alveolar\s+(?:nerve|canal)\b",
        r"\bIAN\b",
        lateralized=True,
    ),
    _c(
        "canal_proximity",
        "mandible",
        "proximity to the mandibular canal",
        r"\b(?:close|proximity|contact|adjacen\w*|abut\w*|contiguous|near)\b[^.]{0,60}"
        r"\b(?:mandibular\s+canal|inferior\s+alveolar)\b",
        r"\b(?:mandibular\s+canal|inferior\s+alveolar)\b[^.]{0,60}\b(?:close|proximity|contact)\b",
        lateralized=True,
        measurable=True,
    ),
    _c("mental_foramen", "mandible", "mental foramen", r"\bmental\s+foram\w*\b", lateralized=True),
    _c("anterior_loop", "mandible", "anterior loop", r"\banterior\s+loop\b", lateralized=True),
    _c(
        "bifid_canal",
        "mandible",
        "bifid mandibular canal",
        r"\bbifid\b",
        r"\bduplicat\w*\s+canal\b",
        r"\bretromolar\s+canal\b",
        lateralized=True,
    ),
    _c(
        "mandibular_atrophy",
        "mandible",
        "mandibular ridge atrophy",
        r"\bmandibul\w*\b[^.]{0,40}\batroph\w*\b",
        r"\batroph\w*\b[^.]{0,40}\bmandibl\w*\b",
        measurable=True,
    ),
    _c(
        "lingual_concavity",
        "mandible",
        "lingual concavity",
        r"\blingual\s+(?:concavit\w*|undercut)\b",
        r"\bsubmandibular\s+fossa\b",
    ),
    # --- maxilla ---------------------------------------------------------
    _c(
        "maxillary_atrophy",
        "maxilla",
        "maxillary ridge atrophy",
        r"\bmaxill\w*\b[^.]{0,40}\batroph\w*\b",
        r"\bridge\s+resorption\b",
        r"\breduced\s+(?:bone\s+)?(?:height|width|volume)\b",
        measurable=True,
    ),
    _c(
        "nasopalatine_canal",
        "maxilla",
        "nasopalatine canal",
        r"\bnasopalatine\s+(?:canal|duct)\b",
        r"\bincisive\s+canal\b",
        measurable=True,
    ),
    _c(
        "nasal_cavity",
        "maxilla",
        "nasal cavity",
        r"\bnasal\s+(?:cavit\w*|floor|fossa)\b",
        r"\bseptum\s+deviat\w*\b",
    ),
    _c(
        "bone_density",
        "maxilla",
        "bone density",
        r"\bbone\s+(?:densit\w*|qualit\w*|mineraliz\w*)\b",
        r"\b(?:D[1-4]|type\s+[IVX]+)\s+bone\b",
        r"\btrabecular\s+bone\b",
        r"\bcortical\s+(?:bone|plate|thickness)\b",
        measurable=True,
    ),
    _c(
        "ridge_dimension",
        "maxilla",
        "alveolar ridge dimensions",
        r"\balveolar\s+(?:ridge|crest|process)\b",
        r"\bbone\s+(?:height|width)\b",
        r"\bresidual\s+ridge\b",
        measurable=True,
    ),
    # --- sinus -----------------------------------------------------------
    _c(
        "maxillary_sinus",
        "sinus",
        "maxillary sinus",
        r"\bmaxillary\s+sinus\w*\b",
        r"\bantrum\b",
        r"\bantral\b",
        lateralized=True,
    ),
    _c(
        "mucosal_thickening",
        "sinus",
        "sinus mucosal thickening",
        r"\bmucosal?\s+thicken\w*\b",
        r"\bmucous\s+membrane\s+thicken\w*\b",
        r"\bsinus\s+mucos\w*\b",
        lateralized=True,
        measurable=True,
    ),
    _c(
        "sinusitis",
        "sinus",
        "sinusitis",
        r"\bsinusit\w*\b",
        r"\bopacif\w*\b",
        r"\bair[-\s]fluid\s+level\b",
        r"\bcomplete\s+opacit\w*\b",
        lateralized=True,
    ),
    _c(
        "mucous_retention_cyst",
        "sinus",
        "mucous retention cyst",
        r"\bretention\s+(?:cyst|pseudocyst)\b",
        r"\bmucocele\b",
        r"\bpolyp\w*\b",
        lateralized=True,
        measurable=True,
    ),
    _c(
        "sinus_septa",
        "sinus",
        "sinus septum",
        r"\bsept(?:um|a)\b[^.]{0,30}\bsinus\b",
        r"\bsinus\b[^.]{0,30}\bsept(?:um|a)\b",
        r"\bunderwood\b",
        lateralized=True,
    ),
    _c(
        "sinus_pneumatization",
        "sinus",
        "sinus pneumatization",
        r"\bpneumatiz\w*\b",
        lateralized=True,
    ),
    _c(
        "oroantral_relation",
        "sinus",
        "root-sinus relationship",
        r"\broots?\b[^.]{0,50}\bsinus\b",
        r"\bsinus\s+floor\b[^.]{0,50}\broots?\b",
        r"\boroantral\b",
        lateralized=True,
    ),
    # --- TMJ -------------------------------------------------------------
    _c(
        "tmj",
        "tmj",
        "temporomandibular joint",
        r"\btemporomandibular\b",
        r"\bTMJ\b",
        r"\bcondyl\w*\b",
        r"\bglenoid\s+fossa\b",
        lateralized=True,
    ),
    _c(
        "condylar_degeneration",
        "tmj",
        "condylar degenerative change",
        r"\bcondyl\w*\b[^.]{0,50}\b(?:flatten\w*|erosion|osteophyt\w*|degenerat\w*|"
        r"remodel\w*|irregular\w*|sclerot\w*)\b",
        r"\bdegenerative\s+(?:change|joint)\w*\b",
        lateralized=True,
    ),
    _c(
        "condylar_asymmetry",
        "tmj",
        "condylar asymmetry",
        r"\basymmetr\w*\b[^.]{0,40}\bcondyl\w*\b",
        r"\bcondyl\w*\b[^.]{0,40}\basymmetr\w*\b",
    ),
    # --- other -----------------------------------------------------------
    _c(
        "airway",
        "other",
        "upper airway",
        r"\bairway\b",
        r"\bpharyn\w*\b",
        r"\badenoid\w*\b",
        r"\bturbinate\w*\b",
    ),
    _c(
        "fracture",
        "other",
        "fracture",
        r"\bfractur\w*\b",
        r"\bdiscontinuit\w*\s+of\s+the\s+cortex\b",
    ),
    _c(
        "foreign_body",
        "other",
        "foreign body",
        r"\bforeign\s+bod\w*\b",
        r"\bradiopaque\s+material\b",
        r"\bextruded\s+(?:material|cement)\b",
        r"\bsurgical\s+(?:plate|screw|mesh)\b",
    ),
    _c(
        "bone_graft",
        "other",
        "bone graft",
        r"\bgraft\w*\b",
        r"\bsinus\s+lift\b",
        r"\baugmentation\b",
        r"\bbiomaterial\b",
    ),
    _c(
        "osteonecrosis",
        "other",
        "osteonecrosis",
        r"\bosteonecro\w*\b",
        r"\bMRONJ\b",
        r"\bsequestr\w*\b",
    ),
    _c(
        "tonsillolith",
        "other",
        "calcification",
        r"\bcalcification\w*\b",
        r"\btonsillolith\w*\b",
        r"\bcarotid\s+calcif\w*\b",
        r"\bsialolith\w*\b",
    ),
    _c(
        "artifact",
        "other",
        "imaging artifact",
        r"\bartifact\w*\b",
        r"\bscatter\b",
        r"\bbeam\s+hardening\b",
        r"\bmotion\b",
    ),
    # --- impression ------------------------------------------------------
    _c(
        "no_significant_finding",
        "impression",
        "no significant abnormality",
        r"\bno\s+(?:significant|relevant|evident|obvious)\s+(?:abnormalit\w*|finding\w*|"
        r"patholog\w*|lesion\w*)\b",
        r"\bwithin\s+normal\s+limits\b",
        r"\bunremarkable\b",
    ),
)

CONCEPTS_BY_KEY: dict[str, Concept] = {concept.key: concept for concept in CONCEPTS}
CONCEPT_KEYS: tuple[str, ...] = tuple(concept.key for concept in CONCEPTS)
SECTION_INDEX: dict[str, int] = {name: index for index, name in enumerate(SECTIONS)}


@dataclass(frozen=True, slots=True)
class Mention:
    """One concept occurrence with its extracted qualifiers."""

    concept: str
    negated: bool
    uncertain: bool
    laterality: str
    teeth: tuple[int, ...] = ()
    measurements: tuple[float, ...] = field(default=())

    @property
    def polarity(self) -> str:
        if self.negated:
            return "absent"
        return "uncertain" if self.uncertain else "present"


def extract_teeth(text: str) -> tuple[int, ...]:
    """Extract FDI tooth numbers, preferring explicit ``tooth NN`` constructions."""
    found: list[int] = []
    for match in TOOTH_NUMBER_RE.finditer(text):
        found.extend(int(value) for value in re.findall(r"\d{2}", match.group(1)))
    if not found:
        found = [int(value) for value in BARE_FDI_RE.findall(text)]
    valid = {number for number in found if number in FDI_PERMANENT or number in FDI_PRIMARY}
    return tuple(sorted(valid))


def extract_measurements(text: str) -> tuple[float, ...]:
    values: list[float] = []
    for amount, unit in MEASUREMENT_RE.findall(text):
        value = float(amount.replace(",", "."))
        values.append(value * 10.0 if unit.lower().startswith(("cm", "centi")) else value)
    return tuple(values)


def detect_laterality(text: str) -> str:
    if BILATERAL_RE.search(text):
        return "bilateral"
    left, right = bool(LEFT_RE.search(text)), bool(RIGHT_RE.search(text))
    if left and right:
        return "bilateral"
    if left:
        return "left"
    if right:
        return "right"
    return "unspecified"


def extract_mentions(phrase: str) -> list[Mention]:
    """Map one phrase onto every ontology concept it expresses."""
    negated = bool(NEGATION_RE.search(phrase))
    uncertain = bool(UNCERTAINTY_RE.search(phrase))
    laterality = detect_laterality(phrase)
    teeth = extract_teeth(phrase)
    measurements = extract_measurements(phrase)
    return [
        Mention(
            concept=concept.key,
            negated=negated,
            uncertain=uncertain,
            laterality=laterality if concept.lateralized else "unspecified",
            teeth=teeth if concept.tooth_specific else (),
            measurements=measurements if concept.measurable else (),
        )
        for concept in CONCEPTS
        if concept.matches(phrase)
    ]


def concept_profile(phrases: Iterable[str]) -> dict[str, str]:
    """Reduce a report to ``{concept: polarity}`` for slicing and error analysis."""
    profile: dict[str, str] = {}
    priority = {"present": 3, "uncertain": 2, "absent": 1}
    for phrase in phrases:
        for mention in extract_mentions(phrase):
            current = profile.get(mention.concept)
            if current is None or priority[mention.polarity] > priority[current]:
                profile[mention.concept] = mention.polarity
    return profile


def section_of(phrase: str) -> str:
    """Best-guess report section for a phrase, used to order generated sentences."""
    for concept in CONCEPTS:
        if concept.matches(phrase):
            return concept.section
    return "other"


def section_rank(phrase: str) -> int:
    return SECTION_INDEX.get(section_of(phrase), SECTION_INDEX["other"])
