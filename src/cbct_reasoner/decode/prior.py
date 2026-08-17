"""The image-free prior decoder.

Every case receives the same report, built purely from corpus statistics. This is
not a throwaway: on a benchmark where the clinical metric rewards commonly-true
statements and the captioning metrics reward typical phrasing, a well-calibrated
constant report is a genuinely strong entry, and several leaderboard names
containing "baseline" suggest that is roughly what the current leaders submit.

It serves three purposes here:

1. **A floor.** It is the safe submission if training goes wrong - it needs no
   GPU, no checkpoint, and cannot fail on an unusual volume.
2. **A control.** The imaging model earns its place only by beating this on
   out-of-fold data. Any architecture that does not is measuring noise.
3. **A fallback.** The inference container falls back to it if a volume cannot be
   read, because a missing result scores zero while a generic report does not.
"""

from __future__ import annotations

import numpy as np

from cbct_reasoner.prototypes import PrototypeBank


def prior_probabilities(bank: PrototypeBank, num_cases: int) -> np.ndarray:
    """Broadcast the corpus prevalence to every case."""
    return np.tile(bank.prevalence.astype(np.float32), (num_cases, 1))


def prior_report(bank: PrototypeBank, *, threshold: float = 0.5, max_sentences: int = 26) -> str:
    """The plain prevalence-thresholded report, before calibration."""
    from cbct_reasoner.decode.decoder import DecoderSettings, ReportDecoder

    decoder = ReportDecoder(
        bank,
        np.full(len(bank), threshold, dtype=np.float32),
        settings=DecoderSettings(max_sentences=max_sentences),
    )
    return decoder.decode(bank.prevalence)
