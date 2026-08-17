# Grand Challenge submission checklist

## Interface

- Read exactly one case from `/input/images/cbct/*.mha` on socket `cbct-image`.
- Write `/output/diagnostic-imaging-report.json` on socket
  `diagnostic-imaging-report`.
- Emit a JSON object containing exactly one string field named `report`.
- Run fully automatically, one case at a time, without internet access.

## Before building

- Train/freeze the model and place it in `model/`.
- Record its SHA-256 checksum and every external data/model source.
- Pin dependencies and the CUDA base image if GPU inference is introduced.
- Confirm all artifacts are allowed by their licenses and challenge cutoff rules.
- Remove secrets, local data paths, patient data, debug dumps, and API keys.

## Smoke tests

- Empty input fails clearly.
- Multiple input volumes fail clearly.
- A valid organizer-format MHA produces valid UTF-8 JSON.
- Very small, anisotropic, metal-heavy, and truncated scans do not crash.
- Repeated runs are deterministic within the stated tolerance.
- The image runs with networking disabled and within platform memory/runtime limits.
- The saved Docker archive loads and runs on a second machine.

## Evidence packet

Archive the source commit, Docker digest/archive checksum, model checksum, split manifest,
training configuration, development metrics by center, hardware/runtime measurements, and
method/data declaration. The challenge requires public reproducible code for top-three
methods and public algorithms/parameters for post-challenge publication eligibility.
