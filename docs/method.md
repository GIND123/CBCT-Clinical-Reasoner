# Baseline and extension guide

## Executable baseline

The included model samples each axis to at most 96 voxels and extracts:

- log voxel dimensions, spacing, and physical extent;
- robust intensity quantiles and normalized moments;
- a 16-bin clipped intensity histogram;
- 4 × 4 pooled mean projections along each of three axes.

Features are median-centered and IQR-scaled. The closest training feature vector by mean
squared distance supplies a report. When a case has multiple references, the report with
the highest average token-set agreement is selected as its deterministic representative.

This provides a useful artifact and container test with low compute. Its limitations are
fundamental: global features do not localize findings, nearest-neighbor text can import
unsupported facts, and full reports in the artifact can be sensitive data. It must remain
labeled a non-clinical baseline.

## Extension interface

Replace `RetrievalReportModel` behind `grand_challenge.run` while preserving:

```python
report: str = model.predict(volume)
write_challenge_output(report, "/output/diagnostic-imaging-report.json")
```

A stronger implementation should return both a report and local-only structured evidence.
Only the report enters the challenge JSON. Persist evidence during validation for clinical
error analysis, but never add undeclared fields to the submitted output.

## Target architecture

Recommended components are a 3D anatomy encoder initialized from ToothFairy segmentation,
tooth- and region-level crops, an ontology prediction head, a relation graph for critical
anatomy, and a constrained text renderer. Use physical-coordinate measurements and retain
an inverse transform to source voxels for evidence overlays.

For each emitted statement, require support from a structured fact. Apply deterministic
rules for tooth identifiers, laterality, numbers, and negation. If a language model is used,
it should realize facts—not discover new facts in unconstrained decoding.
