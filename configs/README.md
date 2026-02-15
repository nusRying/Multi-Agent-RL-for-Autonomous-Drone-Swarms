# Config Templates

This folder holds editable experiment configs so your runs are reproducible.

## Included templates

- `configs/curriculum_v1.yaml`
  - Stage-by-stage training progression.
- `configs/domain_randomization_v1.yaml`
  - Physics and sensor randomization ranges.
- `configs/eval_protocol_v1.yaml`
  - Seed list, metric thresholds, and evaluation episode counts.

## Usage pattern

1. Copy a template to a new versioned file (for example `curriculum_v2.yaml`).
2. Modify only one group of parameters at a time.
3. Log the exact config filename in run logs and weekly reports.

## Notes

- Current training scripts use CLI args directly.
- These files are designed for immediate documentation/reproducibility and easy integration when you add config loaders.

