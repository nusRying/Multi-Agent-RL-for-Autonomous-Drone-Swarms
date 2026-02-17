# Config Templates

This folder holds editable experiment configs so your runs are reproducible.

## Included templates

- `configs/curriculum_v1.yaml`
  - Stage-by-stage training progression.
- `configs/domain_randomization_v1.yaml`
  - Physics and sensor randomization ranges.
- `configs/eval_protocol_v1.yaml`
  - Seed list, metric thresholds, and evaluation episode counts.
- `configs/eval_protocol_quick.yaml`
  - Fast sanity-check protocol (small seed/episode budget).

## Usage pattern

1. Copy a template to a new versioned file (for example `curriculum_v2.yaml`).
2. Modify only one group of parameters at a time.
3. Log the exact config filename in run logs and weekly reports.

## Notes

- `scripts/train_curriculum.py` reads `curriculum_*.yaml` directly.
- `scripts/evaluate_protocol.py` reads `eval_protocol_*.yaml` directly.
- `scripts/train_single_agent.py` and `scripts/train_multi_agent.py` currently use CLI args.
