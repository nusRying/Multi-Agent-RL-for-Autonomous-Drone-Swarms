# Troubleshooting (Windows + Ray RLlib)

## 1. `metrics exporter agent` Errors

Example:
- `Failed to establish connection to the metrics exporter agent`

Meaning:
- Usually non-fatal on Windows local runs.
- Training/evaluation can continue.

Action:
- Ignore unless run stops with a traceback.

## 2. `ArrowInvalid: URI has empty scheme` or `Expected a local filesystem path, got a URI`

Cause:
- Ray/PyArrow checkpoint path parsing differs across versions and platforms.

Status in this repo:
- `scripts/evaluate_protocol.py` and `scripts/train_curriculum.py` now normalize checkpoint paths and try fallback restore formats automatically.

Action:
- Pass a normal path:
  - `--checkpoint "checkpoints/multi_debug"`
- If needed, pass exact checkpoint directory containing `algorithm_state.pkl`.

Find checkpoint folders:
```powershell
Get-ChildItem -Recurse checkpoints -Filter algorithm_state.pkl | Select-Object -ExpandProperty DirectoryName
```

## 3. PowerShell Error with `<...>` Placeholder

Example:
- `The '<' operator is reserved for future use.`

Cause:
- Angle brackets were interpreted as shell syntax.

Action:
- Do not include `< >` in commands.
- Replace placeholders with real paths.

## 4. `reward_mean=0.000 len_mean=0.00` Every Iteration

Cause:
- Ray 2.5x result keys changed; old metric keys can appear empty.

Status in this repo:
- Training scripts now read both old and new result paths via `src/swarm_marl/utils/ray_metrics.py`.

Action:
- Pull latest project files and rerun training.
- Check CSV log (for example `reports/metrics/multi_debug.csv`) to confirm metrics are updating.

## 5. Evaluation Appears Stuck / Very Slow

Cause:
- Full protocol is large (`5 seeds x 100 episodes x 4 scenarios`).

Action:
- Use quick protocol first:
  - `configs/eval_protocol_quick.yaml`
- Then run full protocol once checkpoint quality is acceptable.

## 6. Matplotlib Import Breaks in `comp_vision`

Potential symptom:
- Import errors pointing to `AppData\Roaming\Python\Python39\site-packages`.

Cause:
- Broken user-site package overshadowing conda env package.

Status in this repo:
- `scripts/plot_metrics.py` strips user-site paths before importing matplotlib.

## 7. Ray Deprecation Warnings

Example:
- `UnifiedLogger will be removed...`
- `build has been deprecated...`

Meaning:
- Informational unless traceback follows.

Action:
- Safe to ignore for now.
- We can migrate to newer Ray APIs later once training is stable.

