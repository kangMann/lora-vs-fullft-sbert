# Per-run result files

Put the 413 `result.json` files of the experiment here, one file per run, named `<run_id>.json`.
They are written by the Colab notebook to Google Drive:

| Drive folder (`MyDrive/sts_lora/`) | Runs | Copy? |
|---|---|---|
| `runs/<run_id>/result.json` | 403 main runs (LR search, screening, probe, confirmation, ablation, pilot) | yes |
| `bench/NVIDIA_L4/<run_id>/result.json` | 10 one-epoch efficiency benchmark runs | yes |
| `runs_invalid/` | the run that failed the weight audit and was retrained (archived, not used) | no |
| `stages/`, `tables_v4/`, `logs/`, `code/` | stage files, tables, session log, library | already in `configs/`, `results/` and `src/` |

```bash
# after downloading the Drive folder sts_lora to your computer (e.g. ~/Downloads/sts_lora)
SRC=~/Downloads/sts_lora
mkdir -p results/runs_json
for d in "$SRC"/runs/*/ "$SRC"/bench/NVIDIA_L4/*/; do
  id=$(basename "$d"); [ -f "$d/result.json" ] && cp "$d/result.json" "results/runs_json/$id.json"
done
ls results/runs_json/*.json | wc -l      # expect 413
```

Do not copy `*.pt` checkpoints or `pred_*.npy` into the repository. The same numbers for the 403 main runs are in
`results/tables_v4/T15_per_run.csv`.
