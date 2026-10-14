# LoRA versus Full Fine-Tuning of Sentence-BERT for Semantic Textual Similarity

Code, configuration and frozen results for the article *"LoRA versus Full Fine-Tuning of Sentence-BERT for
Semantic Textual Similarity: A Systematic Empirical Study"* (Fitriyandhi & Marco).

The study trains `xlm-roberta-base` and `roberta-base` as Sentence-BERT encoders on English STS-B with full
fine-tuning and with LoRA (r = 4, 8, 16, 32), under three training strategies (S0 cosine-MSE, S1 S0 + layer-wise LR decay,
S2 CoSENT). It tests equivalence at a margin of 0.01 Spearman (TOST, Holm correction), measures memory, time, energy and
checkpoint size, and evaluates zero-shot transfer to Indonesian.

## Main results

| Model | Best LoRA vs best full fine-tuning (S2) | Seeds (n) | Δ Spearman (95% CI) | Holm p_TOST |
|---|---|---|---|---|
| xlm-roberta-base | r = 8 | 8 | −0.0013 (−0.0066, 0.0040) | 0.003 |
| roberta-base | r = 16 | 9 | +0.0032 (−0.0004, 0.0069) | 0.003 |

Both contrasts are equivalent within δ = 0.01. Seeds 1–10 were run for the best pair; collapsed runs were excluded and
the two methods were paired by seed, which gives n = 8 and n = 9. Details, limitations and the other research questions are in the article.

## Repository layout

```
├── README.md
├── LICENSE                 MIT (code). Datasets keep their own licenses.
├── CITATION.cff
├── requirements.txt        pinned versions; the full Colab freeze is in configs/
├── src/
│   ├── sts_lora_lib.py     training, evaluation, collapse detection, TOST, bootstrap (library used by the notebook)
│   └── stats_from_runs.py  recomputes the best-versus-best contrast from the per-run table (no GPU)
├── notebooks/              Colab notebook v4.9: the full pipeline, stage by stage
├── configs/                experiment_config.json, grid_30_cells.csv, lr_search/lr_final/best_pairs, Colab freeze
├── results/
│   ├── runs_json/          413 per-run result files (403 from runs/, 10 from bench/; see its README)
│   ├── tables_v4/          frozen tables T01–T15 (CSV), Excel and Markdown versions
│   └── audit/              weight audit, retrained run, collapsed runs, GPU session log
├── figures/                make_figures.py and the four manuscript figures
└── SHA256SUMS.txt          checksums of source, config, result and figure files
```

## Quick start (no GPU, about two minutes)

```bash
git clone https://github.com/kangMann/lora-vs-fullft-sbert.git
cd lora-vs-fullft-sbert
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy==2.1.3 pandas==2.2.3 scipy==1.16.3 matplotlib==3.10.0

sha256sum -c SHA256SUMS.txt                               # 1. files are unchanged (macOS: shasum -a 256 -c); run it before step 4
python src/stats_from_runs.py                             # 2. reproduces Table 4 of the article
python src/stats_from_runs.py --delta 0.0075              # 3. sensitivity to the margin
python figures/make_figures.py                            # 4. redraws the figures
```

Expected output of step 2:

```
           model  lora_rank strategy  n_seed  mean_diff   ci_lo  ci_hi  p_tost  p_tost_holm   decision
xlm-roberta-base          8       S2       8    -0.0013 -0.0066 0.0040  0.0030       0.0030 equivalent
    roberta-base         16       S2       9     0.0032 -0.0004 0.0069  0.0013       0.0027 equivalent
```

The paired bootstrap over test pairs needs the saved predictions (`pred_*.npy`, not in this repository). The notebook produces them.

## Full re-run on Google Colab

The reported runs used one NVIDIA L4 (22 GB), Python 3.13, PyTorch 2.11.0, Transformers 5.19.0, PEFT 0.21.2. Budget about 13 h of training
(403 main runs) and 15 Colab sessions. A different GPU changes time, memory and energy; accuracy should stay close but not identical.

1. Open `notebooks/Eksperimen_LoRA_STS_Colab_v4_9.ipynb` in Colab and choose *Runtime → Change runtime type → L4 GPU*.
2. Run the cells from top to bottom. Cell 5 writes `sts_lora_lib.py` (same file as `src/sts_lora_lib.py`) to `MyDrive/sts_lora/code/`.
3. The notebook resumes after a disconnect: finished runs are skipped, and unfinished ones continue from the last epoch.
4. Outputs go to `MyDrive/sts_lora/` (`runs/`, `stages/`, `tables_v4b/`, `figures_v4b/`, `logs/`). Do not change `ROOT` after the first run.
5. Optional: add `HF_TOKEN` to Colab *Secrets* to avoid Hugging Face rate limits.

## Data

Data are not redistributed here. `load_data` in `src/sts_lora_lib.py` downloads them from Hugging Face. Check each dataset card for its license.

| Use | Hugging Face id | Reference |
|---|---|---|
| Training, validation, English test | `sentence-transformers/stsb` | Cer et al. (2017) |
| English and Indonesian transfer | `SemRel/SemRel2024` (configs `eng`, `ind`) | Ousidhoum et al. (2024) |
| Indonesian test (machine-translated STS-B) | `LazarusNLP/stsb_mt_id` | LazarusNLP |

Only English STS-B was used for training. No model saw Indonesian training data.

## Notes for reviewers and re-users

- **Collapse.** A run counts as collapsed under criteria (a)–(c) described in the article; criterion (c) was added after seeing the data. `results/audit/collapsed_runs.csv` lists the 38 collapsed runs among the 403.
- **Stability probe.** Seeds 43 and 44 re-ran each selected learning rate. When a probe run collapsed, the rate was lowered one grid step. This changed the rate for four xlm-roberta-base configurations. Probe results are in `results/tables_v4/T04b_uji_stabilitas.csv` and in no other table.
- **Retrained run.** One run was retrained after a Google Drive write failure (`results/audit/T16b_run_dilatih_ulang.csv`).
- **Energy.** Energy comes from NVML power readings on the GPU only. The one-epoch benchmark energy did not match the training runs, and the cause is unresolved. Treat energy as less certain than time and memory.
- **Table names.** File names and column names in `results/` are in Indonesian, as produced by the notebook (for example `keputusan` = decision, `Ekuivalen` = equivalent).

## License

MIT for code, tables and figures (see `LICENSE`). Datasets keep the licenses of their original sources.

## Citation

Please cite the article and this repository (see `CITATION.cff`). After the Zenodo release, add the DOI here.

Contact: Ari Fitriyandhi, a.fitriyandhi@students.amikom.ac.id
