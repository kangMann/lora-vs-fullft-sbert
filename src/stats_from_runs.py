"""Recompute the best-versus-best contrast (Table 4 of the manuscript) from the per-run table.

Needs no GPU and no model weights. It reads results/tables_v4/T15_per_run.csv plus
configs/lr_final_v4.json and configs/best_pairs_v4.json, then reports the paired
mean difference, the 95% t-interval, the TOST p-value and the Holm-adjusted p-value.
The paired bootstrap over test pairs needs the saved predictions (pred_*.npy) and is not repeated here.

Usage:  python src/stats_from_runs.py [--delta 0.01] [--metric test_en_spearman]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
STAGES = ("confirm", "confirm_extra", "repair")


def load_valid_runs(metric):
    d = pd.read_csv(ROOT / "results/tables_v4/T15_per_run.csv")
    lr_final = json.load(open(ROOT / "configs/lr_final_v4.json"))
    d = d[d.stage.isin(STAGES)].copy()
    key = d.model + "|" + d.method + "|" + d["rank"].astype(str) + "|" + d.strategy
    d = d[np.isclose(d.lr, key.map(lr_final))]
    d["pri"] = (d.stage == "repair").astype(int)           # a repaired run replaces the original
    d = d.sort_values("pri").drop_duplicates(["model", "method", "rank", "strategy", "seed"], keep="last")
    return d[~d.diverged].dropna(subset=[metric])          # collapsed runs are excluded


def tost(diff, delta):
    n = len(diff)
    m, sd = diff.mean(), diff.std(ddof=1)
    se = sd / np.sqrt(n)
    t_lo, t_hi = (m + delta) / se, (m - delta) / se
    p = max(stats.t.sf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
    h = stats.t.ppf(0.975, n - 1) * se
    return dict(n_seed=n, mean_diff=m, sd_diff=sd, ci_lo=m - h, ci_hi=m + h, p_tost=p)


def holm(ps):
    order = np.argsort(ps)
    out, run = np.empty(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, (len(ps) - i) * ps[k])
        out[k] = min(1.0, run)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--metric", default="test_en_spearman")
    a = ap.parse_args()
    d = load_valid_runs(a.metric)
    best = json.load(open(ROOT / "configs/best_pairs_v4.json"))
    rows = []
    for model, bp in best.items():
        (fm, fr, fs), (lm, lr_, ls) = bp["full"], bp["lora"]
        fu = d[(d.model == model) & (d.method == fm) & (d["rank"] == fr) & (d.strategy == fs)].set_index("seed")[a.metric]
        lo = d[(d.model == model) & (d.method == lm) & (d["rank"] == lr_) & (d.strategy == ls)].set_index("seed")[a.metric]
        seeds = sorted(set(fu.index) & set(lo.index))
        r = tost((lo.loc[seeds] - fu.loc[seeds]).values, a.delta)
        r.update(model=model, lora_rank=lr_, strategy=ls, seeds=f"{min(seeds)}-{max(seeds)}")
        rows.append(r)
    t = pd.DataFrame(rows)
    t["p_tost_holm"] = holm(t.p_tost.values)
    t["decision"] = np.where(t.p_tost_holm < 0.05, "equivalent", "inconclusive")
    cols = ["model", "lora_rank", "strategy", "n_seed", "mean_diff", "ci_lo", "ci_hi", "p_tost", "p_tost_holm", "decision"]
    print(t[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
