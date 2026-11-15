"""Run the NO2 experiments and write tables and figures to ./results.

Examples
  python run_experiments.py --quick                    # lighter run
  python run_experiments.py --models hgb_clf hgb_reg   # gradient boosting only (no TensorFlow)
  python run_experiments.py --seeds 5                  # fuller run
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from no2_pipeline import (N_BINS, TARGET, class_edges, evaluate, fit_model, load_data,
                          make_split, permutation_importance, preprocess)

METRICS = ["acc", "macro_f1", "within1", "mae_class", "adj_err_share", "under_err_share"]
ALL_MODELS = ["mlp_plain", "mlp_reg", "hgb_clf", "hgb_reg"]
ALL_SPLITS = ["random", "block", "chrono"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/AirQualityUCI.csv")
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS)
    p.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--features", nargs="+", default=None,
                   help="feature set names (default: all five)")
    p.add_argument("--quick", action="store_true",
                   help="2 seeds, features all+sensors, models hgb_clf+mlp_reg")
    args = p.parse_args()
    if args.quick:
        args.seeds, args.features = 2, ["all", "sensors"]
        args.models = [m for m in args.models if m in ("hgb_clf", "mlp_reg")]
    return args


# --------------------------------------------------------------------------
# Experiment grid
# --------------------------------------------------------------------------
def run_grid(df, feature_sets, args):
    y = df[TARGET].to_numpy()
    uses_mlp = any(m.startswith("mlp") for m in args.models)
    rows = []
    for split in args.splits:
        # the chronological split is fixed, so extra seeds only matter for the networks
        seeds = range(args.seeds) if (split != "chrono" or uses_mlp) else range(1)
        for seed in seeds:
            tr, te = make_split(df, split, seed)
            edges = class_edges(y[tr])
            y_tr, y_te = np.digitize(y[tr], edges), np.digitize(y[te], edges)
            for fs_name in args.features:
                cols = feature_sets[fs_name]
                X_tr, X_te = preprocess(df.loc[tr, cols], df.loc[te, cols])
                for model in args.models:
                    predict = fit_model(model, X_tr, y[tr], y_tr, edges, seed)
                    scores = evaluate(y_te, predict(X_te))
                    rows.append(dict(split=split, seed=seed, features=fs_name,
                                     model=model, **scores))
                    print(f"{split:6s} seed={seed} {fs_name:18s} {model:9s} "
                          f"macro_f1={scores['macro_f1']:.3f}", flush=True)
    return pd.DataFrame(rows)


def diagnostics(df, feature_sets, args, out):
    """Confusion matrices, per-class reports and permutation importance (seed 0)."""
    y = df[TARGET].to_numpy()
    diag_models = [m for m in ("hgb_clf", "mlp_reg") if m in args.models]
    confusions, importances = {}, {}
    for split in ("random", "chrono"):
        if split not in args.splits:
            continue
        tr, te = make_split(df, split, 0)
        edges = class_edges(y[tr])
        y_tr, y_te = np.digitize(y[tr], edges), np.digitize(y[te], edges)
        print(f"\n[{split}] test class counts: {np.bincount(y_te, minlength=N_BINS)}")
        for fs_name in ("all", "sensors"):
            if fs_name not in args.features:
                continue
            cols = feature_sets[fs_name]
            X_tr, X_te = preprocess(df.loc[tr, cols], df.loc[te, cols])
            for model in diag_models:
                predict = fit_model(model, X_tr, y[tr], y_tr, edges, 0)
                y_pred = predict(X_te)
                cm = confusion_matrix(y_te, y_pred, labels=range(N_BINS))
                tag = f"{model}_{split}_{fs_name}"
                pd.DataFrame(cm).to_csv(out / f"confusion_{tag}.csv")
                (out / f"report_{tag}.txt").write_text(
                    classification_report(y_te, y_pred, zero_division=0))
                if model != "hgb_clf":
                    continue
                confusions[(split, fs_name)] = cm
                if split == "chrono":
                    imp = permutation_importance(predict, X_te, y_te, cols)
                    imp.to_csv(out / f"importance_{fs_name}.csv")
                    importances[fs_name] = imp
                    print(f"\nTop features by macro-F1 drop ({fs_name}, chrono):")
                    print(imp.head(8).round(3).to_string())
    return confusions, importances


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_split_comparison(runs, out):
    gb = runs[runs["model"] == "hgb_clf"]
    if gb.empty:
        return
    table = gb.groupby(["features", "split"])["macro_f1"].mean().unstack()
    table = table[[s for s in ALL_SPLITS if s in table.columns]]
    ax = table.plot.bar(figsize=(8, 4))
    ax.set_ylabel("macro-F1 (gradient boosting)")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out / "macro_f1_by_split.png", dpi=150)
    plt.close()


def plot_confusions(confusions, out):
    keys = [k for k in (("random", "sensors"), ("chrono", "sensors")) if k in confusions]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 4))
    for ax, key in zip(np.atleast_1d(axes), keys):
        cm = confusions[key]
        ax.imshow(cm, cmap="Blues")
        for i in range(N_BINS):
            for j in range(N_BINS):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"{key[0]} split, {key[1]} features")
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
    plt.tight_layout()
    plt.savefig(out / "confusion_sensors_random_vs_chrono.png", dpi=150)
    plt.close()


def plot_importance(importances, out):
    for fs_name, imp in importances.items():
        top = imp.head(8)[::-1]
        plt.figure(figsize=(6, 3.5))
        plt.barh(top.index, top.values)
        plt.xlabel("drop in macro-F1 when shuffled")
        plt.title(f"Permutation importance ({fs_name}, chronological test)")
        plt.tight_layout()
        plt.savefig(out / f"importance_{fs_name}.png", dpi=150)
        plt.close()


# --------------------------------------------------------------------------
def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, feature_sets = load_data(args.csv)
    args.features = args.features or list(feature_sets)
    unknown = [f for f in args.features if f not in feature_sets]
    if unknown:
        raise SystemExit(f"unknown feature sets {unknown}; choose from {list(feature_sets)}")
    print(f"{len(df)} labelled rows, {df['dt'].min():%Y-%m-%d} to {df['dt'].max():%Y-%m-%d}")

    runs = run_grid(df, feature_sets, args)
    runs.to_csv(out / "runs.csv", index=False)
    summary = (runs.groupby(["split", "features", "model"])[METRICS]
                   .agg(["mean", "std"]).round(3))
    summary.to_csv(out / "summary.csv")
    print("\n=== Summary (mean / std over seeds) ===")
    print(summary.to_string())

    confusions, importances = diagnostics(df, feature_sets, args, out)
    plot_split_comparison(runs, out)
    plot_confusions(confusions, out)
    plot_importance(importances, out)
    print(f"\nResults written to {out}/")


if __name__ == "__main__":
    main()
