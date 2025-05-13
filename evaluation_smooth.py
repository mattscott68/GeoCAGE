#!/usr/bin/env python3
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

# import your two functions
from run_cage_mlp   import run_mlp
from run_cage_gcn_smooth import run_gcn_with_smoothness as run_gcn

def main():
    datasets = ["cora", "citeseer", "mcc", "pubmed", "dblp"]
    seeds    = [220, 221, 222, 223, 224, 225, 226, 227, 228, 229]

    all_records = []

    for ds in datasets:
        ds_records = []

        for seed in seeds:
            # ─── MLP baseline ───────────────────────────────────────────
            print(f"Seed: {seed}, Dataset: {ds}, MLP")
            t0 = time.time()
            cls_mlp, clu_mlp = run_mlp(ds, seed)
            t_mlp = time.time() - t0
            print(f"→ MLP time: {t_mlp:.2f}s")

            # ─── GCN + Smooth experiment ─────────────────────────────────
            print(f"Seed: {seed}, Dataset: {ds}, GCN+Smooth")
            t0 = time.time()
            cls_gcn, clu_gcn = run_gcn(ds, seed)
            t_gcn = time.time() - t0
            print(f"→ GCN+Smooth time: {t_gcn:.2f}s")

            # print detailed splits for smooth model
            print("=== GCN+Smooth Classification ===")
            print(cls_gcn)
            print("\n=== GCN+Smooth Clustering (rand_score) ===")
            print(clu_gcn.loc[['rand_score']])

            # ─── Gather metrics ───────────────────────────────────────────
            def summarize(df, metric):
                return df.loc[metric].mean()

            # MLP metrics
            mlp_rec = {
                "dataset": ds, "model": "MLP", "seed": seed,
                "accuracy":        summarize(cls_mlp, "accuracy_score"),
                "precision_macro": summarize(cls_mlp, "precision_macro"),
                "precision_micro": summarize(cls_mlp, "precision_micro"),
                "precision_weighted": summarize(cls_mlp, "precision_weighted"),
                "recall_macro":    summarize(cls_mlp, "recall_macro"),
                "recall_micro":    summarize(cls_mlp, "recall_micro"),
                "recall_weighted": summarize(cls_mlp, "recall_weighted"),
                "f1_macro":        summarize(cls_mlp, "f1_macro"),
                "f1_micro":        summarize(cls_mlp, "f1_micro"),
                "f1_weighted":     summarize(cls_mlp, "f1_weighted"),
                "rand_score":      clu_mlp.loc["rand_score", "all"],
                "time_s":          t_mlp
            }

            # GCN+Smooth metrics
            gcn_rec = {
                "dataset": ds, "model": "GCN+Smooth", "seed": seed,
                "accuracy":        summarize(cls_gcn, "accuracy_score"),
                "precision_macro": summarize(cls_gcn, "precision_macro"),
                "precision_micro": summarize(cls_gcn, "precision_micro"),
                "precision_weighted": summarize(cls_gcn, "precision_weighted"),
                "recall_macro":    summarize(cls_gcn, "recall_macro"),
                "recall_micro":    summarize(cls_gcn, "recall_micro"),
                "recall_weighted": summarize(cls_gcn, "recall_weighted"),
                "f1_macro":        summarize(cls_gcn, "f1_macro"),
                "f1_micro":        summarize(cls_gcn, "f1_micro"),
                "f1_weighted":     summarize(cls_gcn, "f1_weighted"),
                "rand_score":      clu_gcn.loc["rand_score", "all"],
                "time_s":          t_gcn
            }

            # collect
            all_records.append(mlp_rec)
            all_records.append(gcn_rec)
            ds_records.append(mlp_rec)
            ds_records.append(gcn_rec)

        # write per-dataset CSV
        df_ds = pd.DataFrame(ds_records)
        per_ds_path = f"evaluation_results_{ds}.csv"
        df_ds.to_csv(per_ds_path, index=False)
        print(f"Wrote per-dataset results to {per_ds_path}")

    # ─── aggregate raw records ────────────────────────────────────────────
    df_all = pd.DataFrame(all_records)
    df_all.to_csv("evaluation_results.csv", index=False)
    print("Wrote raw results to evaluation_results.csv")

    # ─── compute mean ± std over seeds ────────────────────────────────────
    metrics = [
        "accuracy",
        "precision_macro","precision_micro","precision_weighted",
        "recall_macro","recall_micro","recall_weighted",
        "f1_macro","f1_micro","f1_weighted",
        "rand_score","time_s"
    ]
    summary = df_all.groupby(["dataset","model"])[metrics].agg(["mean","std"])
    summary.to_csv("evaluation_summary.csv")
    print("Wrote summary to evaluation_summary.csv")
    print(summary)

    # ─── plotting ────────────────────────────────────────────────────────
    summary = pd.read_csv(
        "evaluation_summary.csv",
        index_col=[0,1],
        header=[0,1]
    )
    means = summary.xs("mean", axis=1, level=1)
    stds  = summary.xs("std",  axis=1, level=1)

    ds_list = means.index.get_level_values("dataset").unique()
    model_list = means.index.get_level_values("model").unique()

    for metric in metrics:
        fig, ax = plt.subplots()
        x = list(range(len(ds_list)))
        width = 0.8 / len(model_list)

        for i, model in enumerate(model_list):
            vals = [means.loc[(ds, model), metric] for ds in ds_list]
            errs = [stds.loc[(ds, model), metric] for ds in ds_list]
            positions = [xi + (i - 0.5)*width for xi in x]
            ax.bar(positions, vals, width, yerr=errs, label=model)

        ax.set_xticks(x)
        ax.set_xticklabels(ds_list)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by model & dataset")
        ax.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()