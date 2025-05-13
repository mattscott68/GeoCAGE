#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

# import your two functions
from run_cage_mlp import run_mlp
from run_cage_gcn_smooth import run_gcn_with_smoothness as run_gcn

def main():
    datasets = ["cora", "citeseer", "mcc"]
    seeds    = [220, 221, 222, 223, 224]

    records = []
    for ds in datasets:
        for seed in seeds:
            # MLP baseline
            print(f"Seed: {seed}, Dataset: {ds}, MLP")
            cls_mlp, clu_mlp = run_mlp(ds, seed)

            # GCN + Smooth experiment
            print(f"Seed: {seed}, Dataset: {ds}, GCN+Smooth")
            cls_gcn, clu_gcn = run_gcn(ds, seed)

            # print detailed splits for smooth model
            print("=== GCN+Smooth Classification ===")
            print(cls_gcn)
            print("\n=== GCN+Smooth Clustering (rand_score) ===")
            print(clu_gcn.loc[['rand_score']])

            # classification metrics
            acc_mlp  = cls_mlp.loc["accuracy_score"].mean()
            f1m_mlp  = cls_mlp.loc["f1_macro"].mean()
            f1w_mlp  = cls_mlp.loc["f1_weighted"].mean()

            acc_gcn  = cls_gcn.loc["accuracy_score"].mean()
            f1m_gcn  = cls_gcn.loc["f1_macro"].mean()
            f1w_gcn  = cls_gcn.loc["f1_weighted"].mean()

            # clustering metrics
            rand_mlp = clu_mlp.loc["rand_score", "all"]
            rand_gcn = clu_gcn.loc["rand_score", "all"]

            # record results
            records.append({
                "dataset": ds,
                "model":   "MLP",
                "seed":    seed,
                "accuracy":     acc_mlp,
                "f1_macro":     f1m_mlp,
                "f1_weighted":  f1w_mlp,
                "rand_score":   rand_mlp
            })
            records.append({
                "dataset": ds,
                "model":   "GCN+Smooth",
                "seed":    seed,
                "accuracy":     acc_gcn,
                "f1_macro":     f1m_gcn,
                "f1_weighted":  f1w_gcn,
                "rand_score":   rand_gcn
            })

    # aggregate raw records
    df = pd.DataFrame(records)
    df.to_csv("evaluation_results.csv", index=False)
    print("Wrote raw results to evaluation_results.csv")

    # compute mean ± std over seeds
    summary = df.groupby(["dataset","model"]
            )[["accuracy","f1_macro","f1_weighted","rand_score"]]
        .agg(["mean","std"])
    summary.to_csv("evaluation_summary.csv")
    print("Wrote summary to evaluation_summary.csv")
    print(summary)

    # reload for plotting
    summary = pd.read_csv("evaluation_summary.csv", index_col=[0,1], header=[0,1])
    means = summary.xs("mean", axis=1, level=1)
    stds  = summary.xs("std",  axis=1, level=1)

    datasets = means.index.get_level_values("dataset").unique()
    models   = means.index.get_level_values("model").unique()

    # bar charts per metric
    for metric in ["accuracy","f1_macro","f1_weighted","rand_score"]:
        fig, ax = plt.subplots()
        x = list(range(len(datasets)))
        width = 0.35

        for i, model in enumerate(models):
            vals  = [means.loc[(ds, model), metric] for ds in datasets]
            errs  = [stds.loc[(ds, model), metric] for ds in datasets]
            positions = [xi + (i - 0.5)*width for xi in x]
            ax.bar(positions, vals, width, yerr=errs, label=model)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by model & dataset")
        ax.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
