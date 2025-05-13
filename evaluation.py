#!/usr/bin/env python3
import os
import pandas as pd

# import your two functions
from run_cage_mlp import run_mlp
from run_cage_gcn import run_gcn

def main():
    datasets = ["cora", "citeseer", "mcc"]
    seeds    = [220, 221, 222, 223, 224]

    records = []
    for ds in datasets:
        for seed in seeds:
            print(f"Seed: {seed}, Dataset: {ds}, MLP")
            cls_mlp, clu_mlp = run_mlp(ds, seed)
            print(f"Seed: {seed}, Dataset: {ds}, GCN")
            cls_gcn, clu_gcn = run_gcn(ds, seed)

            # classification
            acc_mlp  = cls_mlp.loc["accuracy_score"].mean()
            f1m_mlp  = cls_mlp.loc["f1_macro"].mean()
            f1w_mlp  = cls_mlp.loc["f1_weighted"].mean()

            acc_gcn  = cls_gcn.loc["accuracy_score"].mean()
            f1m_gcn  = cls_gcn.loc["f1_macro"].mean()
            f1w_gcn  = cls_gcn.loc["f1_weighted"].mean()

            # clustering
            rand_mlp = clu_mlp.loc["rand_score", "all"]
            rand_gcn = clu_gcn.loc["rand_score", "all"]

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
                "model":   "GCN",
                "seed":    seed,
                "accuracy":     acc_gcn,
                "f1_macro":     f1m_gcn,
                "f1_weighted":  f1w_gcn,
                "rand_score":   rand_gcn
            })

    df = pd.DataFrame(records)
    df.to_csv("evaluation_results.csv", index=False)
    print("Wrote raw results to evaluation_results.csv")

    # mean ± std over seeds for each (dataset, model)
    summary = df.groupby(["dataset","model"])[["accuracy","f1_macro","f1_weighted","rand_score"]].agg(["mean","std"])
    summary.to_csv("evaluation_summary.csv")
    print("Wrote summary to evaluation_summary.csv")
    print(summary)

    summary = pd.read_csv("evaluation_summary.csv", index_col=[0,1], header=[0,1])

    # split out means & stds
    means = summary.xs("mean", axis=1, level=1)
    stds  = summary.xs("std",  axis=1, level=1)

    datasets = means.index.get_level_values("dataset").unique()
    models   = means.index.get_level_values("model").unique()

    # for each metric, draw one bar chart
    for metric in ["accuracy","f1_macro","f1_weighted","rand_score"]:
        fig, ax = plt.subplots()
        x = list(range(len(datasets)))
        width = 0.35

        for i, model in enumerate(models):
            # extract the mean & std for this model across datasets
            vals  = [means.loc[(ds, model), metric] for ds in datasets]
            errs  = [stds.loc[(ds, model), metric] for ds in datasets]
            positions = [xi + (i - 0.5)*(width) for xi in x]
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
