#!/usr/bin/env python3
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from run_cage_gcn_smooth   import run_gcn_with_smoothness as run_geocage
from run_cage_mlp          import run_mlp
# -- you’ll need to implement wrappers for each baseline --
from baselines import (
  run_LAP, run_SDNE, run_node2vec, run_DeepWalk,
  run_KATE, run_Doc2Vec, run_DW_D2V,
  run_TriDNR, run_GAT2Vec
)
def main():
    datasets = ["cora", "citeseer", "mcc", "pubmed", "dblp"]
    seeds    = [220,221,222,223,224,225,226,227,228,229]

    methods = {
        "GeoCAGE":    run_geocage,
        "LAP":        run_LAP,
        "SDNE":       run_SDNE,
        "node2vec":   run_node2vec,
        "DeepWalk":   run_DeepWalk,
        "KATE":       run_KATE,
        "Doc2Vec":    run_Doc2Vec,
        "DW+D2V":     run_DW_D2V,
        "TriDNR":     run_TriDNR,
        "GAT2Vec":    run_GAT2Vec
    }

    all_records = []

    for ds in datasets:
        ds_records = []
        for seed in seeds:
            for name, runner in methods.items():
                print(f"Seed {seed}, Dataset {ds}, Method {name}")
                t0 = time.time()
                cls, clu = runner(ds, seed)
                elapsed = time.time() - t0
                print(f"→ {name} took {elapsed:.1f}s")

                # mean over repeated splits
                def m(mkey): return cls.loc[mkey].mean()
                rec = {
                    "dataset": ds,
                    "method":  name,
                    "seed":    seed,
                    "accuracy":         m("accuracy_score"),
                    "f1_macro":         m("f1_macro"),
                    "rand_score":       clu.loc["rand_score","all"],
                    "time_s":           elapsed
                }
                all_records.append(rec)
                ds_records.append(rec)

        # write per‑dataset
        df_ds = pd.DataFrame(ds_records)
        df_ds.to_csv(f"eval_{ds}.csv", index=False)
        print(f"Wrote eval_{ds}.csv")

    df_all = pd.DataFrame(all_records)
    df_all.to_csv("eval_all.csv", index=False)
    print("Wrote eval_all.csv")

    # summary
    metrics = ["accuracy","f1_macro","rand_score","time_s"]
    summary = df_all.groupby(["dataset","method"])[metrics].agg(["mean","std"])
    summary.to_csv("summary.csv")
    print("Wrote summary.csv\n", summary)

    # quick bar‐plot of accuracy
    fig,ax = plt.subplots(figsize=(8,4))
    for i,method in enumerate(methods):
        vals = [summary.loc[(ds,method),("accuracy","mean")] for ds in datasets]
        errs = [summary.loc[(ds,method),("accuracy","std")]  for ds in datasets]
        x = list(range(len(datasets)))
        ax.bar([xi + i*0.08 for xi in x], vals, width=0.08, yerr=errs, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Accuracy")
    ax.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
