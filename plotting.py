import pandas as pd
import matplotlib.pyplot as plt


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