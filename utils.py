import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import pandas as pd
import torch
from scipy import stats

import config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_comparison_df(random_results, uncertainty_results, metric="test_weighted_f1"):
    return pd.DataFrame(
        {
            "Labeled Samples": [r["labeled_count"] for r in random_results],
            "Random Sampling": [r.get(metric, r.get("test_f1", 0)) for r in random_results],
            "Uncertainty Sampling": [u.get(metric, u.get("test_f1", 0)) for u in uncertainty_results],
            "Difference": [
                u.get(metric, u.get("test_f1", 0)) - r.get(metric, r.get("test_f1", 0))
                for r, u in zip(random_results, uncertainty_results)
            ],
        }
    )


def average_results(runs):
    if not runs:
        return []
    df = pd.concat([pd.DataFrame(run) for run in runs])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    averaged = (
        df.groupby("labeled_count", as_index=False)[numeric_cols.tolist()]
        .mean()
        .sort_values("labeled_count")
        .to_dict(orient="records")
    )
    return averaged


def print_comparison(random_results, uncertainty_results, seed=None):
    if seed is not None:
        print(f"\nSeed {seed}")
    comparison_df = get_comparison_df(random_results, uncertainty_results)
    print(comparison_df.to_string(index=False))


def save_experiment_results(random_runs, uncertainty_runs, seeds, config_dict, timestamp, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for seed_idx, seed in enumerate(seeds):
        for r, u in zip(random_runs[seed_idx], uncertainty_runs[seed_idx]):
            row = {"seed": seed, "labeled_count": r["labeled_count"]}
            for key, value in r.items():
                if key not in ["iteration", "labeled_count"] and isinstance(value, (int, float)):
                    row[f"random_{key}"] = value
            for key, value in u.items():
                if key not in ["iteration", "labeled_count"] and isinstance(value, (int, float)):
                    row[f"uncertainty_{key}"] = value
            row.update(config_dict)
            rows.append(row)
    csv_path = os.path.join(output_dir, f"experiment_{timestamp}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return csv_path


def plot_comparison(random_runs, uncertainty_runs, seeds, config_dict, timestamp, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for seed_idx, seed in enumerate(seeds):
        for r, u in zip(random_runs[seed_idx], uncertainty_runs[seed_idx]):
            row = {"seed": seed, "labeled_count": r["labeled_count"]}
            for key, value in r.items():
                if isinstance(value, (int, float)):
                    row[f"random_{key}"] = value
            for key, value in u.items():
                if isinstance(value, (int, float)):
                    row[f"uncertainty_{key}"] = value
            rows.append(row)
    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("test_weighted_f1", "Weighted F1"),
        ("test_macro_f1", "Macro F1"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        r_col = f"random_{metric}"
        u_col = f"uncertainty_{metric}"
        if r_col in df.columns and u_col in df.columns:
            sns.lineplot(data=df, x="labeled_count", y=r_col, label="Random", marker="o", errorbar=("ci", 95), ax=ax)
            sns.lineplot(data=df, x="labeled_count", y=u_col, label="Uncertainty", marker="o", errorbar=("ci", 95), ax=ax)
        ax.set_xlabel("Labeled Samples")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(loc="lower right")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"experiment_{timestamp}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nPlot saved to: {plot_path}")

    plot_per_class_learning(random_runs, uncertainty_runs, config_dict, timestamp, output_dir)


def plot_per_class_learning(random_runs, uncertainty_runs, config_dict, timestamp, output_dir="results"):
    num_classes = config_dict.get("num_labels", config.NUM_LABELS)
    dataset = config_dict.get("dataset", config.DATASET)

    # hard coded solution for our gametox merged verions
    if dataset == "gametox":
        class_names = {0: "NON_TOXIC", 1: "INSULTS", 2: "OTHER_OFFENSIVE", 3: "HATE", 4: "THREATS", 5: "EXTREMISM"}
    elif dataset == "gametox_merged":
        class_names = {0: "NON_TOXIC", 1: "INSULTS", 2: "OTHER_OFFENSIVE", 3: "HATE/THREATS/EXTREMISM"}
    else:
        class_names = {i: f"Class {i}" for i in range(num_classes)}

    rows = []
    for run_idx, (r_run, u_run) in enumerate(zip(random_runs, uncertainty_runs)):
        for r, u in zip(r_run, u_run):
            for cls_idx in range(num_classes):
                key = f"test_f1_class_{cls_idx}"
                rows.append({"run": run_idx, "labeled_count": r["labeled_count"], "class": class_names.get(cls_idx, f"Class {cls_idx}"), "class_idx": cls_idx, "Random": r.get(key, 0), "Uncertainty": u.get(key, 0)})

    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    ncols = 2
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    num_plots = min(num_classes, nrows * ncols)
    for cls_idx in range(num_plots):
        ax = axes[cls_idx]
        cls_df = df[df["class_idx"] == cls_idx]
        if not cls_df.empty and (cls_df["Random"].sum() > 0 or cls_df["Uncertainty"].sum() > 0):
            sns.lineplot(data=cls_df, x="labeled_count", y="Random", label="Random", marker="o", errorbar=("ci", 95), ax=ax)
            sns.lineplot(data=cls_df, x="labeled_count", y="Uncertainty", label="Uncertainty", marker="o", errorbar=("ci", 95), ax=ax)
        ax.set_title(class_names.get(cls_idx, f"Class {cls_idx}"))
        ax.set_xlabel("Labeled Samples")
        ax.set_ylabel("F1 Score")
        ax.legend(loc="lower right", fontsize=8)

    plt.suptitle("GameTox - Random Sampling vs Uncertainty Sampling", fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"per_class_{timestamp}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Per-class plot saved to: {plot_path}")
