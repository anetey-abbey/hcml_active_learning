import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import numpy as np
import pandas as pd
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_comparison_df(random_results, uncertainty_results):
    return pd.DataFrame(
        {
            "Labeled Samples": [r["labeled_count"] for r in random_results],
            "Random Sampling": [r["test_f1"] for r in random_results],
            "Uncertainty Sampling": [u["test_f1"] for u in uncertainty_results],
            "Difference": [
                u["test_f1"] - r["test_f1"]
                for r, u in zip(random_results, uncertainty_results)
            ],
        }
    )


def average_results(runs):
    if not runs:
        return []
    df = pd.concat([pd.DataFrame(run) for run in runs])
    averaged = (
        df.groupby("labeled_count", as_index=False)
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


def save_experiment_results(
    random_runs,
    uncertainty_runs,
    seeds,
    config_dict,
    timestamp,
    output_dir="results",
):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for seed_idx, seed in enumerate(seeds):
        for r, u in zip(random_runs[seed_idx], uncertainty_runs[seed_idx]):
            row = {
                "seed": seed,
                "labeled_count": r["labeled_count"],
                "random_val_f1": r["val_f1"],
                "random_test_f1": r["test_f1"],
                "uncertainty_val_f1": u["val_f1"],
                "uncertainty_test_f1": u["test_f1"],
            }
            row.update(config_dict)
            rows.append(row)

    csv_path = os.path.join(output_dir, f"experiment_{timestamp}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    return csv_path


def plot_comparison(
    random_runs,
    uncertainty_runs,
    seeds,
    config_dict,
    timestamp,
    output_dir="results",
):
    rows = []
    for seed_idx, seed in enumerate(seeds):
        for r, u in zip(random_runs[seed_idx], uncertainty_runs[seed_idx]):
            rows.append(
                {
                    "seed": seed,
                    "labeled_count": r["labeled_count"],
                    "random_test_f1": r["test_f1"],
                    "uncertainty_test_f1": u["test_f1"],
                }
            )

    df = pd.DataFrame(rows)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.lineplot(
        data=df,
        x="labeled_count",
        y="random_test_f1",
        label="Random Sampling",
        marker="o",
        errorbar="sd",
    )
    sns.lineplot(
        data=df,
        x="labeled_count",
        y="uncertainty_test_f1",
        label="Uncertainty Sampling",
        marker="o",
        errorbar="sd",
    )
    plt.xlabel("Number of Labeled Samples")
    plt.ylabel("Test F1 Score")
    plt.title("Active Learning Performance")
    plt.legend(loc="center right")

    config_text = "\n".join([f"{k}: {v}" for k, v in config_dict.items()])
    plt.text(
        x=1,
        y=0,
        s=config_text,
        transform=plt.gca().transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=9,
        bbox=dict(facecolor="white"),
    )

    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"experiment_{timestamp}.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nPlot saved to: {plot_path}")
