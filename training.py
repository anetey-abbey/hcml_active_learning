import argparse
import warnings
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as transformers_logging
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

from dataset import load_data

from utils import (
    average_results, 
    plot_comparison, 
    print_comparison, 
    set_seed, 
    save_experiment_results,
)

import config


def parse_args():
    parser = argparse.ArgumentParser(description="Active learning training script")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--num-runs", type=int, default=config.NUM_RUNS)
    parser.add_argument("--dataset", type=str, default=config.DATASET)
    parser.add_argument("--model-name", type=str, default=config.MODEL_NAME)
    parser.add_argument("--num-labels", type=int, default=config.NUM_LABELS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--pool-size", type=int, default=config.POOL_SIZE)
    parser.add_argument("--initial-labeled", type=int, default=config.INITIAL_LABELED)
    parser.add_argument("--num-iterations", type=int, default=config.NUM_ITERATIONS)
    parser.add_argument("--samples-per-iteration", type=int, default=config.SAMPLES_PER_ITERATION)
    parser.add_argument("--annotator-noise", type=float, default=config.ANNOTATOR_NOISE)
    parser.add_argument("--uncertainty-sampling-strategy", type=str, default=config.UNCERTAINTY_STRATEGY)
    return parser.parse_args()

def compute_all_metrics(y_true, y_pred, num_classes):
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(num_classes))
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    for i in range(num_classes):
        metrics[f"f1_class_{i}"] = per_class_f1[i]
    return metrics


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, num_classes):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return compute_all_metrics(true_labels, predictions, num_classes)


def train_model(
    train_dataset,
    val_loader,
    test_loader,
    device,
    model_name,
    num_labels,
    batch_size,
    learning_rate,
    epochs,
    model=None,
    seed=None,
):
    if model is None and seed is not None:
        set_seed(seed)

    model = model or AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for _ in range(epochs):
        train_epoch(model, train_loader, optimizer, device)

    val_metrics = evaluate(model, val_loader, device, num_labels)
    test_metrics = evaluate(model, test_loader, device, num_labels)
    return model, val_metrics, test_metrics


def main():
    from sampling import active_learning_loop

    args = parse_args()

    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("\n")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    seeds = [args.seed + i for i in range(args.num_runs)]

    train_data = load_data(f"data/{args.dataset}/train.jsonl")
    val_data = load_data(f"data/{args.dataset}/validation.jsonl")
    test_data = load_data(f"data/{args.dataset}/test.jsonl")

    print(f"Data lengths: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    random_runs = []
    uncertainty_runs = []

    # run experiment for multiple seeds
    for idx, seed in enumerate(seeds):
        print(f"\nRun {idx + 1} out of {args.num_runs}")
        set_seed(seed)
        rng = np.random.default_rng(seed)
        pool_indices = rng.choice(len(train_data), size=args.pool_size, replace=False)
        pool_data = [train_data[i] for i in pool_indices]
        initial_indices = rng.choice(len(pool_data), args.initial_labeled, replace=False)

        print(f"\nSeed {seed}: RANDOM SAMPLING (Baseline)")
        random_results = active_learning_loop(
            pool_data,
            val_data,
            test_data,
            tokenizer,
            device,
            strategy="random",
            initial_indices=initial_indices,
            rng_seed=seed + 1,
            seed=seed,
            model_name=args.model_name,
            num_labels=args.num_labels,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            num_iterations=args.num_iterations,
            samples_per_iteration=args.samples_per_iteration,
            annotator_noise=args.annotator_noise,
            uncertainty_strategy=args.uncertainty_sampling_strategy,
        )

        print(f"Seed {seed}: UNCERTAINTY SAMPLING (Active Learning)")
        uncertainty_results = active_learning_loop(
            pool_data,
            val_data,
            test_data,
            tokenizer,
            device,
            strategy="uncertainty",
            initial_indices=initial_indices,
            rng_seed=seed + 2,
            seed=seed,
            model_name=args.model_name,
            num_labels=args.num_labels,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            num_iterations=args.num_iterations,
            samples_per_iteration=args.samples_per_iteration,
            annotator_noise=args.annotator_noise,
            uncertainty_strategy=args.uncertainty_sampling_strategy,
        )

        print_comparison(random_results, uncertainty_results, seed=seed)
        
        random_runs.append(random_results)
        uncertainty_runs.append(uncertainty_results)

    config_dict = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "initial_labeled": args.initial_labeled,
        "samples_per_iteration": args.samples_per_iteration,
        "num_iterations": args.num_iterations,
        "pool_size": args.pool_size,
        "annotator_noise": args.annotator_noise,
        "uncertainty_strategy": args.uncertainty_sampling_strategy,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_experiment_results(
        random_runs,
        uncertainty_runs,
        seeds,
        config_dict,
        timestamp,
    )
    plot_comparison(
        random_runs,
        uncertainty_runs,
        seeds,
        config_dict,
        timestamp,
    )

    # show final average result
    if len(seeds) > 1:
        mean_random = average_results(random_runs)
        mean_uncertainty = average_results(uncertainty_runs)
        print(f"\nFinal results mean for {args.num_runs} seeds")
        print_comparison(mean_random, mean_uncertainty, seed="mean")


if __name__ == "__main__":
    main()
