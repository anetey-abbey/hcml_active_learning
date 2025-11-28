import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import CustomDataset
from training import train_model


def get_uncertainty_scores(model, dataloader, device):
    """
    Margin sampling: smaller margin between top-2 class probs = higher uncertainty.
    """
    model.eval()
    margins = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)

            # Margin based on difference between top 2 class probabilities (lower margin means more uncertain)
            top2 = torch.topk(probs, k=2, dim=1).values
            margin = top2[:, 0] - top2[:, 1]
            margins.extend(margin.cpu().numpy())

    return np.array(margins)


def active_learning_loop(
    pool_data,
    val_data,
    test_data,
    tokenizer,
    device,
    strategy="uncertainty",
    initial_indices=None,
    rng_seed=None,
    seed=None,
    model_name=config.MODEL_NAME,
    num_labels=config.NUM_LABELS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    epochs=config.EPOCHS,
    num_iterations=config.NUM_ITERATIONS,
    samples_per_iteration=config.SAMPLES_PER_ITERATION,
    annotator_noise=config.ANNOTATOR_NOISE,
):
    rng = np.random.default_rng(rng_seed)
    if initial_indices is not None:
        labeled_indices = np.array(initial_indices)
    else:
        labeled_indices = rng.choice(len(pool_data), config.INITIAL_LABELED, replace=False)
    unlabeled_indices = np.array([i for i in range(len(pool_data)) if i not in labeled_indices])
    val_loader = DataLoader(CustomDataset(val_data, tokenizer), batch_size=batch_size)
    test_loader = DataLoader(CustomDataset(test_data, tokenizer), batch_size=batch_size)

    results = []
    model = None

    # main active learning loop (train model and get new samples iteratively)
    pbar = tqdm(range(num_iterations + 1), desc=f"{strategy.upper()}", unit="iter")
    for iteration in pbar:
        labeled_data = [pool_data[i] for i in labeled_indices]
        train_dataset = CustomDataset(
            labeled_data,
            tokenizer,
            noise_rate=annotator_noise,
            rng=rng,
        )

        pbar.set_postfix({"samples": len(labeled_data)})

        model, val_f1, test_f1 = train_model(
            train_dataset,
            val_loader,
            test_loader,
            device,
            model_name=model_name,
            num_labels=num_labels,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs,
            model=model,
            seed=seed,
        )

        results.append(
            {
                "iteration": iteration,
                "labeled_count": len(labeled_data),
                "val_f1": val_f1,
                "test_f1": test_f1,
            }
        )

        # Select new samples to label for the next iteration
        if iteration < num_iterations and len(unlabeled_indices) > 0:
            if strategy == "uncertainty":
                unlabeled_data = [pool_data[i] for i in unlabeled_indices]
                unlabeled_loader = DataLoader(
                    CustomDataset(unlabeled_data, tokenizer),
                    batch_size=batch_size,
                )
                margins = get_uncertainty_scores(model, unlabeled_loader, device)
                selected_unlabeled_indices = np.argsort(margins)[:samples_per_iteration]
                selected_pool_indices = unlabeled_indices[selected_unlabeled_indices]

            else:
                selected_pool_indices = rng.choice(
                    unlabeled_indices,
                    min(samples_per_iteration, len(unlabeled_indices)),
                    replace=False,
                )

            labeled_indices = np.concatenate([labeled_indices, selected_pool_indices])
            unlabeled_indices = np.array([i for i in unlabeled_indices if i not in selected_pool_indices])

    return results
