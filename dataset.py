import json
import torch
from torch.utils.data import Dataset

from config import NUM_LABELS


class CustomDataset(Dataset):
    "Tokenize text and return labels"

    def __init__(self, data, tokenizer, max_length=128, noise_rate=0.0, rng=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noisy_labels = {}

        # change labels to simulate mistakes made by data annotators using noise_rate
        # Note: real annotator noise is often non-symmetric (e.g. worse for some classes)
        if noise_rate > 0 and rng is not None:
            for idx in range(len(data)):
                if rng.random() < noise_rate:
                    true_label = data[idx]["label"]
                    wrong_labels = [l for l in range(NUM_LABELS) if l != true_label]
                    self.noisy_labels[idx] = rng.choice(wrong_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(
                self.noisy_labels.get(idx, item["label"]), dtype=torch.long
            ),
        }


def load_data(file_path, n_samples=None):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
            if n_samples and len(data) >= n_samples:
                break
    return data
