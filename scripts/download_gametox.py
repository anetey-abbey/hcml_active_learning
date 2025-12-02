import json
import os
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from io import StringIO

OUTPUT_DIR = "data/gametox"
CSV_URL = "https://raw.githubusercontent.com/shucoll/GameTox/main/gametox.csv"

# Our code expects integers as class labels
LABEL_MAP = {
    0: "NON_TOXIC",
    1: "INSULTS_FLAMING",
    2: "OTHER_OFFENSIVE",
    3: "HATE_HARASSMENT",
    4: "THREATS",
    5: "EXTREMISM"
}


def convert_and_save(df, output_path):
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            label = int(row["label"])
            record = {
                "text_label": LABEL_MAP[label],
                "text": row["message"],
                "label": label
            }
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(df)} samples to {output_path}")


def main():
    csv_content = requests.get(CSV_URL).text
    
    df = pd.read_csv(StringIO(csv_content))
    print(f"Total samples: {len(df)}")

    # Stratified split: 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    convert_and_save(train_df, os.path.join(OUTPUT_DIR, "train.jsonl"))
    convert_and_save(val_df, os.path.join(OUTPUT_DIR, "validation.jsonl"))
    convert_and_save(test_df, os.path.join(OUTPUT_DIR, "test.jsonl"))

    print(f"\nDataset saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
