# Human-Centered ML Active Learning

This repo shows our HCML AL experiments.


## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows, use `venv\Scripts\activate`

## Data Format

The project expects data in JSONL format (one JSON object per line). Each line must have two fields:

- `text`: the input text to classify
- `label`: an integer label starting from 0

Place your data files in `data/<dataset_name>/` with the following structure:

```
data/
  dataset_name/
    train.jsonl
    validation.jsonl
    test.jsonl
```

### Binary Classification Example

For binary tasks use labels 0 and 1:

```jsonl
{"text": "This is a positive example", "label": 1}
{"text": "This is a negative example", "label": 0}
```

### Multi-class Classification Example

For multi-class tasks, use integer labels starting from 0:

```jsonl
{"text": "Breaking news about politics", "label": 0}
{"text": "The team won the championship", "label": 1}
{"text": "Stock market reaches new high", "label": 2}
```

## Datasets

- [aegis](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0)
- [ag_news](https://huggingface.co/datasets/ag_news)
- [hate_speech_offensive](https://huggingface.co/datasets/hate_speech_offensive)

## Configuration

Edit `config.py` to set up experiment variables.

## How to run

```bash
python training.py
```

Results (graph + csv) are saved to the `results` folder.
