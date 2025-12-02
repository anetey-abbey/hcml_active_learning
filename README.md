# Human-Centered ML Active Learning

This repo shows our HCML AL experiments.

## Dataset

**gametox_merged**: ([gametox](https://raw.githubusercontent.com/shucoll/GameTox/main/gametox.csv)) This is a modified version of the gametox dataset, with the only change being that the classes "hate", "threats", and "extremism" have been merged into a single class. This was done because these individual classes had too low of a sample size for effective model training and evaluation in the active learning setting.

#### Original GameTox (6 classes)

| Label ID | Class Name | Count | Percentage |
|----------|------------|-------|------------|
| 0 | NON_TOXIC | 43,497 | 81.0% |
| 1 | INSULTS_FLAMING | 7,407 | 13.8% |
| 2 | OTHER_OFFENSIVE | 2,343 | 4.4% |
| 3 | HATE_HARASSMENT | 349 | 0.6% |
| 4 | THREATS | 75 | 0.1% |
| 5 | EXTREMISM | 30 | 0.1% |
| **Total** | | **53,701** | **100%** |

#### Merged GameTox (4 classes)

| Label ID | Class Name | Count | Percentage |
|----------|------------|-------|------------|
| 0 | NON_TOXIC | 43,497 | 81.0% |
| 1 | INSULTS_FLAMING | 7,407 | 13.8% |
| 2 | OTHER_OFFENSIVE | 2,343 | 4.4% |
| 3 | HATE_THREATS_EXTREMISM | 454 | 0.8% |
| **Total** | | **53,701** | **100%** |


## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows, use `venv\Scripts\activate`

## Configuration

Edit `config.py` to set up experiment variables.

## How to run

```bash
python training.py
```

Results (graphs + csv) are saved to the `results` folder.
