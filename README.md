# CS-184A-FINAL: Multimodal Transformers for Schizophrenia Classification

CS 184A (UC Irvine, Fall 2026) final project. We classify **schizophrenia vs. healthy controls** using three kinds of data from the [COBRE](https://coins.trendscenter.org/) and BrainGluSchi studies:

- **Cognitive tests:** a neuropsychological battery
- **Brain structure:** T1 MRI, summarized as region volumes and thickness
- **Brain function:** resting-state fMRI, summarized as region-to-region connectivity

The core question is **what brain scans add beyond cognitive tests**. We compare Transformer models (FT-Transformer, Brain Network Transformer and a multimodal Transformer) against logistic regression and XGBoost. Every model is evaluated with cross-validation, confidence intervals and confound checks.

The plan is in the proposal doc

## Team

| Member             | Owns                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------- |
| Gilberto Fernandez | Cognitive-test pipeline, evaluation and statistics (CV, bootstrap, permutation tests) |
| Daniel Chu         | Resting-state fMRI pipeline, model training                                           |
| Alex Gonzalez      | T1 structural pipeline, model setup, early-psychosis test and interpretation          |

## Project structure

Folders marked _(planned)_ are created when their first code lands.

```
CS-184A-FINAL/
├── README.md                 # this file
├── DATA_GUIDE.md             # what every data file is, and which ones we use
├── data_inventory.csv        # one row per data file: counts, purpose, used or not
├── .gitignore                # keeps all subject data out of Git
├── requirements.txt          # (planned) Python 3.11/3.12 dependencies
│
├── COBRE data/               # raw clinical CSVs           ┐
├── COBRE MR/                 # raw MRI zips (~127 GB)      │ local only, git-ignored,
├── BrainGluSchi data/        # raw clinical CSVs           │ covered by the COINS DUA
├── BrainGluSchi MR/          # raw MRI zips (~58 GB)       ┘
│
├── data/                     # (planned) git-ignored; shared via private Drive
│   ├── interim/              #   extracted NIfTI scans (T1, resting fMRI)
│   └── features/             #   labels.csv, cognition.csv, structure.csv, connectivity/
│
├── src/                      # (planned) all reusable code
│   ├── data/                 #   load clinical CSVs, build labels, make CV splits
│   ├── preprocessing/        #   T1 (FastSurfer), resting fMRI (nilearn), site harmonization (ComBat)
│   ├── models/               #   baselines, FT-Transformer, Brain Network Transformer, fusion
│   └── evaluation/           #   CV runner, bootstrap CIs, permutation tests, confound checks
│
├── configs/                  # (planned) one YAML file per experiment
├── notebooks/                # (planned) exploration only; nothing others depend on
├── results/                  # (planned) committed: metric tables and figures, no subject IDs
└── reports/                  # (planned) proposal and final report
```

## Data

- **The raw data is never committed or shared publicly.** It falls under the COINS data use agreement, so every team member needs to be on that agreement.
- **Processing happens on one machine.** The raw MRI stays on that machine, which has the GPU. Only the small feature tables in `data/features/` are shared, through a private Drive folder.
- **See [`DATA_GUIDE.md`](DATA_GUIDE.md)** for what each file contains, the label mapping, and which forms must never be used as inputs (label leakage).

## Setup

You need **Python 3.11 or 3.12**. Newer versions aren't supported by every package yet.

```bash
# Windows (if you don't have 3.12: `py install 3.12`)
py -V:3.12 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt     # ~3 GB on Windows/Linux (PyTorch with CUDA)
python src/check_env.py             # should end with "All good."
```

- **GPU:** `requirements.txt` installs PyTorch with CUDA 13.2, which needs a recent NVIDIA driver. With an older driver, see the comment at the top of `requirements.txt`. Without a GPU, the tabular models still run fine on the CPU.
- **VS Code:** choose `.venv` as the Python interpreter (Ctrl+Shift+P → *Python: Select Interpreter*).
- **Data:** download the shared feature files into `data/features/` (see [Data](#data)).

## Pipeline

```
raw zips ──► extract (T1 + rest only) ──► preprocess ──► feature tables ──► models ──► evaluation
                                           T1: FastSurfer        cognition      Stage 1: one model per data type
                                           rest: nilearn         structure      Stage 2: late fusion (average)
                                           ComBat harmonize      connectivity   Stage 3: multimodal Transformer
```

The run commands for each step will be added as the code is written.

## Conventions

- **Work on a branch per feature** and open a pull request into `main`.
- **Never commit subject-level data.** That covers IDs, per-subject rows and scans. Only aggregate results go in `results/`.
- **Put shared logic in `src/`.** Notebooks import from it and don't define logic of their own.
- **Fix random seeds, and give every experiment its own file in `configs/`**, so results can be reproduced.
