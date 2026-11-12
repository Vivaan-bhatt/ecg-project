# ECG diagnostic superclass classification on PTB-XL

Code for the paper "Multi-Model Comparison of Deep and Classical Architectures for ECG Diagnostic Superclass Classification on PTB-XL".

## Contents

- `notebooks/01_train_models_v2.ipynb`: model training (M1-M4) and the original M2+M3+M4 ensemble.
- `notebooks/02_revision_analysis.ipynb`: M6 (MiniRocket + LightGBM), evaluation, bootstrap analysis, and most of the paper's tables and figures.
- `analysis_ensembles.py`: recomputes the ensemble results used in the paper (see below).

Both notebooks run on Google Colab with Google Drive mounted and read and write a Drive folder named `results_v3`. Run 01, then 02, each top to bottom.

Data: PTB-XL v1.0.3, https://physionet.org/content/ptb-xl/1.0.3/ (not included).

## Model and ensemble naming used in the paper

| Name | Definition |
|---|---|
| M1 | Mamba-style SSM (pure PyTorch) |
| M2 | 1D residual CNN |
| M3 | BiLSTM |
| M4 | CRNN with attention |
| M5 | Soft-voting ensemble of M1, M3, M4 (the three deep models with the highest validation macro F1). **Primary ensemble.** |
| M5A | Soft-voting ensemble of M1, M2, M3, M4 (no selection) |
| M5T | Soft-voting ensemble of M2, M3, M4 (members chosen on test results; **exploratory**) |
| M6 | MiniRocket + LightGBM, without SMOTE |
| M6 (SMOTE) | MiniRocket + LightGBM, with SMOTE (ablation) |

**Label warning for older outputs.** In the `summary`, `pairwise` and `per_class` CSVs written by earlier notebook runs, "M5" is the all-four ensemble (M5A, not M1+M3+M4), "M6" is the SMOTE variant, and "M6_noSMOTE" is the paper's M6. Relabel these before reuse. The probability files `probs_test_M5.npy` / `probs_val_M5.npy` from the first export hold the M2+M3+M4 average (M5T).

## Ensemble analysis script

`analysis_ensembles.py` recomputes test accuracy, macro F1 with patient-level bootstrap CIs (10,000 resamples, seed 42), macro AUROC and AUPRC, ECE (15 bins), Brier score, log loss, paired macro-F1 differences with 95% and 90% intervals, and false-NORM counts.

Usage:

```
python analysis_ensembles.py DIR
```

`DIR` must contain:

- `probs_test_M1.npy`, `probs_test_M2.npy`, `probs_test_M3.npy`, `probs_test_M4.npy`
- `probs_test_M6_noSMOTE.npy` (used as M6) and `probs_test_M6.npy` (the SMOTE version)
- `y_test.npy` (test labels, same record order as the probability files)
- `test_idx.npy` and `patient_ids_12lead.npy` (to map test records to patients for the bootstrap)

It builds M5, M5A and M5T by averaging the saved softmax outputs. Expected headline results: M5 macro F1 0.656, M5A 0.657, M5T 0.665, M6 0.645; false-NORM counts 179 (M5), 215 (M5T), 273 (M6).

## Reproducibility files

- `patient_ids_12lead.npy`, `record_ids_raw_12lead.npy`: patient and record order used by both notebooks.
- `train_idx.npy`, `val_idx.npy`, `test_idx.npy`: split indices (patient-level 70/15/15, seed 42).
- `y_raw_12lead.npy`, `label_rule.json`: the re-derived single-label ground truth and the rule that produced it. `label_rule.json` records `count_score: 11`; the labels are an approximate re-derivation (test counts within 2 per class of the original analysis; HYP total 5 below it).
- `requirements_used.txt`: library versions in the analysis runtime (Colab CPU); the deep models were trained on a T4 GPU runtime whose versions were not recorded.

Seed 42 is used for the patient split, model training, the bootstrap and SMOTE.

## License

Code is released under the MIT License (see `LICENSE`). PTB-XL is distributed by PhysioNet under its own license (Creative Commons Attribution 4.0 at the time of writing; check the dataset page); the index, label and probability arrays in this repository are derived from it, so please cite PTB-XL (Wagner et al., Scientific Data 2020) when reusing them.
