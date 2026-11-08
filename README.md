# ECG diagnostic superclass classification on PTB-XL

Code for the paper "Multi-Model Comparison of Deep and Classical Architectures for ECG Diagnostic Superclass Classification on PTB-XL".

- `notebooks/01_train_models_v2.ipynb`: model training (M1-M4) and the M5 ensemble.
- `notebooks/02_revision_analysis.ipynb`: M6 (MiniRocket + LightGBM), evaluation, bootstrap analysis, and the paper's tables and figures.

Both notebooks run on Google Colab with Google Drive mounted and read and write a Drive folder named `results_v3`. Run 01, then 02, each top to bottom.

Data: PTB-XL v1.0.3, <https://physionet.org/content/ptb-xl/1.0.3/> (not included).

## Reproducibility files
- patient_ids_12lead.npy, record_ids_raw_12lead.npy: patient and record order used by both notebooks
- train_idx.npy, val_idx.npy, test_idx.npy: split indices (patient-level 70/15/15, seed 42)
- y_raw_12lead.npy, label_rule.json: the re-derived single-label ground truth and the rule that produced it
- requirements_used.txt: library versions in the analysis runtime (Colab CPU); the deep models were trained on a T4 GPU runtime whose versions were not recorded

Seed 42 is used for the patient split, model training, the bootstrap and SMOTE.
