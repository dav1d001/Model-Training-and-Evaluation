# Credit-Card Default Prediction: Traditional ML vs Deep Learning

ALU Machine Learning summative project. Binary classification of credit-card default on the UCI
*Default of Credit Card Clients* dataset (30,000 rows, 23 predictors, 22.1% positive class).

| Resource | Link |
|---|---|
| Written report (IEEE style) | `<ADD LINK or PDF>` |
| Google Drive folder (raw data zip, model weights) | `<ADD LINK>` |
| Presentation video | `<ADD LINK>` |

## Question

Given six months of repayment status, billed amounts and payments, which cardholders will miss the next
payment? I compare scikit-learn models with TensorFlow/Keras networks fed by `tf.data`, under one
leakage-free protocol, and ask whether the extra complexity of deep learning pays off on tabular credit data.

## Headline results (held-out test set, 4,500 cardholders)

| Model | ROC-AUC (95% CI) | Avg. precision | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Logistic regression, engineered features (M2) | 0.750 (0.732-0.768) | 0.497 | 0.571 | 0.479 | 0.521 |
| Tuned histogram gradient boosting (M8) | 0.783 (0.767-0.799) | 0.557 | 0.593 | 0.501 | 0.543 |
| Best MLP from random search (D12) | 0.778 (0.762-0.794) | 0.552 | 0.565 | 0.518 | 0.540 |

Thresholds maximise F1 on the validation set. 29 logged experiments in total (see `results/experiments.csv`
and the experiment table in the report).

## Repository layout

```
credit_default_classification.ipynb   main notebook (runs top to bottom)
creditfeatures.py                     cleaning + feature engineering used by the saved pipelines
requirements.txt                      pinned dependencies
results/                              experiment table, test metrics, error and cost analysis (CSV/JSON)
figures/                              figures used in the report
wandb_logs/                           Weights & Biases offline runs (one per experiment)
logs/                                 TensorBoard logs for the Keras runs
models/                               saved best models (also in the Drive folder)
data/                                 NOT committed; place raw.csv here (see below)
```

## Reproduce

1. `pip install -r requirements.txt` (Python 3.13 was used; TensorFlow 2.21).
2. Get the raw data. Either let the notebook download it from UCI (cell in section 2) or download the zipped
   raw dataset from the Drive folder and save the CSV as `data/raw.csv`. The raw file is deliberately not in git.
3. Run `jupyter nbconvert --to notebook --execute credit_default_classification.ipynb` or open the notebook
   in Jupyter/Colab and run all cells. A full run takes about 11 minutes on two CPU cores.
   `QUICK=1` shrinks all searches for a smoke test (numbers from a quick run are not meaningful).
4. View logs: `tensorboard --logdir logs`; W&B offline runs can be synced later with `wandb sync wandb_logs/wandb/offline-run-*`.

All seeds are fixed (Python, NumPy, TensorFlow) and TensorFlow deterministic ops are enabled. Two complete
executions gave the same experiment table.

## Using the saved models

```python
import joblib, creditfeatures            # creditfeatures must be importable
ml = joblib.load("models/best_ml_pipeline.joblib")
p = ml.predict_proba(raw_rows)[:, 1]      # raw_rows: DataFrame with the original predictor columns (no ID, no target)
flag = p >= 0.2576                        # validation-chosen F1 threshold
```

The Keras model is in `models/best_dl_model.keras` (and legacy `.h5`); its fitted preprocessor and threshold are
in `models/best_dl_preprocessor.joblib`.

## Data

Yeh, I.-C. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C55S3H. Collected from a Taiwanese card issuer, April to September 2005.
