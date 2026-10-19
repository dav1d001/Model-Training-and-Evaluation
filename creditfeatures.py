"""Stateless cleaning and feature engineering for the credit-card default dataset.

Kept in a module (not in the notebook) so a pickled scikit-learn pipeline can be loaded anywhere:
    import creditfeatures, joblib
    model = joblib.load("models/best_ml_pipeline.joblib")
    model.predict_proba(raw_rows)[:, 1]        # raw_rows: DataFrame with the original predictor columns
"""
import numpy as np

PAY = [f"PAY_{i}" for i in range(1, 7)]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT = [f"PAY_AMT{i}" for i in range(1, 7)]
CAT = ["SEX", "EDUCATION", "MARRIAGE"]
ENG = ["UTIL_MEAN", "UTIL_MAX", "UTIL_LAST", "BILL_TREND", "PAYRATIO_MEAN", "PAYRATIO_MIN",
       "PAYRATIO_LAST", "N_DELAY", "MAX_DELAY", "MEAN_STATUS", "ANY_CREDIT_BAL"]
DEMOGRAPHIC = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]


def signed_log1p(x):
    """sign(x) * log(1 + |x|): compresses heavy-tailed money amounts and keeps negative balances."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def prepare(df):
    """Clean undocumented codes and add engineered credit-risk features. Input: raw predictor columns."""
    d = df.copy()
    if "PAY_0" in d.columns:
        d = d.rename(columns={"PAY_0": "PAY_1"})
    d["EDUCATION"] = d["EDUCATION"].replace({0: 4, 5: 4, 6: 4})   # undocumented codes -> "others"
    d["MARRIAGE"] = d["MARRIAGE"].replace({0: 3})

    # utilisation = positive bill / credit limit (winsorised at 3x the limit)
    util = d[BILL].clip(lower=0).div(d["LIMIT_BAL"], axis=0).clip(upper=3)
    d["UTIL_MEAN"], d["UTIL_MAX"], d["UTIL_LAST"] = util.mean(axis=1), util.max(axis=1), util["BILL_AMT1"]
    d["BILL_TREND"] = ((d["BILL_AMT1"] - d["BILL_AMT6"]) / d["LIMIT_BAL"]).clip(-3, 3)

    # payment ratio: the payment made in month k settles the bill of month k+1 (nothing owed -> ratio 1)
    ratios = []
    for k in range(1, 6):
        prev = d[f"BILL_AMT{k + 1}"]
        r = np.where(prev > 0, d[f"PAY_AMT{k}"] / prev.where(prev > 0, 1), 1.0)
        ratios.append(np.clip(r, 0, 2))
    R = np.column_stack(ratios)
    d["PAYRATIO_MEAN"], d["PAYRATIO_MIN"], d["PAYRATIO_LAST"] = R.mean(1), R.min(1), R[:, 0]

    d["N_DELAY"] = (d[PAY] >= 1).sum(axis=1)          # months with a payment delay
    d["MAX_DELAY"] = d[PAY].max(axis=1)
    d["MEAN_STATUS"] = d[PAY].mean(axis=1)
    d["ANY_CREDIT_BAL"] = (d[BILL] < 0).any(axis=1).astype(int)
    return d
