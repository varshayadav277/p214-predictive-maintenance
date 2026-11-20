"""Train the three models (health class, failure within 48 h, RUL) and print results."""
import os
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    mean_absolute_error, mean_squared_error, confusion_matrix
)
from features import load_and_clean, add_features, feature_columns, make_labels, CLASSES

TEST_MACHINES = ["RM03", "RM07"]  # never seen during training

def make_models():
    """XGBoost if it is installed, otherwise scikit-learn Gradient Boosting."""
    try:
        from xgboost import XGBClassifier, XGBRegressor
        clf = lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.07, random_state=2026)
        reg = lambda: XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.06, random_state=2026)
        name = "XGBoost"
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        clf = lambda: GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.07, random_state=2026)
        reg = lambda: GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.06, random_state=2026)
        name = "Gradient Boosting (scikit-learn)"
    return clf, reg, name

def fit_models(train_df, cols):
    clf, reg, name = make_models()
    y_cls, y_48, y_rul = make_labels(train_df)
    X = train_df[cols]
    return {
        "class": clf().fit(X, y_cls),
        "fail48": clf().fit(X, y_48),
        "rul": reg().fit(X, y_rul),
        "features": cols,
        "name": name
    }

def predict(models, df):
    X = df[models["features"]]
    return (
        models["class"].predict(X),
        models["fail48"].predict_proba(X)[:, 1],
        np.clip(models["rul"].predict(X), 0, None)
    )

def lead_time(df, pred_cls):
    """Hours before failure at which the machine was first flagged Warning/Critical."""
    out = {}
    for m in df.machine_id.unique():
        sel = (df.machine_id == m).values
        flagged = np.where(pred_cls[sel] >= 1)[0]
        out[m] = int(df.rul_hours.values[sel][flagged[0]]) if len(flagged) else 0
    return out

def evaluate(models, test_df):
    y_cls, y_48, y_rul = make_labels(test_df)
    p_cls, p_prob, p_rul = predict(models, test_df)
    P, R, F, _ = precision_recall_fscore_support(y_cls, p_cls, average="macro")
    pc, rc, fc, _ = precision_recall_fscore_support(y_cls, p_cls, average=None)
    p48, r48, f48, _ = precision_recall_fscore_support(y_48, (p_prob >= 0.5).astype(int), average="binary")
    return {
        "accuracy": accuracy_score(y_cls, p_cls),
        "precision": P,
        "recall": R,
        "f1": F,
        "per_class": {c: (pc[i], rc[i], fc[i]) for i, c in enumerate(CLASSES)},
        "fail48": (p48, r48, f48, roc_auc_score(y_48, p_prob)),
        "rul_mae": mean_absolute_error(y_rul, p_rul),
        "rul_rmse": mean_squared_error(y_rul, p_rul) ** 0.5,
        "lead": lead_time(test_df, p_cls),
        "confusion": confusion_matrix(y_cls, p_cls)
    }

def split(df):
    return df[~df.machine_id.isin(TEST_MACHINES)], df[df.machine_id.isin(TEST_MACHINES)]

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    df = add_features(load_and_clean())
    cols = feature_columns(df)
    train_df, test_df = split(df)
    models = fit_models(train_df, cols)
    res = evaluate(models, test_df)
    joblib.dump(models, "models/p214_models.joblib")
    print(f"Model: {models['name']} | train rows {len(train_df)} | test rows {len(test_df)}")
    print(f"Health class acc {res['accuracy']:.3f} P {res['precision']:.3f} R {res['recall']:.3f} F1 {res['f1']:.3f}")
    for c, (p, r, f) in res["per_class"].items():
        print(f"  {c:9s} P {p:.3f} R {r:.3f} F1 {f:.3f}")
    p, r, f, auc = res["fail48"]
    print(f"Fail <=48h P {p:.3f} R {r:.3f} F1 {f:.3f} AUC {auc:.4f}")
    print(f"RUL MAE {res['rul_mae']:.1f} h RMSE {res['rul_rmse']:.1f} h")
    print("Alert lead time (h before failure):", res["lead"])
    print("Confusion matrix:\n", res["confusion"])
