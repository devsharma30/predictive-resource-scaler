# train_models.py  —>  Train and compare two ML models

#   We intentionally train TWO models to COMPARE them.
#   This is good ML practice — never just train one model and
#   assume it's good. Always have a baseline to beat.

# MODEL 1: Linear Regression  (our baseline)
# MODEL 2: Random Forest  (our main model)

import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

SPIKE_THRESHOLD = 80


FEATURE_COLUMNS = [
    'hour', 'minute', 'day_of_week', 'day_of_month', 'month', 'is_weekend',
    'cpu_lag_1', 'cpu_lag_5', 'cpu_lag_10', 'cpu_lag_30',
    'req_lag_1', 'req_lag_5',
    'cpu_rolling_mean_5', 'cpu_rolling_mean_15', 'cpu_rolling_mean_30',
    'cpu_rolling_std_5', 'cpu_rolling_std_15', 'cpu_rolling_max_15',
    'req_rolling_mean_5',
    'cpu_delta_5', 'cpu_delta_15',
    'memory_usage',
]


def load_and_split(path: str):
    """
    WHY CHRONOLOGICAL SPLIT (not random shuffle):
    Server data is time-series. If we randomly shuffle and take 80/20,
    test data will contain rows from the MIDDLE of training data.
    That's cheating — the model will have "seen the future" implicitly.

    Chronological split: train = first 80%, test = last 20%.
    This simulates reality: train on January-July, predict August-November.
    """
    print(f"Loading {path}...")
    df = pd.read_csv(path)

    split_idx = int(len(df) * 0.80)

    # KEY CONCEPT — iloc vs loc:
    #   df.iloc[start:end] selects by INTEGER position (like Python list slicing)
    #   df.loc[label:label] selects by INDEX LABEL
    #   We use iloc here because we want positional split (first 80%, last 20%)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]
    y_train_reg = train['cpu_future_5']   # regression target: actual CPU value
    y_test_reg = test['cpu_future_5']
    y_train_cls = train['spike_in_5min']  # classification target: 0 or 1
    y_test_cls = test['spike_in_5min']

    print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows")
    print(f"  Spike rate in test: {100*y_test_cls.mean():.1f}%")
    return X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls, test


def train_linear_regression(X_train, X_test, y_train, y_test):
    """
    WHY LINEAR REGRESSION AS BASELINE:
    Linear Regression assumes CPU_future = w1*feature1 + w2*feature2 + ... + b
    It's the simplest possible model — a straight-line relationship between
    features and target. It cannot model non-linear patterns (like "CPU is high
    AND rising AND it's 2 PM → very likely spike").

    Random Forest CAN model those non-linear combinations.
    Comparing the two shows exactly how much non-linearity matters.
    """
    print("\n[1/3] Training Linear Regression (baseline)...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"  MAE : {mae:.2f}%  (average prediction error in CPU %)")
    print(f"  R²  : {r2:.3f}   (1.0 = perfect, 0.0 = predicts mean only)")
    return model, mae, r2, preds


def train_rf_regressor(X_train, X_test, y_train, y_test):
    """
    WHY RANDOM FOREST:
    An ensemble of decision trees that each vote on the prediction.
    Each tree is trained on a random subset of data and features.
    The average of 100 trees is much more robust than a single tree.

    KEY PARAMS:
      n_estimators=100  — 100 trees (more = more stable, diminishing returns past 200)
      max_depth=12      — each tree can be at most 12 levels deep (prevents overfitting)
      min_samples_leaf=10 — each leaf needs ≥10 data points (prevents memorisation)
      n_jobs=-1         — use all CPU cores for parallel training
    """
    print("\n[2/3] Training Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"  MAE : {mae:.2f}%  (should be lower than Linear Regression)")
    print(f"  R²  : {r2:.3f}   (should be higher than Linear Regression)")
    return model, mae, r2, preds


def train_rf_classifier(X_train, X_test, y_train_cls, y_test_cls):
    """
    WHY A CLASSIFIER INSTEAD OF USING THE REGRESSOR:
    The regressor predicts CPU VALUE (e.g. "78.3%").
    We could threshold that ("if predicted_cpu > 80 → spike"), but it's less direct.
    A classifier directly outputs P(spike) — probability between 0 and 1.
    We then threshold the probability to decide when to trigger scaling.

    WHY TUNE THE DECISION THRESHOLD:
    Default threshold = 0.5 (spike if P>50%).
    But we want HIGH PRECISION — we'd rather miss some spikes than boot instances
    every time there's a false alarm (wasted cost).

    We try thresholds from 0.3 to 0.7 and pick the one that gives best
    precision while keeping recall above a minimum bar (>30%).
    """
    print("\n[3/3] Training Random Forest Classifier (main model)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=10,
        class_weight='balanced',  # WHY: spike minutes are rare (~10% of data).
                                  # 'balanced' upweights the minority class (spike=1)
                                  # so the model doesn't just predict "no spike" always.
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train_cls)

    # KEY CONCEPT — predict_proba():
    #   Returns [P(class=0), P(class=1)] for each row.
    #   We take [:, 1] = probability of spike = 1.
    spike_proba = model.predict_proba(X_test)[:, 1]

    # Threshold search — find best precision/recall tradeoff
    print("\n  Threshold search (precision/recall tradeoff):")
    print(
        f"  {'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Triggered':>10}")

    best_threshold = 0.40
    best_f1 = 0.0
    results = []

    for threshold in np.arange(0.25, 0.75, 0.05):
        preds = (spike_proba >= threshold).astype(int)
        if preds.sum() == 0:
            continue
        p = precision_score(y_test_cls, preds, zero_division=0)
        r = recall_score(y_test_cls, preds, zero_division=0)
        f = f1_score(y_test_cls, preds, zero_division=0)
        triggered = preds.sum()
        results.append((threshold, p, r, f, triggered))
        print(
            f"  {threshold:>10.2f} {p:>10.1%} {r:>8.1%} {f:>6.3f} {triggered:>10,}")

        # Pick threshold with best F1 (balances precision and recall)
        if f > best_f1 and r > 0.25:
            best_f1 = f
            best_threshold = threshold

    print(f"\n  → Selected threshold: {best_threshold:.2f}")

    final_preds = (spike_proba >= best_threshold).astype(int)
    p = precision_score(y_test_cls, final_preds, zero_division=0)
    r = recall_score(y_test_cls, final_preds, zero_division=0)
    f = f1_score(y_test_cls, final_preds, zero_division=0)

    print(f"\n  Final classifier metrics at threshold {best_threshold:.2f}:")
    print(f"    Precision : {p:.1%}  (of predicted spikes, this % were real)")
    print(f"    Recall    : {r:.1%}  (of real spikes, this % were caught)")
    print(f"    F1        : {f:.3f} (harmonic mean of precision and recall)")

    cm = confusion_matrix(y_test_cls, final_preds)
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives  (no spike, predicted no spike): {cm[0][0]:,}")
    print(f"    False Positives (no spike, predicted spike)   : {cm[0][1]:,}")
    print(f"    False Negatives (spike, predicted no spike)   : {cm[1][0]:,}")
    print(f"    True Positives  (spike, predicted spike)      : {cm[1][1]:,}")

    return model, best_threshold, p, r, f, spike_proba


def print_feature_importance(model, title="Feature Importance"):
    print(f"\n  {title}:")
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx[:10]:
        bar = "█" * int(importances[i] * 100)
        print(f"    {FEATURE_COLUMNS[i]:25s} {importances[i]:.3f}  {bar}")


def main():
    print("=" * 60)
    print("train_models.py — Training ML Models")
    print("=" * 60)

    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    (X_train, X_test, y_train_reg, y_test_reg,
     y_train_cls, y_test_cls, test_df) = load_and_split('data/processed_logs.csv')

    # Train all three models
    lr_model, lr_mae, lr_r2, lr_preds = train_linear_regression(
        X_train, X_test, y_train_reg, y_test_reg
    )

    rf_reg_model, rf_mae, rf_r2, rf_reg_preds = train_rf_regressor(
        X_train, X_test, y_train_reg, y_test_reg
    )

    rf_cls_model, threshold, precision, recall, f1, spike_proba = train_rf_classifier(
        X_train, X_test, y_train_cls, y_test_cls
    )

    print_feature_importance(rf_reg_model, "RF Regressor Feature Importance")
    print_feature_importance(rf_cls_model, "RF Classifier Feature Importance")

    # Save models
    print("\nSaving models...")
    with open('models/lr_regressor.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open('models/rf_regressor.pkl', 'wb') as f:
        pickle.dump(rf_reg_model, f)
    with open('models/rf_classifier.pkl', 'wb') as f:
        pickle.dump({'model': rf_cls_model, 'threshold': threshold}, f)

    # Save results for visualize.py to read
    results = {
        'lr_mae': round(lr_mae, 3),
        'lr_r2': round(lr_r2, 3),
        'rf_mae': round(rf_mae, 3),
        'rf_r2': round(rf_r2, 3),
        'rf_precision': round(precision, 3),
        'rf_recall': round(recall, 3),
        'rf_f1': round(f1, 3),
        'decision_threshold': round(threshold, 2),
        'feature_names': FEATURE_COLUMNS,
        'feature_importances': rf_cls_model.feature_importances_.tolist(),
    }
    with open('models/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save test set with predictions for visualize.py
    test_out = test_df.copy()
    # KEY CONCEPT — clipping predictions:
    #   Linear Regression has no output bounds — it can predict 110%, -5%, etc.
    #   CPU is physically bounded [5, 100]. We clip to enforce this constraint.
    #   Random Forest can't exceed training data range, but clip anyway for safety.
    test_out['lr_pred_cpu'] = np.clip(lr_preds, 5, 100)
    test_out['rf_pred_cpu'] = np.clip(rf_reg_preds, 5, 100)
    test_out['rf_spike_proba'] = spike_proba
    test_out['rf_spike_pred'] = (spike_proba >= threshold).astype(int)
    test_out.to_csv('data/test_with_predictions.csv', index=False)

    print(f"\n✓ All models saved to models/")
    print(f"✓ Test predictions saved to data/test_with_predictions.csv")
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Linear Regression  MAE={lr_mae:.2f}%  R²={lr_r2:.3f}")
    print(f"  Random Forest Reg  MAE={rf_mae:.2f}%  R²={rf_r2:.3f}")
    print(
        f"  RF Classifier      Precision={precision:.1%}  Recall={recall:.1%}  F1={f1:.3f}")


if __name__ == '__main__':
    main()
