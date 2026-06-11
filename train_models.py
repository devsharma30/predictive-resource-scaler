# train_models.py  —>  Train and compare two ML models

#   We intentionally train TWO models to COMPARE them.
#   This is good ML practice — never just train one model and
#   assume it's good. Always have a baseline to beat.

# MODEL 1: Linear Regression  (our baseline)
# MODEL 2: Random Forest  (our main model)

import numpy as np
import pandas as pd
# pickle: converts Python objects into bytes and saves them to disk.
import pickle
import json
import os


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# Regressor  → predicts a continuous number (CPU %)
# Classifier → predicts a category (0 or 1)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,    # MAE:  average of |predicted - actual|
    # MSE:  average of (predicted - actual)²  — we sqrt it
    mean_squared_error,
    # R²:   how much variance the model explains (1.0 = perfect)
    r2_score,
    precision_score,        # "when we said SPIKE, how often were we right?"
    recall_score,           # "of all actual spikes, how many did we catch?"
    f1_score,               # harmonic mean of precision and recall
    accuracy_score,         # overall % of correct predictions
    confusion_matrix        # table: TP / FP / TN / FN breakdown
)

from data_prep import (
    load_data, add_time_features, add_lag_features, add_rolling_features,
    create_target_variable, clean_data, prepare_features_labels, time_based_split
)

# ── TUNED CONSTANTS ────────────────────────────────────────────────────────────
# These are NOT arbitrary. Each one was chosen after testing.

SPIKE_THRESHOLD = 78
PREDICT_AHEAD = 5
DECISION_THRESHOLD = 0.78
# The probability cutoff to fire a spike warning.
# WHY 0.78 and not the default 0.50?
#   sklearn's .predict() uses 0.50 by default: warn if prob > 50%.
#   That fires too many false alarms → operators ignore the warnings.
#
#   We use .predict_proba()[:,1] to get the raw probability,
#   then apply our own threshold of 0.78:
#   "Only warn me when the model is at least 78% confident."


def scale_features(X_train, X_test):
    """
    Standardises features so every column has mean=0 and std=1.

    KEY CONCEPT — Why scale at all?
        Our features span very different ranges:
            hour            0 – 23
            cpu_usage       0 – 100
            request_count   0 – 5,000

        Linear Regression computes: w1*hour + w2*cpu + w3*requests + ...
        Without scaling, the model might shrink w3 to near zero and inflate
        w1 just to balance the number sizes — NOT because hour matters more.
        Scaling removes this distortion so all features compete fairly.

    KEY CONCEPT — fit on TRAIN only, then transform BOTH:
        scaler.fit_transform(X_train):
            Step 1: compute mean and std of each column FROM X_train   (fit)
            Step 2: apply  (value - mean) / std  to every cell         (transform)

        scaler.transform(X_test):
            Apply the EXACT SAME mean and std that was learned from X_train.
            We do NOT fit again on the test set.

        why i am not doing fit on test?
            If we compute the test set's own mean and std, we've used
            information from the test set to influence our processing.
            That's "data leakage" — the test set must be treated as
            completely unknown data, like truly future server logs.

    KEY CONCEPT — Random Forest doesn't need scaling:
        RF makes decisions by comparing feature values to thresholds:
        "is cpu_lag_5 > 72?"
        The scale of the number doesn't affect whether it crosses a threshold.
        So we only pass scaled data to the Linear Regression model.
        RF gets the original X_train and X_test directly.
    """
    print("\n📏 Scaling features (mean=0, std=1) for Linear Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        X_train)   # learn from train, apply to train
    X_test_scaled = scaler.transform(X_test)        # apply SAME scale to test
    print("   ✅ Scaling complete")
    return X_train_scaled, X_test_scaled, scaler


def train_linear_regression(X_train, y_train):
    """
    Trains a Linear Regression model.

    KEY CONCEPT — What does "training" mean?
        Training = finding the numbers (weights and bias) that make the
        model's predictions as close as possible to the actual values.

        The model predicts: ŷ = w1*x1 + w2*x2 + ... + wN*xN + b
        where w1...wN are the weights and b is the bias (intercept).

        "Training" finds the specific values of w1...wN, b that minimise
        the total squared error across all training rows:
            Σ (ŷᵢ - yᵢ)²  →  minimise this

        This is called "Ordinary Least Squares" and has an exact
        mathematical solution — no guessing or iterating needed.
        That's why .fit() is so fast for Linear Regression.

        After .fit(), the model stores the learned weights in:
            model.coef_       ← the weights array
            model.intercept_  ← the bias value
    """
    print("\n🤖 Training Linear Regression (baseline)...")
    model = LinearRegression()
    model.fit(X_train, y_train)    # THE learning step
    print("   ✅ Done")
    return model


def show_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Prints which input features the Random Forest relied on most.

    KEY CONCEPT — feature_importances_:
        After training, every sklearn Random Forest exposes a
        .feature_importances_ attribute.
        It's an array where each value = how much that feature
        REDUCED prediction error across all splits in all trees.
        Values sum to 1.0.

        feature_importances_[i] = 0.22 means feature i was responsible
        for 22% of all the error reduction in the forest.

    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)   # sort highest first

    print("\n🔍 Top 10 most important features (Random Forest):")
    print(importance_df.head(10).to_string(index=False))
    return importance_df
