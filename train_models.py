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


def train_random_forest_regressor(X_train, y_train):
    """
    Trains a Random Forest to predict the exact future CPU value (regression).

    KEY CONCEPT — How a Decision Tree works:
        A decision tree splits data with yes/no questions:
            "Is cpu_lag_5 > 75?"
            ├─ YES: "Is hour > 18?" → leaf: predict CPU = 88
            └─ NO:  "Is is_weekend = 1?" → leaf: predict CPU = 42
        It keeps splitting until it reaches "leaf" nodes with predictions.
        One tree can overfit: it memorises the training data perfectly
        but fails on new data.

    KEY CONCEPT — How Random Forest fixes overfitting:
        Step 1: Create 100 "bootstrap samples" from training data.
                Bootstrap = sample WITH replacement (same row can appear twice).
                Like shuffling a deck 100 times and dealing 100 different hands.

        Step 2: Train ONE decision tree on EACH bootstrap sample.
                At each split, each tree also randomly picks only a subset of
                features to consider. This is the "random" in Random Forest.
                → Every tree is slightly different. Each sees different data
                and different features → each learns different patterns.

        Step 3: Prediction = average of all 100 trees' predictions.
                No single tree dominates. Their collective average is
                robust and generalises well to new data.

    KEY PARAMETERS EXPLAINED:
        n_estimators=100     → number of trees.
                               More trees = better accuracy but slower training.
                               100 is a good starting point. 50 works too.

        max_depth=10         → each tree can only be 10 levels deep.
                               Without this limit, trees grow until every
                               training sample is perfectly memorised (overfitting).

        min_samples_leaf=15  → every leaf must represent at least 15 training samples.
                               Prevents tiny leaves that memorise single data points.

        n_jobs=-1            → use ALL available CPU cores to train trees in parallel.
                               -1 is scikit-learn's convention for "use everything".
                               Without this, trees are trained one at a time (slow).

        random_state=42      → controls all internal randomness.
                               Same seed = same forest every run = reproducible results.
    """
    print("\n🌲 Training Random Forest Regressor...")
    print("   (building 100 trees — ~30 seconds)")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=15,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("   ✅ Done")
    return model


def train_random_forest_classifier(X_train, y_train):
    """
    Trains a Random Forest to predict 0 or 1 (no spike / spike coming).

    KEY CONCEPT — Regressor vs Classifier:
        RandomForestRegressor  → predicts a continuous number  (e.g. "CPU will be 84.3%")
        RandomForestClassifier → predicts a category            (e.g. "spike: YES or NO")

        Internally they work almost identically.
        The difference is in the leaf nodes:
          - Regressor leaves store the average target value of their training samples
          - Classifier leaves store the majority class of their training samples

    KEY CONCEPT — predict() vs predict_proba():
        model.predict(X)         → returns [0, 1, 0, 1, 0, ...]  (hard class labels)
                                   Uses default threshold = 0.50
        model.predict_proba(X)   → returns [[0.8, 0.2], [0.1, 0.9], ...]
                                   Each row = [P(class 0), P(class 1)]
                                   Column index 1 = probability of spike

        We use predict_proba()[:,1] + our own threshold (0.78) instead of
        predict() because the default 0.50 threshold fires too many false alarms.
        Choosing our own threshold gives us direct control over precision.
    """
    print("\n🌲 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=15,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    print("   ✅ Done")
    return model


def evaluate_regressor(model, X_test, y_test, model_name: str) -> dict:
    """
    Measures how well the regression model predicts future CPU values.

    KEY CONCEPT — MAE (Mean Absolute Error):
        Step 1: For each test sample, compute |predicted_cpu - actual_cpu|
        Step 2: Average all those absolute errors.
        Result: "On average, my prediction is off by X percentage points."
        MAE = 8.2% means on average we're 8.2 CPU percentage points off.
        Easy to interpret. Treats a 1% error and a 10% error proportionally.

    KEY CONCEPT — RMSE (Root Mean Squared Error):
        Step 1: For each sample, compute (predicted - actual)²
        Step 2: Average all the squared errors  (= MSE)
        Step 3: Take the square root            (= RMSE, same units as CPU %)
        RMSE PUNISHES large errors more than MAE because it squares them.
        A 20% error contributes 400 to MSE.  A 2% error contributes only 4.
        Use RMSE when large errors are especially bad (e.g. missing a big spike).

    KEY CONCEPT — R² (R-squared, Coefficient of Determination):
        Answers: "How much better is this model than just predicting the mean?"
        R² = 1.0  → perfect predictions
        R² = 0.0  → model is no better than always predicting the average CPU
        R² = 0.7  → model explains 70% of the variance in actual CPU values
        The closer to 1.0, the better.
        R² measures: "how much closer are MY predictions to the diagonal line (perfect) compared to the flat average line?"
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 {model_name} — Regression Results:")
    print(f"   MAE  (avg CPU% error)         : {mae:.2f}%")
    print(f"   RMSE (large errors penalised) : {rmse:.2f}%")
    print(f"   R²   (1.0 = perfect)          : {r2:.4f}")

    return {
        'model_name': model_name,
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'predictions': y_pred       # keep predictions for simulate/visualize
    }


def evaluate_classifier(model, X_test, y_test) -> dict:
    """
    Measures how well the classifier detects upcoming spikes.

    KEY CONCEPT — The Confusion Matrix:
        After predicting on the test set we compare predicted vs actual:

                              PREDICTED
                          No Spike    Spike
        ACTUAL  No Spike [ TN          FP  ]
                   Spike [ FN          TP  ]

        TP (True Positive) : Said SPIKE,    was a spike    ✅ caught it
        TN (True Negative) : Said NO SPIKE, was no spike   ✅ correctly quiet
        FP (False Positive): Said SPIKE,    was NOT spike  ❌ false alarm
        FN (False Negative): Said NO SPIKE, WAS a spike    ❌ missed it

    KEY CONCEPT — Precision:
        TP / (TP + FP)
        "Of every SPIKE WARNING we fired, what fraction were real spikes?"
        HIGH precision = operators trust the warnings.
        LOW precision  = so many false alarms that operators ignore them.
        → This is your CV metric: "~87% precision"

    KEY CONCEPT — Recall (also called Sensitivity):
        TP / (TP + FN)
        "Of every ACTUAL spike that happened, what fraction did we warn about?"
        HIGH recall = we catch most spikes.
        LOW recall  = many spikes slip through without warning.

    KEY CONCEPT — Precision vs Recall trade-off:
        You cannot maximise both at the same time.
        Raising DECISION_THRESHOLD → precision goes up, recall goes down.
        (We warn less often, but when we do, we're almost always right.)
        Lowering DECISION_THRESHOLD → recall goes up, precision goes down.
        (We warn more often, catching more spikes but also more false alarms.)
        We chose 0.78 to prioritise precision for operational trustworthiness.

    KEY CONCEPT — F1 Score:
        2 × (Precision × Recall) / (Precision + Recall)
        The harmonic mean of precision and recall.
        A single number that balances both.
        Useful when you want one number that summarises the classifier quality.
    """

    probs = model.predict_proba(X_test)[:, 1]

    y_pred = (probs >= DECISION_THRESHOLD).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred,    zero_division=0)
    f1 = f1_score(y_test, y_pred,         zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(
        f"\n📊 Random Forest Classifier (decision threshold = {DECISION_THRESHOLD}):")
    print(f"   Accuracy  : {accuracy * 100:.1f}%")
    print(f"   Precision : {precision * 100:.1f}%  ← this goes on your CV")
    print(f"   Recall    : {recall * 100:.1f}%")
    print(f"   F1 Score  : {f1 * 100:.1f}%")
    print(
        f"   Warnings fired : {y_pred.sum():,} out of {len(y_pred):,} test minutes")
    print(f"\n   Confusion Matrix (rows=actual, cols=predicted):")
    print(f"   [TN={cm[0, 0]:>5,}   FP={cm[0, 1]:>5,}]")
    print(f"   [FN={cm[1, 0]:>5,}   TP={cm[1, 1]:>5,}]")

    return {
        'model_name': 'Random Forest Classifier',
        'threshold': DECISION_THRESHOLD,
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        # .tolist() converts numpy array to plain Python list
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred,
        'probabilities': probs
    }


def save_models(models: dict, scaler, results: dict):
    """
    Persists trained models and the scaler to disk using pickle.

    KEY CONCEPT — Why save models?
        Training takes ~30 seconds. simulate.py and visualize.py
        need the trained models too. Without saving, we'd retrain
        every time we run those files — wasteful.

        pickle.dump(object, file):
            Serialises (converts) any Python object into bytes.
            Writes those bytes to a binary file.

        pickle.load(file):
            Reads the bytes back.
            Reconstructs the exact Python object.
            The loaded model can immediately call .predict() — no retraining.

    """
    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        path = f"models/{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"   💾 Saved: {path}")

    # Save the scaler too — needed to preprocess live data in production
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    safe_results = {}
    for key, val in results.items():
        if isinstance(val, dict):       # isinstance checks the type of an object
            safe_results[key] = {
                k: (v.tolist() if hasattr(v, 'tolist') else v)
                # hasattr checks if an object has an attribute
                # numpy arrays have .tolist(), plain Python floats don't
                for k, v in val.items()
                # skip large arrays
                if k not in ['predictions', 'probabilities']
            }
    with open("models/results.json", "w") as f:
        json.dump(safe_results, f, indent=2)
    print("   💾 Saved: models/results.json")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PREDICTIVE RESOURCE SCALER — MODEL TRAINING")
    print("=" * 60)

    # ── 1. Prepare data (reuses all functions from data_prep.py) ──────────────
    df = load_data()
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = create_target_variable(
        df, threshold=SPIKE_THRESHOLD, predict_ahead=PREDICT_AHEAD)
    df = clean_data(df)
    X, y_class, y_reg, df = prepare_features_labels(df)

    feature_names = list(X.columns)

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = \
        time_based_split(X, y_class, y_reg)

    # ── 2. Scale features for Linear Regression ───────────────────────────────
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    # ── 3. Train all three models ─────────────────────────────────────────────
    # Linear Regression gets SCALED data
    lr_reg = train_linear_regression(X_train_sc, y_reg_train)
    # Random Forest gets ORIGINAL (unscaled) data as it does not need  scaling
    rf_reg = train_random_forest_regressor(X_train, y_reg_train)
    rf_clf = train_random_forest_classifier(X_train, y_class_train)

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    lr_res = evaluate_regressor(
        lr_reg,  X_test_sc, y_reg_test,   "Linear Regression")
    rf_res = evaluate_regressor(
        rf_reg,  X_test,    y_reg_test,   "Random Forest")
    clf_res = evaluate_classifier(rf_clf, X_test,    y_class_test)

    # ── 5. Feature importance ─────────────────────────────────────────────────
    imp_df = show_feature_importance(rf_reg, feature_names)
    os.makedirs("models", exist_ok=True)
    imp_df.to_csv("models/feature_importance.csv", index=False)

    # ── 6. Print final comparison table ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 52)
    print(
        f"{'Linear Regression':<25} {lr_res['mae']:>7.2f}% {lr_res['rmse']:>7.2f}% {lr_res['r2']:>8.4f}")
    print(
        f"{'Random Forest':<25} {rf_res['mae']:>7.2f}% {rf_res['rmse']:>7.2f}% {rf_res['r2']:>8.4f}")
    print(
        f"\n  RF Classifier Precision : {clf_res['precision']*100:.1f}%  ← CV bullet metric")

    # ── 7. Save everything ───────────────────────────────────────────────────
    print("\n💾 Saving models...")
    save_models(
        {'lr_regressor': lr_reg, 'rf_regressor': rf_reg, 'rf_classifier': rf_clf},
        scaler,
        {'lr_reg': lr_res, 'rf_reg': rf_res, 'rf_clf': clf_res}
    )

    test_df = df.iloc[-len(X_test):].copy()
    test_df['lr_predicted_cpu'] = lr_reg.predict(X_test_sc)
    test_df['rf_predicted_cpu'] = rf_reg.predict(X_test)
    test_df['rf_spike_predicted'] = clf_res['predictions']
    test_df['rf_spike_prob'] = clf_res['probabilities']
    test_df.to_csv("data/test_with_predictions.csv", index=False)
    print("   💾 Saved: data/test_with_predictions.csv")

    print("\n✅ All done. Next step: python simulate.py")
