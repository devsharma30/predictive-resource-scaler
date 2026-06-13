# main.py  —  Run the entire project pipeline in one command

from visualize import (
    chart1_raw_cpu_pattern, chart2_actual_vs_predicted,
    chart3_reactive_vs_proactive, chart4_cost_comparison,
    chart5_feature_importance
)
from simulate import run_simulation, print_and_return_results
from train_models import (
    scale_features, train_linear_regression,
    train_random_forest_regressor, train_random_forest_classifier,
    evaluate_regressor, evaluate_classifier,
    show_feature_importance, save_models,
    SPIKE_THRESHOLD, PREDICT_AHEAD
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from data_prep import (
    load_data, add_time_features, add_lag_features, add_rolling_features,
    create_target_variable, clean_data, prepare_features_labels, time_based_split
)
from generate_data import generate_server_data
import os
import time
import pandas as pd
import json

print("=" * 65)
print("  PREDICTIVE RESOURCE SCALER — Full Pipeline")
print("  Running all 5 steps in sequence...")
print("=" * 65)

# ── STEP 1: Generate Data ──────────────────────────────────────────────────────
print("\n[1/5] 📊 Generating synthetic server data...")
t = time.time()

os.makedirs("data", exist_ok=True)
df_raw = generate_server_data(days=90)
df_raw.to_csv("data/server_logs.csv", index=False)

print(
    f"      ✅ Done in {time.time()-t:.1f}s  —  {len(df_raw):,} rows generated")


# ── STEP 2: Feature Engineering ───────────────────────────────────────────────
print("\n[2/5] ⚙️  Engineering features...")
t = time.time()


df = load_data()
df = add_time_features(df)
df = add_lag_features(df)
df = add_rolling_features(df)
df = create_target_variable(df, threshold=78, predict_ahead=5)
df = clean_data(df)
df.to_csv("data/processed_logs.csv", index=False)

print(
    f"      ✅ Done in {time.time()-t:.1f}s  —  {len(df.columns)} feature columns created")


# ── STEP 3: Train Models ───────────────────────────────────────────────────────
print("\n[3/5] 🤖 Training models (~30 seconds)...")
t = time.time()


X, y_class, y_reg, df = prepare_features_labels(df)
feature_names = list(X.columns)
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = \
    time_based_split(X, y_class, y_reg)

X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

lr_reg = train_linear_regression(X_train_sc, y_reg_train)
rf_reg = train_random_forest_regressor(X_train, y_reg_train)
rf_clf = train_random_forest_classifier(X_train, y_class_train)

lr_res = evaluate_regressor(
    lr_reg,  X_test_sc, y_reg_test,   "Linear Regression")
rf_res = evaluate_regressor(rf_reg,  X_test,    y_reg_test,   "Random Forest")
clf_res = evaluate_classifier(rf_clf, X_test,    y_class_test)

imp_df = show_feature_importance(rf_reg, feature_names)
os.makedirs("models", exist_ok=True)
imp_df.to_csv("models/feature_importance.csv", index=False)

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

print(f"      ✅ Done in {time.time()-t:.1f}s")
print(
    f"         Linear Regression  MAE={lr_res['mae']:.2f}%  RMSE={lr_res['rmse']:.2f}%  R²={lr_res['r2']:.3f}")
print(
    f"         Random Forest      MAE={rf_res['mae']:.2f}%  RMSE={rf_res['rmse']:.2f}%  R²={rf_res['r2']:.3f}")
print(
    f"         RF Classifier      Precision={clf_res['precision']*100:.1f}%  Recall={clf_res['recall']*100:.1f}%")


# ── STEP 4: Simulation ─────────────────────────────────────────────────────────
print("\n[4/5] ⚡ Running reactive vs proactive simulation...")
t = time.time()


reactive, proactive = run_simulation(test_df)
results = print_and_return_results(reactive, proactive)

pd.DataFrame(reactive.minute_log).to_csv(
    "data/reactive_log.csv",   index=False)
pd.DataFrame(proactive.minute_log).to_csv(
    "data/proactive_log.csv", index=False)
with open("data/simulation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"      ✅ Done in {time.time()-t:.1f}s")


# ── STEP 5: Charts ─────────────────────────────────────────────────────────────
print("\n[5/5] 📈 Generating charts...")
t = time.time()


chart1_raw_cpu_pattern(df)
chart2_actual_vs_predicted(test_df)
chart3_reactive_vs_proactive(reactive.minute_log, proactive.minute_log)
chart4_cost_comparison(results)
chart5_feature_importance()

print(f"      ✅ Done in {time.time()-t:.1f}s")


# ── FINAL SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ✅ ALL DONE")
print("=" * 65)
print(f"\n  📁 Files generated:")
print(f"     data/server_logs.csv")
print(f"     data/processed_logs.csv")
print(f"     data/test_with_predictions.csv")
print(f"     data/reactive_log.csv")
print(f"     data/proactive_log.csv")
print(f"     data/simulation_results.json")
print(f"     models/lr_regressor.pkl")
print(f"     models/rf_regressor.pkl")
print(f"     models/rf_classifier.pkl")
print(f"     models/feature_importance.csv")
print(f"     charts/01_raw_cpu_pattern.png")
print(f"     charts/02_actual_vs_predicted.png")
print(f"     charts/03_reactive_vs_proactive.png")
print(f"     charts/04_cost_comparison.png")
print(f"     charts/05_feature_importance.png")
print(f"\n  📊 Key Results:")
print(f"     RF Precision  : {clf_res['precision']*100:.1f}%")
print(f"     RF MAE        : {rf_res['mae']:.2f}% CPU")
print(f"     Cost saved    : ${results['cost_saved']:.2f}")
print(f"     Breach reduced: {results['breach_reduced']:,} minutes")
