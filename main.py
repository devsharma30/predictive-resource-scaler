# main.py  —  Run the entire project pipeline in one command

import subprocess
import sys
import time


def run_step(script: str, step_num: int, total: int):
    """
    Runs a Python script as a subprocess and streams its output in real time.
    Exits immediately if any script fails (no point running later steps on bad data).
    """
    print(f"\n{'='*60}")
    print(f"STEP {step_num}/{total}: {script}")
    print(f"{'='*60}")

    start = time.time()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False   # let stdout/stderr stream directly to terminal
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n✗ FAILED: {script} exited with code {result.returncode}")
        print("Pipeline stopped. Fix the error above and re-run main.py")
        sys.exit(1)

    print(f"\n✓ {script} completed in {elapsed:.1f}s")


def main():
    print("=" * 60)
    print("PREDICTIVE RESOURCE SCALER — Full Pipeline")
    print("=" * 60)
    print("Running all 5 steps in sequence...")

    steps = [
        'generate_data.py',
        'data_prep.py',
        'train_models.py',
        'simulate.py',
        'visualize.py',
    ]

    pipeline_start = time.time()

    for i, script in enumerate(steps, 1):
        run_step(script, i, len(steps))

    total_time = time.time() - pipeline_start

    print(f"\n{'='*60}")
    print(f"ALL STEPS COMPLETE — {total_time:.0f}s total")
    print(f"{'='*60}")
    print("Output files:")
    print("  data/server_logs.csv          ← synthetic server data")
    print("  data/processed_logs.csv       ← feature-engineered data")
    print("  data/test_with_predictions.csv ← model predictions on test set")
    print("  data/reactive_log.csv         ← minute-by-minute reactive sim")
    print("  data/proactive_log.csv        ← minute-by-minute proactive sim")
    print("  data/simulation_results.json  ← cost/SLA breach summary")
    print("  models/rf_classifier.pkl      ← main ML model")
    print("  models/results.json           ← model accuracy metrics")
    print("  charts/01_raw_cpu_pattern.png")
    print("  charts/02_actual_vs_predicted.png")
    print("  charts/03_reactive_vs_proactive.png")
    print("  charts/04_cost_comparison.png")
    print("  charts/05_feature_importance.png")


if __name__ == '__main__':
    main()
