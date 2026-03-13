"""
monitor.py - Drift detection using Evidently AI.
Compares new prediction data against the training reference dataset.

Usage:
    python src/monitor.py --current data/current_batch.csv
"""

import argparse
import logging
import os
import pandas as pd
import yaml

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_drift_report(reference_path, current_path, output_path):
    logger.info(f"Loading reference data from {reference_path}")
    reference = pd.read_csv(reference_path)

    logger.info(f"Loading current data from {current_path}")
    current = pd.read_csv(current_path)

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])
    report.run(reference_data=reference, current_data=current)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)
    logger.info(f"Drift report saved to {output_path}")

    drift_detected = report.as_dict()["metrics"][1]["result"]["dataset_drift"]
    if drift_detected:
        logger.warning("DRIFT DETECTED - consider retraining the model.")
    else:
        logger.info("No significant drift detected.")

    return drift_detected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True, help="Path to current batch CSV")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_drift_report(
        reference_path=cfg["monitoring"]["reference_data_path"],
        current_path=args.current,
        output_path=cfg["monitoring"]["report_output_path"],
    )