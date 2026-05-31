"""Run policy replay on a logged procurement CSV dataset."""

from __future__ import annotations

import argparse
import json

import numpy as np

from config.settings import EngineConfig
from evaluation.offline import (
    LoggedProcurementDataset,
    evaluate_agent_on_snapshots,
    evaluate_logged_choices,
)
from models.bayesian_user import BayesianPreferenceModel


def main() -> None:
    """Load logged snapshots, replay the policy, and print JSON metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to logged procurement candidates CSV")
    parser.add_argument("--eps-reg", type=float, default=0.8)
    parser.add_argument("--eps-var", type=float, default=0.8)
    parser.add_argument("--confidence-percentile", type=float, default=95.0)
    parser.add_argument("--regret-method", choices=["ellipsoid", "sampled"], default="ellipsoid")
    args = parser.parse_args()

    dataset = LoggedProcurementDataset.from_csv(args.csv_path)
    d = len(dataset.feature_columns)
    model = BayesianPreferenceModel(d=d, m0=np.zeros(d), S0=np.eye(d))
    config = EngineConfig(
        eps_reg=args.eps_reg,
        eps_var=args.eps_var,
        tau_util=-float("inf"),
        confidence_percentile=args.confidence_percentile,
        regret_method=args.regret_method,
    )

    payload = {
        "source_path": dataset.source_path,
        "feature_columns": dataset.feature_columns,
        "logged_policy": evaluate_logged_choices(dataset).to_dict(),
        "agent_policy": evaluate_agent_on_snapshots(dataset, model, config).to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
