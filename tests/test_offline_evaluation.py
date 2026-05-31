import tempfile
import unittest
from pathlib import Path

import numpy as np

from config.settings import EngineConfig
from evaluation.offline import (
    LoggedProcurementDataset,
    evaluate_agent_on_snapshots,
    evaluate_logged_choices,
)
from models.bayesian_user import BayesianPreferenceModel


class TestOfflineEvaluation(unittest.TestCase):
    def _write_dataset(self, directory: str) -> Path:
        path = Path(directory) / "logged_procurement.csv"
        path.write_text(
            "\n".join(
                [
                    "request_id,item_id,chosen_item_id,realized_utility,price_norm,quality_norm",
                    "r1,a,b,0.1,1.0,0.0",
                    "r1,b,b,1.0,0.0,1.0",
                    "r2,c,c,0.7,0.2,0.8",
                    "r2,d,c,0.3,0.8,0.2",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def test_loads_logged_procurement_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = LoggedProcurementDataset.from_csv(self._write_dataset(directory))

        self.assertEqual(len(dataset.snapshots), 2)
        self.assertEqual(dataset.feature_columns, ["price_norm", "quality_norm"])
        self.assertEqual(dataset.snapshots[0].logged_choice_item_id, "b")

    def test_evaluates_logged_choice_regret(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = LoggedProcurementDataset.from_csv(self._write_dataset(directory))
            metrics = evaluate_logged_choices(dataset)

        self.assertEqual(metrics.num_snapshots, 2)
        self.assertEqual(metrics.purchases, 2)
        self.assertAlmostEqual(metrics.avg_regret, 0.0)

    def test_replays_agent_on_logged_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = LoggedProcurementDataset.from_csv(self._write_dataset(directory))

        model = BayesianPreferenceModel(
            d=2,
            m0=np.array([-0.5, 1.0]),
            S0=np.eye(2) * 0.001,
        )
        config = EngineConfig(
            eps_reg=1.0,
            eps_var=1.0,
            tau_util=-1.0,
            regret_method="ellipsoid",
        )
        metrics = evaluate_agent_on_snapshots(dataset, model, config)

        self.assertEqual(metrics.num_snapshots, 2)
        self.assertEqual(metrics.purchases, 2)
        self.assertEqual(metrics.abstentions, 0)
        self.assertLessEqual(metrics.max_regret, 0.4)


if __name__ == "__main__":
    unittest.main()
