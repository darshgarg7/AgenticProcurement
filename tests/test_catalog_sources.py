import json
import tempfile
import unittest
from pathlib import Path

from environment.catalog_sources import (
    CsvCatalogSource,
    JsonCatalogSource,
    load_catalog_with_report,
)


class TestCatalogSources(unittest.TestCase):
    def test_csv_catalog_source_reports_loaded_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.csv"
            path.write_text(
                "\n".join(
                    [
                        "item_id,price_norm,quality_norm",
                        "1,0.2,0.8",
                        "2,0.5,0.5",
                    ]
                ),
                encoding="utf-8",
            )
            catalog, report = load_catalog_with_report(CsvCatalogSource(path))

        self.assertEqual(len(catalog), 2)
        self.assertEqual(report.num_items, 2)
        self.assertEqual(report.feature_columns, ["price_norm", "quality_norm"])

    def test_json_catalog_source_accepts_records_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(
                json.dumps(
                    {
                        "records": [
                            {"item_id": 1, "price_norm": 0.2, "quality_norm": 0.8},
                            {"item_id": 2, "price_norm": 0.5, "quality_norm": 0.5},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            catalog = JsonCatalogSource(path.as_uri()).load()

        self.assertEqual(len(catalog), 2)
        self.assertEqual(catalog.feature_columns, ["price_norm", "quality_norm"])


if __name__ == "__main__":
    unittest.main()
