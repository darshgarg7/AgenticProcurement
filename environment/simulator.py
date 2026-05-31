"""Stochastic market simulator and lightweight product catalog."""

import csv
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import EnvConfig
from core.interfaces import IMarketEnvironment, Observation


class ProductCatalog:
    """Small numeric CSV table tailored to the simulator's access patterns."""

    def __init__(self, records: list[dict[str, float | int]], columns: list[str]):
        """Create a catalog from parsed CSV records."""
        if not records:
            raise ValueError("product catalog must contain at least one item")
        if "item_id" not in columns:
            raise ValueError("product catalog must include an item_id column")
        self.records = records
        self.columns = columns
        self.index = list(range(len(records)))
        self.feature_columns = [c for c in columns if c not in ["item_id", "true_utility"]]
        if not self.feature_columns:
            raise ValueError("product catalog must include at least one feature column")
        self._id_to_pos = {int(row["item_id"]): pos for pos, row in enumerate(records)}

    @classmethod
    def from_csv(cls, path: str | Path) -> "ProductCatalog":
        """Load a numeric product catalog from disk."""
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path} is missing a CSV header")
            records: list[dict[str, float | int]] = []
            for raw in reader:
                row: dict[str, float | int] = {}
                for key, value in raw.items():
                    if key == "item_id":
                        row[key] = int(value)
                    else:
                        row[key] = float(value)
                records.append(row)
        return cls(records, list(reader.fieldnames))

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "ProductCatalog":
        """Build a catalog from already-decoded numeric records."""
        if not records:
            raise ValueError("product catalog must contain at least one item")

        columns = list(records[0].keys())
        normalized: list[dict[str, float | int]] = []
        for raw in records:
            if set(raw.keys()) != set(columns):
                raise ValueError("all product records must share the same columns")
            row: dict[str, float | int] = {}
            for key, value in raw.items():
                if key == "item_id":
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            normalized.append(row)
        return cls(normalized, columns)

    def __len__(self) -> int:
        """Return the number of catalog records."""
        return len(self.records)

    def __getitem__(self, column: str) -> np.ndarray:
        """Return a numeric column as a NumPy array."""
        return np.array([row[column] for row in self.records], dtype=np.float64)

    def row_by_item_id(self, item_id: int) -> dict[str, float | int]:
        """Return one row by its external item identifier."""
        return self.records[self._id_to_pos[int(item_id)]]

    def feature_matrix(self, indices: list[int]) -> np.ndarray:
        """Return feature rows for internal catalog indices."""
        return np.array(
            [[self.records[idx][col] for col in self.feature_columns] for idx in indices],
            dtype=np.float64,
        )

    def item_ids(self, indices: list[int]) -> list[int]:
        """Return external item identifiers for internal catalog indices."""
        return [int(self.records[idx]["item_id"]) for idx in indices]

    def update_column(self, column: str, indices: list[int], values: np.ndarray) -> None:
        """Update one numeric column for the selected internal indices."""
        for idx, value in zip(indices, values, strict=False):
            self.records[idx][column] = float(value)


class StochasticMarket(IMarketEnvironment):
    """Market with random stock-outs and price perturbations."""

    def __init__(self, config: EnvConfig | None = None, rng: Any | None = None):
        """Load the catalog and initialize all items as available."""
        self.config = config if config is not None else EnvConfig()
        self.rng = np.random if rng is None else rng
        self.items = ProductCatalog.from_csv(self.config.data_path)
        self.feature_columns = self.items.feature_columns
        self.available_indices = list(self.items.index)
        
    def _normalize_features(self, x: np.ndarray) -> np.ndarray:
        """L2-normalize feature rows while preserving zero rows."""
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    def observe(self) -> Observation:
        """Return the currently available item ids and normalized features."""
        if len(self.available_indices) == 0:
            return Observation(
                item_ids=[],
                features=np.empty((0, len(self.feature_columns)), dtype=np.float64),
            )
            
        item_ids = self.items.item_ids(self.available_indices)
        features = self._normalize_features(self.items.feature_matrix(self.available_indices))
        return Observation(item_ids=item_ids, features=features)

    def step(self) -> None:
        """Advance the market by applying stock-out and price dynamics."""
        if not self.available_indices:
            return
            
        out_of_stock_probs = self.rng.uniform(size=len(self.available_indices))
        out_of_stock = out_of_stock_probs < self.config.alpha
        self.available_indices = [
            idx for idx, should_remove in zip(self.available_indices, out_of_stock, strict=False)
            if not should_remove
        ]
            
        if "price_norm" in self.feature_columns and len(self.available_indices) > 0:
            current_prices = self.items["price_norm"][self.available_indices]
            fluctuations = self.rng.normal(
                loc=1.0,
                scale=self.config.price_fluctuation,
                size=len(self.available_indices),
            )
            next_prices = np.maximum(current_prices * fluctuations, 0.01)
            self.items.update_column("price_norm", self.available_indices, next_prices)
