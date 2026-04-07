import numpy as np
import pandas as pd
from core.interfaces import IMarketEnvironment, Observation
from config.settings import EnvConfig

class StochasticMarket(IMarketEnvironment):
    def __init__(self, config: EnvConfig | None = None):
        self.config = config if config is not None else EnvConfig()
        self.items = pd.read_csv(self.config.data_path)
        self.available_indices = list(self.items.index)
        
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        feature_cols = [c for c in df.columns if c not in ['item_id', 'true_utility']]
        x = df[feature_cols].values.astype(np.float64)
        
        # Row-wise L2 norm
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        return x / norms

    def observe(self) -> Observation:
        if len(self.available_indices) == 0:
            return Observation(item_ids=[], features=np.empty((0, 0)))
            
        available_df = self.items.loc[self.available_indices].copy()
        item_ids = available_df.index.tolist()
        features = self._extract_features(available_df)
        return Observation(item_ids=item_ids, features=features)

    def step(self) -> None:
        if not self.available_indices:
            return
            
        # Items go out of stock randomly
        out_of_stock_probs = np.random.rand(len(self.available_indices))
        out_of_stock = out_of_stock_probs < self.config.alpha
        indices_to_remove = np.array(self.available_indices)[out_of_stock]
        for idx in indices_to_remove:
            self.available_indices.remove(idx)
            
        # Price fluctuates
        if 'price_norm' in self.items.columns and len(self.available_indices) > 0:
            fluctuations = np.random.normal(1.0, self.config.price_fluctuation, len(self.available_indices))
            self.items.loc[self.available_indices, 'price_norm'] *= fluctuations
            self.items.loc[self.available_indices, 'price_norm'] = np.maximum(
                self.items.loc[self.available_indices, 'price_norm'], 0.01)
