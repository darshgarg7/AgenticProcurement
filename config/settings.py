from dataclasses import dataclass

@dataclass
class EngineConfig:
    eps_reg: float = 1.0
    eps_var: float = 0.8
    tau_util: float = 0.5
    num_samples: int = 100
    base_search_ig: float = 0.1

@dataclass
class EnvConfig:
    data_path: str = "data/products.csv"
    alpha: float = 0.02
    price_fluctuation: float = 0.05
    
@dataclass
class ModelConfig:
    sigma2: float = 0.05
