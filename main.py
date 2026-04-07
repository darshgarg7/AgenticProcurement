import os
import numpy as np
import pandas as pd
from experiments.run_experiment import run_experiment

def generate_dataset(path, num_items=1000):
    print(f"Generating synthetic dataset at {path}...")
    np.random.seed(42)
    # Features: price (normalized), rating (1-5 scaled), quality score
    price = np.random.uniform(10, 1000, num_items)
    rating = np.random.uniform(1, 5, num_items)
    quality = np.random.normal(50, 15, num_items)
    
    category_raw = np.random.randint(0, 5, num_items)
    category = pd.get_dummies(category_raw, prefix='cat').values.astype(float) # one-hot

    # Scale them
    price_norm = price / 1000.0
    rating_norm = rating / 5.0
    quality_norm = quality / 100.0

    features = np.column_stack([price_norm, rating_norm, quality_norm, category])
    # d = 1 + 1 + 1 + 5 = 8
    
    cols = ['price_norm', 'rating_norm', 'quality_norm'] + [f'cat_{i}' for i in range(5)]
    df = pd.DataFrame(features, columns=cols)
    df.index.name = 'item_id'
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print("Dataset generated.\n")

if __name__ == "__main__":
    DATA_PATH = "data/products.csv"
    if not os.path.exists(DATA_PATH):
        generate_dataset(DATA_PATH, num_items=500)
    
    # Run experiment pipeline
    run_experiment(DATA_PATH, num_episodes=50, d=8)
