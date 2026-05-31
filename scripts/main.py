"""Entry point for dataset generation and the compact experiment runner."""

import csv
import os

import numpy as np

from experiments.run_experiment import run_experiment


def generate_dataset(path, num_items=1000):
    """Generate the synthetic product CSV used by the simulator."""
    print(f"Generating synthetic dataset at {path}...")
    np.random.seed(42)
    # Features: price (normalized), rating (1-5 scaled), quality score
    price = np.random.uniform(10, 1000, num_items)
    rating = np.random.uniform(1, 5, num_items)
    quality = np.random.normal(50, 15, num_items)
    
    category_raw = np.random.randint(0, 5, num_items)
    category = np.eye(5, dtype=float)[category_raw]

    # Scale them
    price_norm = price / 1000.0
    rating_norm = rating / 5.0
    quality_norm = quality / 100.0

    features = np.column_stack([price_norm, rating_norm, quality_norm, category])
    # d = 1 + 1 + 1 + 5 = 8
    
    cols = ['price_norm', 'rating_norm', 'quality_norm'] + [f'cat_{i}' for i in range(5)]
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['item_id', *cols])
        for item_id, row in enumerate(features):
            writer.writerow([item_id, *row])
    print("Dataset generated.\n")

if __name__ == "__main__":
    DATA_PATH = "data/products.csv"
    if not os.path.exists(DATA_PATH):
        generate_dataset(DATA_PATH, num_items=500)
    
    # Run experiment pipeline
    run_experiment(DATA_PATH, num_episodes=50, d=8)
