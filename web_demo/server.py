"""
Flask backend for the Delegated Procurement web demo.

Endpoints:
  GET  /                   → serves the main page
  GET  /api/results        → full experiment results JSON
  GET  /api/figures/<name> → serves figure PNGs
  POST /api/run-episode    → runs a live episode, returns step-by-step data with product info
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, send_file

from environment.simulator import StochasticMarket
from decision.delegation_engine import DelegationEngine
from models.bayesian_user import BayesianPreferenceModel
from core.interfaces import Purchase, QueryUser, Wait, Search
from config.settings import EngineConfig, EnvConfig, ModelConfig, PersonaConfig
from web_demo.product_emojis import PRODUCT_EMOJI_MAP

app = Flask(__name__, static_folder='static')

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'products.csv')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'full_experiment_results.json')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')

# ── Product catalog for display ─────────────────────────────────────────────

CATEGORY_NAMES = ['Electronics', 'Home & Kitchen', 'Sports & Outdoors', 'Fashion', 'Books & Media']
CATEGORY_EMOJI = ['💻', '🏠', '⚽', '👕', '📚']

PRODUCT_ADJECTIVES = [
    ['Premium', 'Compact', 'Ultra', 'Smart', 'Wireless', 'Pro', 'Mini', 'Elite'],
    ['Elegant', 'Modern', 'Classic', 'Deluxe', 'Ceramic', 'Stainless', 'Bamboo', 'Copper'],
    ['Performance', 'Lightweight', 'Heavy-Duty', 'All-Weather', 'Carbon', 'Flex', 'Aero', 'Trail'],
    ['Designer', 'Vintage', 'Casual', 'Slim-Fit', 'Organic', 'Merino', 'Stretch', 'Heritage'],
    ['Bestselling', 'Illustrated', 'Collector', 'Limited', 'Essential', 'Complete', 'Pocket', 'Deluxe'],
]

PRODUCT_NOUNS = [
    ['Headphones', 'Speaker', 'Keyboard', 'Monitor', 'Charger', 'Webcam', 'Mouse', 'Tablet',
     'Smartwatch', 'Power Bank', 'USB Hub', 'Microphone'],
    ['Coffee Maker', 'Blender', 'Toaster', 'Knife Set', 'Dutch Oven', 'Air Fryer', 'Tea Kettle',
     'Cutting Board', 'Spice Rack', 'Water Filter', 'Dish Set', 'Candle Set'],
    ['Running Shoes', 'Yoga Mat', 'Dumbbell Set', 'Hiking Pack', 'Water Bottle', 'Bike Light',
     'Tennis Racket', 'Jump Rope', 'Resistance Bands', 'Foam Roller', 'Climbing Gear', 'Swim Goggles'],
    ['Leather Jacket', 'Sneakers', 'Denim Jeans', 'Wool Sweater', 'Sunglasses', 'Watch',
     'Backpack', 'Scarf', 'Belt', 'Boots', 'Hoodie', 'Polo Shirt'],
    ['Novel', 'Cookbook', 'Art Book', 'Biography', 'Guide', 'Anthology', 'Journal',
     'Atlas', 'Manual', 'Encyclopedia', 'Graphic Novel', 'Planner'],
]


def get_product_info(item_id, price_norm, rating_norm, quality_norm, cat_idx):
    """Generate realistic product display info from normalized features."""
    adj_list = PRODUCT_ADJECTIVES[cat_idx]
    noun_list = PRODUCT_NOUNS[cat_idx]
    adj = adj_list[item_id % len(adj_list)]
    noun = noun_list[item_id % len(noun_list)]
    name = f"{adj} {noun}"

    price = 9.99 + price_norm * 240  # $9.99 – $249.99
    rating = 1.0 + rating_norm * 4.0  # 1.0 – 5.0
    quality_pct = int(quality_norm * 100)

    return {
        'id': int(item_id),
        'name': name,
        'category': CATEGORY_NAMES[cat_idx],
        'emoji': PRODUCT_EMOJI_MAP.get(noun, CATEGORY_EMOJI[cat_idx]),
        'price': round(price, 2),
        'rating': round(rating, 1),
        'quality': quality_pct,
        'price_norm': round(float(price_norm), 3),
        'rating_norm': round(float(rating_norm), 3),
        'quality_norm': round(float(quality_norm), 3),
    }


def items_from_obs(obs, raw_df):
    """Build product info list from an Observation."""
    products = []
    for i, item_id in enumerate(obs.item_ids):
        row = raw_df.loc[raw_df['item_id'] == item_id].iloc[0]
        cat_idx = 0
        for c in range(5):
            if row[f'cat_{c}'] == 1.0:
                cat_idx = c
                break
        products.append(get_product_info(
            item_id, row['price_norm'], row['rating_norm'], row['quality_norm'], cat_idx
        ))
    return products


# ── Load raw product data ───────────────────────────────────────────────────

RAW_DF = pd.read_csv(DATA_PATH)


# ── Pages ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


# ── API: Search products ─────────────────────────────────────────────────────

@app.route('/api/search')
def search_products():
    q = request.args.get('q', '').strip().lower()
    if len(q) < 1:
        return jsonify([])
    results = []
    for _, row in RAW_DF.iterrows():
        cat_idx = 0
        for c in range(5):
            if row[f'cat_{c}'] == 1.0:
                cat_idx = c
                break
        info = get_product_info(
            int(row['item_id']), row['price_norm'], row['rating_norm'],
            row['quality_norm'], cat_idx
        )
        if q in info['name'].lower() or q in info['category'].lower():
            results.append(info)
        if len(results) >= 12:
            break
    return jsonify(results)


# ── API: Experiment results ─────────────────────────────────────────────────

@app.route('/api/results')
def get_results():
    if not os.path.exists(RESULTS_PATH):
        return jsonify({'error': 'No experiment results found'}), 404
    with open(RESULTS_PATH, 'r') as f:
        data = json.load(f)
    return jsonify(data)


@app.route('/api/figures/<name>')
def get_figure(name):
    if not name.endswith('.png'):
        return jsonify({'error': 'Invalid file type'}), 400
    safe_name = os.path.basename(name)
    fpath = os.path.join(FIG_DIR, safe_name)
    if not os.path.exists(fpath):
        return jsonify({'error': 'Figure not found'}), 404
    return send_file(fpath, mimetype='image/png')


# ── API: Run live episode ───────────────────────────────────────────────────

@app.route('/api/run-episode', methods=['POST'])
def run_episode():
    body = request.get_json(force=True)
    persona_choice = body.get('persona', 'balanced')
    eps_reg = float(body.get('eps_reg', 0.8))
    eps_var = float(body.get('eps_var', 0.8))
    max_steps = int(body.get('max_steps', 30))
    seed = int(body.get('seed', 42))

    # Clamp
    eps_reg = max(0.1, min(eps_reg, 5.0))
    eps_var = max(0.1, min(eps_var, 5.0))
    max_steps = max(1, min(max_steps, 100))
    seed = max(0, min(seed, 99999))

    d = 8
    rng = np.random.RandomState(seed)

    if persona_choice in ('budget_shopper', 'quality_maximizer', 'balanced'):
        factory = getattr(PersonaConfig, persona_choice)
        persona = factory(d=d, seed=seed)
        true_theta = persona.true_theta
        prior_m0 = persona.prior_mean
        prior_S0 = persona.prior_cov
    else:
        true_theta = rng.randn(d)
        true_theta /= np.linalg.norm(true_theta)
        prior_m0 = true_theta + rng.normal(0, 0.4, d)
        prior_S0 = np.eye(d) * 0.5

    engine_config = EngineConfig(eps_reg=eps_reg, eps_var=eps_var, tau_util=0.0)
    model_config = ModelConfig(sigma2=0.05)
    model = BayesianPreferenceModel(d=d, m0=prior_m0, S0=prior_S0, config=model_config)
    engine = DelegationEngine(model, config=engine_config)
    env = StochasticMarket(config=EnvConfig(data_path=DATA_PATH))

    steps = []
    epi_unc_history = []
    action_counts = {"Purchase": 0, "QueryUser": 0, "Wait": 0, "Search": 0}
    outcome = {"purchased": False, "steps": max_steps}

    for step in range(max_steps):
        obs = env.observe()

        if obs.features.shape[0] == 0:
            steps.append({
                "step": step + 1, "action": "Search",
                "reason": "No items available — searching for new products",
                "epi_unc": None, "exp_util": None,
                "products": [], "target_item": None,
            })
            action_counts["Search"] += 1
            env.step()
            continue

        products = items_from_obs(obs, RAW_DF)
        epi_unc = model.epistemic_uncertainty(obs.features)
        exp_util = model.expected_utility(obs.features)
        best_idx = int(np.argmax(exp_util))
        epi_unc_history.append(float(epi_unc[best_idx]))

        # Top-5 products by expected utility for display
        top_indices = np.argsort(-exp_util)[:5]
        top_products = [products[i] for i in top_indices]

        action = engine.decide(obs)

        if isinstance(action, Purchase):
            idx = obs.item_ids.index(action.item_id)
            utils = obs.features @ true_theta
            best_u = float(np.max(utils))
            chosen_u = float(utils[idx])
            realized_regret = best_u - chosen_u
            target = products[idx]

            steps.append({
                "step": step + 1, "action": "Purchase",
                "reason": f"Purchased {target['name']}",
                "epi_unc": round(float(epi_unc[best_idx]), 4),
                "exp_util": round(float(exp_util[best_idx]), 4),
                "products": top_products, "target_item": target,
            })
            action_counts["Purchase"] += 1
            outcome = {
                "purchased": True,
                "item": target,
                "realized_regret": round(realized_regret, 4),
                "steps": step + 1,
            }
            break

        elif isinstance(action, QueryUser):
            idx = obs.item_ids.index(action.item_id)
            x = obs.features[idx]
            y = float(true_theta @ x) + np.random.normal(0, np.sqrt(model_config.sigma2))
            model.update(x, y)
            target = products[idx]

            steps.append({
                "step": step + 1, "action": "QueryUser",
                "reason": f"Asking user about {target['name']}",
                "epi_unc": round(float(epi_unc[best_idx]), 4),
                "exp_util": round(float(exp_util[best_idx]), 4),
                "products": top_products, "target_item": target,
            })
            action_counts["QueryUser"] += 1

        elif isinstance(action, Wait):
            env.step()
            steps.append({
                "step": step + 1, "action": "Wait",
                "reason": "Waiting for better deals",
                "epi_unc": round(float(epi_unc[best_idx]), 4),
                "exp_util": round(float(exp_util[best_idx]), 4),
                "products": top_products, "target_item": None,
            })
            action_counts["Wait"] += 1

        else:
            env.step()
            steps.append({
                "step": step + 1, "action": "Search",
                "reason": "Browsing for more options",
                "epi_unc": round(float(epi_unc[best_idx]), 4),
                "exp_util": round(float(exp_util[best_idx]), 4),
                "products": top_products, "target_item": None,
            })
            action_counts["Search"] += 1

    return jsonify({
        "outcome": outcome,
        "steps": steps,
        "epi_unc_history": epi_unc_history,
        "action_counts": action_counts,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001)
