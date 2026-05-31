"""
Flask backend for the Delegated Procurement web demo.

Endpoints:
  GET  /                   → serves the main page
  GET  /api/results        → full experiment results JSON
  GET  /api/figures/<name> → serves figure PNGs
  POST /api/run-episode    → runs a live episode, returns step-by-step data with product info
"""

import os
import secrets
import sys
import uuid

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json

import numpy as np
from flask import (
    Flask,
    Response,
    g,
    jsonify,
    request,
    send_file,
    send_from_directory,
    stream_with_context,
)

from config.settings import EngineConfig, EnvConfig, PersonaConfig
from core.episode import make_episode_context, step_agent_episode
from environment.simulator import ProductCatalog
from web_demo.product_emojis import PRODUCT_EMOJI_MAP

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app = Flask(__name__, static_folder=static_dir)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("PROCUREMENT_MAX_CONTENT_LENGTH", 1_048_576))

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'products.csv')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'full_experiment_results.json')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
API_KEY_ENV = "PROCUREMENT_API_KEY"
PUBLIC_ENDPOINTS = {"index", "static", "serve_static", "healthz", "readyz"}


@app.before_request
def attach_request_id():
    """Attach a stable request id for logs and downstream tracing."""
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))


@app.before_request
def require_api_key():
    """Require an API key for API routes when production auth is configured."""
    expected = os.environ.get(API_KEY_ENV)
    if not expected or request.endpoint in PUBLIC_ENDPOINTS:
        return None

    supplied = request.headers.get("X-API-Key") or request.args.get("api_key", "")
    if not secrets.compare_digest(supplied, expected):
        return jsonify({"error": "Unauthorized", "request_id": g.request_id}), 401
    return None


@app.after_request
def add_operational_headers(response):
    """Expose request metadata useful for production logs and incident triage."""
    response.headers["X-Request-ID"] = g.get("request_id", str(uuid.uuid4()))
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response

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


def product_from_item_id(item_id, catalog):
    """Build demo product metadata for one catalog item."""
    row = catalog.row_by_item_id(item_id)
    cat_idx = 0
    for c in range(5):
        if row[f'cat_{c}'] == 1.0:
            cat_idx = c
            break
    return get_product_info(
        item_id, row['price_norm'], row['rating_norm'], row['quality_norm'], cat_idx
    )


# ── Load raw product data ───────────────────────────────────────────────────

RAW_DF = ProductCatalog.from_csv(DATA_PATH)


# ── Pages ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the demo shell."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/healthz')
def healthz():
    """Return a lightweight liveness signal."""
    return jsonify({"status": "ok", "service": "agentic-procurement"})


@app.route('/readyz')
def readyz():
    """Return readiness based on required local runtime dependencies."""
    checks = {
        "catalog_exists": os.path.exists(DATA_PATH),
        "catalog_loaded": len(RAW_DF) > 0,
    }
    status = 200 if all(checks.values()) else 503
    return jsonify({"status": "ready" if status == 200 else "not_ready", "checks": checks}), status


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static demo assets."""
    return send_from_directory(app.static_folder, path)


# ── API: Search products ─────────────────────────────────────────────────────

@app.route('/api/search')
def search_products():
    """Return up to 12 product-display records matching a query."""
    q = request.args.get('q', '').strip().lower()
    if len(q) < 1:
        return jsonify([])
    results = []
    for row in RAW_DF.records:
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
    """Return precomputed experiment results for the dashboard."""
    if not os.path.exists(RESULTS_PATH):
        return jsonify({'error': 'No experiment results found'}), 404
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return jsonify(data)


@app.route('/api/figures/<name>')
def get_figure(name):
    """Serve a generated experiment figure by file name."""
    if not name.endswith('.png'):
        return jsonify({'error': 'Invalid file type'}), 400
    safe_name = os.path.basename(name)
    fpath = os.path.join(FIG_DIR, safe_name)
    if not os.path.exists(fpath):
        return jsonify({'error': 'Figure not found'}), 404
    return send_file(fpath, mimetype='image/png')


# ── API: Run live episode ───────────────────────────────────────────────────

def _episode_params(values):
    """Parse, validate, and clamp episode request parameters."""
    try:
        persona_choice = values.get('persona', 'balanced')
        eps_reg = float(values.get('eps_reg', 0.8))
        eps_var = float(values.get('eps_var', eps_reg))
        max_steps = int(values.get('max_steps', 30))
        seed = int(values.get('seed', 42))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid episode parameter") from exc

    return {
        'persona': persona_choice,
        'eps_reg': max(0.1, min(eps_reg, 5.0)),
        'eps_var': max(0.1, min(eps_var, 5.0)),
        'max_steps': max(1, min(max_steps, 100)),
        'seed': max(0, min(seed, 99999)),
    }


def _initialize_episode(params):
    """Create a seeded episode context from request parameters."""
    d = 8
    rng = np.random.RandomState(params['seed'])

    if params['persona'] in ('budget_shopper', 'quality_maximizer', 'balanced'):
        factory = getattr(PersonaConfig, params['persona'])
        persona = factory(d=d, seed=params['seed'])
        true_theta = persona.true_theta
        prior_m0 = persona.prior_mean
        prior_S0 = persona.prior_cov
    else:
        true_theta = rng.randn(d)
        true_theta /= np.linalg.norm(true_theta)
        prior_m0 = true_theta + rng.normal(0, 0.4, d)
        prior_S0 = np.eye(d) * 0.5

    engine_config = EngineConfig(
        eps_reg=params['eps_reg'],
        eps_var=params['eps_var'],
        tau_util=0.0,
    )
    return make_episode_context(
        true_theta=true_theta,
        data_path=DATA_PATH,
        prior_m0=prior_m0,
        prior_S0=prior_S0,
        engine_config=engine_config,
        env_config=EnvConfig(data_path=DATA_PATH),
        rng=rng,
    )


def _round_or_none(value, digits=4):
    """Round numeric values while preserving nulls for JSON output."""
    return None if value is None else round(float(value), digits)


def iter_episode_events(params):
    """Yield step and completion events for one live demo episode."""
    context = _initialize_episode(params)
    steps = []
    epi_unc_history = []
    action_counts = {"Purchase": 0, "QueryUser": 0, "Wait": 0, "Search": 0}
    outcome = {"purchased": False, "steps": params['max_steps']}

    for step_index in range(params['max_steps']):
        step = step_agent_episode(context, step_index, top_k=5)
        step_payload = _demo_payload_from_step(step, context.env.items)
        steps.append(step_payload)
        action_counts[step.action] += 1
        if step.epi_unc is not None:
            epi_unc_history.append(float(step.epi_unc))
        yield "step", step_payload

        if step.action == "Purchase":
            target = step_payload["target_item"]
            outcome = {
                "purchased": True,
                "item": target,
                "realized_regret": _round_or_none(step.realized_regret),
                "estimated_wc_regret": _round_or_none(step.estimated_wc_regret),
                "steps": step.step,
            }
            break
    else:
        outcome = {"purchased": False, "steps": len(steps)}

    yield "complete", {
        "outcome": outcome,
        "steps": steps,
        "epi_unc_history": epi_unc_history,
        "action_counts": action_counts,
    }


def _demo_payload_from_step(step, catalog):
    """Convert an internal episode step into the browser payload shape."""
    top_products = [product_from_item_id(item_id, catalog) for item_id in step.top_item_ids]
    target = product_from_item_id(step.item_id, catalog) if step.item_id is not None else None
    reason = _reason_for_step(step, target)
    return {
        "step": step.step,
        "action": step.action,
        "reason": reason,
        "epi_unc": _round_or_none(step.epi_unc),
        "exp_util": _round_or_none(step.exp_util),
        "estimated_wc_regret": _round_or_none(step.estimated_wc_regret),
        "products": top_products,
        "target_item": target,
    }


def _reason_for_step(step, target):
    """Create a concise user-facing explanation for a demo step."""
    if step.action == "Purchase":
        name = target["name"] if target else "selected item"
        return f"Purchased {name} (est. regret={step.estimated_wc_regret:.3f})"
    if step.action == "QueryUser":
        name = target["name"] if target else "selected item"
        return f"Asking user about {name}"
    if step.action == "Wait":
        if step.wait_advantage is None:
            return "Waiting for better deals"
        return f"Waiting for better deals (wait advantage={step.wait_advantage:.3f})"
    return "No items available — searching for new products" if not step.top_item_ids else "Browsing for more options"


@app.route('/api/run-episode', methods=['POST'])
def run_episode():
    """Run an episode eagerly and return the full JSON transcript."""
    try:
        params = _episode_params(request.get_json(force=True))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    payload = None
    for event_name, event_payload in iter_episode_events(params):
        if event_name == "complete":
            payload = event_payload
    return jsonify(payload)


@app.route('/api/run-episode-stream')
def run_episode_stream():
    """Stream an episode as Server-Sent Events."""
    try:
        params = _episode_params(request.args)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    @stream_with_context
    def generate():
        for event_name, event_payload in iter_episode_events(params):
            yield f"event: {event_name}\n"
            yield f"data: {json.dumps(event_payload)}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == '__main__':
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(debug=debug, port=5001)
