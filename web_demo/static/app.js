/* ─── Agentic Procurement — E-Commerce Demo Frontend ──────────────── */

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// DOM refs
const runBtn = $('#run-btn');
const personaSelect = $('#persona-select');
const epsRegSlider = $('#eps-reg');
const epsRegVal = $('#eps-reg-val');
const speedSelect = $('#speed-select');
const seedInput = $('#seed-input');
const maxStepsInput = $('#max-steps-input');
const statusStrip = $('#status-strip');
const statusStep = $('#status-step');
const statusAction = $('#status-action');
const statusReason = $('#status-reason');
const statusUnc = $('#status-unc');
const progressBar = $('#progress-bar');
const productGrid = $('#product-grid');
const cartItems = $('#cart-items');
const cartTotal = $('#cart-total');
const cartTotalPrice = $('#cart-total-price');
const checkoutBtn = $('#checkout-btn');
const cartBadge = $('#cart-badge');
const cartIconWrap = $('#cart-btn');
const logEntries = $('#log-entries');
const resultOverlay = $('#result-overlay');
const resultIcon = $('#result-icon');
const resultTitle = $('#result-title');
const resultDetails = $('#result-details');
const closeOverlay = $('#close-overlay');
const analysisSection = $('#analysis-section');
const runAgainBtn = $('#run-again-btn');
const searchInput = $('#search-input');
const searchDropdown = $('#search-dropdown');

let isRunning = false;
let lastEpisodeData = null; // store for analysis
let searchTimeout = null;

// ─── Category → SVG icon mapping ───────────────────────────
const CATEGORY_ICONS = {
  'Electronics': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>',
  'Home & Kitchen': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
  'Sports & Outdoors': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 000 20 14.5 14.5 0 000-20"/><line x1="2" y1="12" x2="22" y2="12"/></svg>',
  'Fashion': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20.38 3.46L16 2 12 5 8 2 3.62 3.46a2 2 0 00-1.34 2.23l.58 3.47a1 1 0 00.99.84H6v10c0 1.1.9 2 2 2h8a2 2 0 002-2V10h2.15a1 1 0 00.99-.84l.58-3.47a2 2 0 00-1.34-2.23z"/></svg>',
  'Books & Media': '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>',
};

const CATEGORY_CSS_CLASS = {
  'Electronics': 'cat-electronics',
  'Home & Kitchen': 'cat-home',
  'Sports & Outdoors': 'cat-sports',
  'Fashion': 'cat-fashion',
  'Books & Media': 'cat-books',
};

// SVG icon strings for overlay
const SVG_CHECK_CIRCLE = '<svg viewBox="0 0 24 24" fill="none" stroke="#067d62" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>';
const SVG_CLOCK = '<svg viewBox="0 0 24 24" fill="none" stroke="#c45500" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>';
const SVG_SEARCH_SM = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="11" cy="11" r="7"/><line x1="20" y1="20" x2="16" y2="16"/></svg>';
const SVG_CHECK_SM = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>';
const SVG_STAR_SM = '<svg width="12" height="12" viewBox="0 0 24 24" fill="#ff9900" stroke="#ff9900" stroke-width="1"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>';

// ─── Slider update ─────────────────────────────────────────
epsRegSlider.addEventListener('input', () => {
  epsRegVal.textContent = epsRegSlider.value;
});

// ─── Close overlay ─────────────────────────────────────────
closeOverlay.addEventListener('click', () => {
  resultOverlay.style.display = 'none';
});

// ─── Star rendering ────────────────────────────────────────
function renderStars(rating) {
  const full = Math.floor(rating);
  const half = rating - full >= 0.5 ? 1 : 0;
  const empty = 5 - full - half;
  return '★'.repeat(full) + (half ? '½' : '') + '☆'.repeat(empty);
}

// ─── Build product card HTML ───────────────────────────────
function makeCard(product, tag) {
  const card = document.createElement('div');
  card.className = 'product-card';
  card.dataset.id = product.id;

  let tagHTML = '';
  if (tag === 'best') tagHTML = `<div class="card-action-tag best">${SVG_STAR_SM} Top Pick</div>`;
  if (tag === 'query') tagHTML = `<div class="card-action-tag query">${SVG_SEARCH_SM} Evaluating</div>`;
  if (tag === 'purchase') tagHTML = `<div class="card-action-tag purchase">${SVG_CHECK_SM} Selected</div>`;

  const catClass = CATEGORY_CSS_CLASS[product.category] || '';
  const catIcon = CATEGORY_ICONS[product.category] || '';

  card.innerHTML = `
    ${tagHTML}
    <div class="card-img ${catClass}">${product.emoji}</div>
    <div class="card-body">
      <div class="card-cat">${product.category}</div>
      <div class="card-name">${product.name}</div>
      <div class="card-rating">
        <span class="card-stars">${renderStars(product.rating)}</span>
        <span class="card-rating-num">${product.rating}</span>
      </div>
      <div class="card-quality">Quality: ${product.quality}%</div>
      <div class="card-price-row">
        <span class="card-price">$${product.price.toFixed(2)}</span>
        <span class="card-prime">FREE Delivery</span>
      </div>
    </div>
  `;
  return card;
}

// ─── Add log entry ─────────────────────────────────────────
function addLog(step, action, msg) {
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  entry.innerHTML = `
    <span class="log-step">${step}</span>
    <span class="log-action ${action}">${action}</span>
    <span class="log-msg">${msg}</span>
  `;
  logEntries.prepend(entry);
}

// ─── Fly to cart animation ─────────────────────────────────
function flyToCart(cardEl) {
  const rect = cardEl.getBoundingClientRect();
  const cartRect = cartIconWrap.getBoundingClientRect();

  const clone = document.createElement('div');
  clone.className = 'fly-clone';
  clone.style.left = rect.left + 'px';
  clone.style.top = rect.top + 'px';
  clone.style.width = rect.width + 'px';
  clone.style.height = rect.height + 'px';
  clone.style.background = '#fff';
  clone.innerHTML = cardEl.querySelector('.card-img').outerHTML;
  clone.querySelector('.card-img').style.height = '100%';
  document.body.appendChild(clone);

  // Calculate direction
  const dx = cartRect.left - rect.left;
  const dy = cartRect.top - rect.top;
  clone.style.setProperty('--dx', dx + 'px');
  clone.style.setProperty('--dy', dy + 'px');
  clone.style.animation = 'none';
  // Force reflow
  void clone.offsetWidth;
  clone.style.transition = 'all 0.6s cubic-bezier(0.2, 0.8, 0.2, 1)';
  clone.style.transform = `translate(${dx}px, ${dy}px) scale(0.15)`;
  clone.style.opacity = '0';

  setTimeout(() => {
    clone.remove();
    cartIconWrap.classList.add('cart-shake');
    setTimeout(() => cartIconWrap.classList.remove('cart-shake'), 400);
  }, 600);
}

// ─── Add item to cart sidebar ──────────────────────────────
function addToCart(product) {
  const empty = cartItems.querySelector('.cart-empty');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = 'cart-item';
  const miniIcon = CATEGORY_ICONS[product.category] || '';
  item.innerHTML = `
    <div class="cart-item-emoji">${miniIcon}</div>
    <div class="cart-item-info">
      <div class="cart-item-name">${product.name}</div>
      <div class="cart-item-price">$${product.price.toFixed(2)}</div>
    </div>
  `;
  cartItems.appendChild(item);

  cartBadge.textContent = '1';
  cartTotal.style.display = 'flex';
  cartTotalPrice.textContent = '$' + product.price.toFixed(2);
  checkoutBtn.style.display = 'block';
}

// ─── Animate checkout ──────────────────────────────────────
function animateCheckout() {
  return new Promise(resolve => {
    checkoutBtn.textContent = 'Processing...';
    setTimeout(() => {
      checkoutBtn.classList.add('success');
      checkoutBtn.innerHTML = `${SVG_CHECK_SM} Order Placed!`;
      setTimeout(resolve, 600);
    }, 800);
  });
}

// ─── Delay utility ─────────────────────────────────────────
function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ─── Reset UI ──────────────────────────────────────────────
function resetUI() {
  productGrid.innerHTML = '<div class="grid-placeholder"><p>Click <strong>▶ Start Shopping</strong> to watch the AI agent browse this store.</p></div>';
  cartItems.innerHTML = '<p class="cart-empty">Cart is empty</p>';
  cartTotal.style.display = 'none';
  checkoutBtn.style.display = 'none';
  checkoutBtn.classList.remove('success');
  checkoutBtn.innerHTML = '<svg class=\"btn-icon\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><polyline points=\"20 6 9 17 4 12\"/></svg> Checkout';
  cartBadge.textContent = '0';
  logEntries.innerHTML = '';
  statusStrip.style.display = 'none';
  progressBar.style.width = '0%';
  analysisSection.style.display = 'none';
}

// ─── Run Episode ───────────────────────────────────────────
async function runEpisode() {
  if (isRunning) return;
  isRunning = true;
  runBtn.disabled = true;
  runBtn.textContent = 'Running...';
  resetUI();

  const params = {
    persona: personaSelect.value,
    eps_reg: parseFloat(epsRegSlider.value),
    eps_var: parseFloat(epsRegSlider.value),
    max_steps: parseInt(maxStepsInput.value) || 30,
    seed: parseInt(seedInput.value),
  };

  const speed = parseInt(speedSelect.value);

  try {
    const resp = await fetch('/api/run-episode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const data = await resp.json();

    statusStrip.style.display = 'block';
    productGrid.innerHTML = '';

    const totalSteps = data.steps.length;

    // Animate through steps
    for (let i = 0; i < totalSteps; i++) {
      const step = data.steps[i];
      const pct = ((i + 1) / totalSteps) * 100;
      progressBar.style.width = pct + '%';

      // Update status bar
      statusStep.textContent = `${step.step} / ${totalSteps}`;
      statusAction.textContent = step.action;
      statusAction.style.color =
        step.action === 'Purchase' ? '#067d62' :
        step.action === 'QueryUser' ? '#0066c0' :
        step.action === 'Wait' ? '#c45500' : '#8779a9';
      statusReason.textContent = step.reason;
      statusUnc.textContent = step.epi_unc !== null ? step.epi_unc.toFixed(4) : '—';

      // Add to log
      addLog(step.step, step.action, step.reason);

      if (step.action === 'Search') {
        // Show products refreshing
        productGrid.innerHTML = '';
        if (step.products && step.products.length > 0) {
          step.products.forEach((p, idx) => {
            const tag = idx === 0 ? 'best' : null;
            const card = makeCard(p, tag);
            card.style.animationDelay = (idx * 80) + 'ms';
            productGrid.appendChild(card);
          });
        }
        await wait(speed);

      } else if (step.action === 'QueryUser') {
        // Show products, highlight the queried one
        productGrid.innerHTML = '';
        if (step.products) {
          step.products.forEach((p, idx) => {
            const isTarget = step.target_item && p.id === step.target_item.id;
            const tag = isTarget ? 'query' : (idx === 0 ? 'best' : null);
            const card = makeCard(p, tag);
            if (isTarget) card.classList.add('querying');
            else card.classList.add('dimmed');
            card.style.animationDelay = (idx * 60) + 'ms';
            productGrid.appendChild(card);
          });
        }
        await wait(speed * 1.2);
        // Remove dimming after query
        productGrid.querySelectorAll('.dimmed').forEach(c => c.classList.remove('dimmed'));
        productGrid.querySelectorAll('.querying').forEach(c => c.classList.remove('querying'));

      } else if (step.action === 'Wait') {
        // Show products, dim them slightly to indicate waiting
        productGrid.innerHTML = '';
        if (step.products) {
          step.products.forEach((p, idx) => {
            const tag = idx === 0 ? 'best' : null;
            const card = makeCard(p, tag);
            card.style.opacity = '0.6';
            productGrid.appendChild(card);
          });
        }
        await wait(speed * 0.8);
        // Restore opacity
        productGrid.querySelectorAll('.product-card').forEach(c => c.style.opacity = '1');

      } else if (step.action === 'Purchase') {
        // Show products, highlight purchased item, fly to cart
        productGrid.innerHTML = '';
        let purchasedCard = null;
        if (step.products) {
          step.products.forEach((p, idx) => {
            const isTarget = step.target_item && p.id === step.target_item.id;
            const tag = isTarget ? 'purchase' : null;
            const card = makeCard(p, tag);
            if (isTarget) {
              card.classList.add('highlight');
              purchasedCard = card;
            } else {
              card.classList.add('dimmed');
            }
            productGrid.appendChild(card);
          });
        }

        await wait(speed);

        // Fly-to-cart animation
        if (purchasedCard && step.target_item) {
          flyToCart(purchasedCard);
          await wait(700);
          addToCart(step.target_item);
          await wait(400);
          await animateCheckout();
          await wait(400);
        }

        // Show result overlay
        const o = data.outcome;
        resultIcon.innerHTML = o.purchased ? SVG_CHECK_CIRCLE : SVG_CLOCK;
        resultTitle.textContent = o.purchased ? 'Purchase Complete!' : 'No Purchase Made';

        let detailHTML = '';
        if (o.purchased && o.item) {
          detailHTML += `<div class="result-row"><span class="rk">Item</span><span class="rv">${o.item.name}</span></div>`;
          detailHTML += `<div class="result-row"><span class="rk">Price</span><span class="rv">$${o.item.price.toFixed(2)}</span></div>`;
          detailHTML += `<div class="result-row"><span class="rk">Rating</span><span class="rv">${o.item.rating} ★</span></div>`;
          detailHTML += `<div class="result-row"><span class="rk">Quality</span><span class="rv">${o.item.quality}%</span></div>`;
        }
        detailHTML += `<div class="result-row"><span class="rk">Steps Taken</span><span class="rv">${o.steps}</span></div>`;
        if (o.realized_regret !== undefined) {
          const regClass = o.realized_regret < 0.1 ? 'good' : 'bad';
          detailHTML += `<div class="result-row"><span class="rk">Realized Regret</span><span class="rv ${regClass}">${o.realized_regret.toFixed(4)}</span></div>`;
        }

        // Action breakdown
        detailHTML += `<div class="result-row"><span class="rk">Queries → User</span><span class="rv">${data.action_counts.QueryUser}</span></div>`;
        detailHTML += `<div class="result-row"><span class="rk">Wait Actions</span><span class="rv">${data.action_counts.Wait}</span></div>`;
        detailHTML += `<div class="result-row"><span class="rk">Searches</span><span class="rv">${data.action_counts.Search}</span></div>`;

        resultDetails.innerHTML = detailHTML;
        resultOverlay.style.display = 'flex';
      }
    }

    // Store full data for analysis
    lastEpisodeData = data;
    lastEpisodeData._params = params;

    // If no purchase was made, still show the result
    if (!data.outcome.purchased) {
      resultIcon.innerHTML = SVG_CLOCK;
      resultTitle.textContent = 'Agent Did Not Purchase';
      let detailHTML = `<div class="result-row"><span class="rk">Steps Taken</span><span class="rv">${data.outcome.steps}</span></div>`;
      detailHTML += `<div class="result-row"><span class="rk">Queries → User</span><span class="rv">${data.action_counts.QueryUser}</span></div>`;
      detailHTML += `<div class="result-row"><span class="rk">Wait Actions</span><span class="rv">${data.action_counts.Wait}</span></div>`;
      detailHTML += `<div class="result-row"><span class="rk">Searches</span><span class="rv">${data.action_counts.Search}</span></div>`;
      detailHTML += `<div class="result-row"><span class="rk">Tip</span><span class="rv bad">Try a higher ε or different seed</span></div>`;
      resultDetails.innerHTML = detailHTML;
      resultOverlay.style.display = 'flex';
    }

  } catch (err) {
    alert('Error running episode: ' + err.message);
  }

  isRunning = false;
  runBtn.disabled = false;
  runBtn.textContent = '▶  Start Shopping';
}

// ═══════════════════════════════════════════════════════════════
// ANALYSIS SECTION
// ═══════════════════════════════════════════════════════════════

function showAnalysis() {
  if (!lastEpisodeData) return;
  const data = lastEpisodeData;
  const o = data.outcome;
  const ac = data.action_counts;
  const params = data._params;

  analysisSection.style.display = 'block';

  // ── Metrics ──────────────────────────────────────────────
  $('#m-steps').textContent = o.steps;
  const regEl = $('#m-regret');
  if (o.realized_regret !== undefined) {
    regEl.textContent = o.realized_regret.toFixed(4);
    regEl.className = 'metric-val ' + (o.realized_regret < 0.1 ? 'good' : 'bad');
  } else {
    regEl.textContent = 'N/A';
    regEl.className = 'metric-val bad';
  }
  $('#m-queries').textContent = ac.QueryUser;
  $('#m-waits').textContent = ac.Wait;
  $('#m-searches').textContent = ac.Search;

  // ── Narrative ────────────────────────────────────────────
  $('#narrative-text').innerHTML = generateNarrative(data, params);

  // ── Uncertainty chart ────────────────────────────────────
  drawUncertaintyChart(data.epi_unc_history, data.steps);

  // ── Action donut chart ───────────────────────────────────
  drawActionChart(ac);

  // ── Timeline ─────────────────────────────────────────────
  buildTimeline(data.steps);

  // Scroll into view
  setTimeout(() => {
    analysisSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 200);
}

// ── Narrative generator ────────────────────────────────────
function generateNarrative(data, params) {
  const o = data.outcome;
  const ac = data.action_counts;
  const unc = data.epi_unc_history;
  const persona = params.persona.replace('_', ' ');

  let text = '';

  // Opening
  text += `The <strong>${persona}</strong> agent was deployed with a regret threshold of <strong>ε=${params.eps_reg}</strong>. `;

  // Uncertainty journey
  if (unc.length >= 2) {
    const startU = unc[0].toFixed(3);
    const endU = unc[unc.length - 1].toFixed(4);
    const reduction = ((1 - unc[unc.length - 1] / unc[0]) * 100).toFixed(0);
    text += `Initial epistemic uncertainty was <strong>${startU}</strong>, which dropped to <strong>${endU}</strong> (a <strong>${reduction}% reduction</strong> in model uncertainty). `;
  }

  // Action breakdown
  if (ac.QueryUser > 0) {
    text += `The agent queried the user <strong>${ac.QueryUser} time${ac.QueryUser > 1 ? 's' : ''}</strong> to refine its belief about preferences. `;
  }
  if (ac.Wait > 0) {
    text += `It chose to wait <strong>${ac.Wait} time${ac.Wait > 1 ? 's' : ''}</strong>, using Monte Carlo rollouts to estimate that future market states could yield better options. `;
  }
  if (ac.Search > 0) {
    text += `It performed <strong>${ac.Search} search${ac.Search > 1 ? 'es' : ''}</strong> to explore the product catalog. `;
  }

  // Outcome
  if (o.purchased) {
    text += `<br><br>After <strong>${o.steps} steps</strong>, the agent purchased <strong>${o.item.name}</strong> `;
    text += `at <strong>$${o.item.price.toFixed(2)}</strong> (${o.item.rating}★, ${o.item.quality}% quality). `;
    if (o.realized_regret < 0.01) {
      text += `The realized regret was <strong style="color:#067d62">${o.realized_regret.toFixed(4)}</strong>, essentially the best possible choice given the available options. This demonstrates the agent's ability to learn user preferences efficiently.`;
    } else if (o.realized_regret < 0.1) {
      text += `The realized regret was <strong style="color:#067d62">${o.realized_regret.toFixed(4)}</strong>, well within the safety threshold, indicating a near-optimal purchase.`;
    } else {
      text += `The realized regret was <strong style="color:#b12704">${o.realized_regret.toFixed(4)}</strong>, which is above ideal. The agent may have been too aggressive with the given uncertainty level.`;
    }
  } else {
    text += `<br><br>The agent <strong>did not purchase</strong> within the ${o.steps}-step budget. `;
    text += `This typically happens when the regret threshold (ε=${params.eps_reg}) is too strict for the agent's remaining uncertainty. `;
    text += `The Bayesian safety gate prevents the agent from committing to a purchase it is not sufficiently confident about, a core feature of the delegated procurement framework.`;
  }

  // Regret threshold assessment
  text += `<br><br><strong>Threshold assessment:</strong> `;
  const eps = params.eps_reg;
  if (o.purchased && o.realized_regret < 0.05 && o.steps <= 15) {
    text += `The chosen ε=${eps} was well-calibrated for this scenario. The agent purchased confidently within a reasonable number of steps and achieved very low regret.`;
  } else if (o.purchased && o.realized_regret < 0.1) {
    if (o.steps > 20) {
      text += `ε=${eps} was slightly conservative. While the agent found a good purchase, it took ${o.steps} steps to commit. A slightly higher threshold (e.g., ε=${(eps + 0.2).toFixed(1)}) could speed up decision-making with minimal regret trade-off.`;
    } else {
      text += `ε=${eps} was a reasonable choice. The agent balanced exploration and exploitation effectively, achieving low regret in ${o.steps} steps.`;
    }
  } else if (o.purchased && o.realized_regret >= 0.1) {
    text += `ε=${eps} may have been too permissive. The agent purchased quickly but with higher-than-ideal regret (${o.realized_regret.toFixed(4)}). A lower threshold (e.g., ε=${Math.max(0.1, eps - 0.3).toFixed(1)}) would enforce stricter safety checks before committing.`;
  } else {
    text += `ε=${eps} was too conservative for this market configuration. The agent's safety gate never cleared because uncertainty remained above the threshold. Consider increasing to ε=${(eps + 0.3).toFixed(1)} or higher, or allowing more steps for the agent to gather sufficient feedback.`;
  }

  return text;
}

// ── Uncertainty line chart (Canvas API) ────────────────────
function drawUncertaintyChart(uncHistory, steps) {
  const canvas = $('#unc-chart');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;

  // Hi-DPI scaling
  const rect = canvas.parentElement.getBoundingClientRect();
  const W = Math.min(rect.width - 48, 560);
  const H = 240;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx.scale(dpr, dpr);

  if (uncHistory.length === 0) {
    ctx.fillStyle = '#aaa';
    ctx.font = '14px Inter, sans-serif';
    ctx.fillText('No uncertainty data available', W / 2 - 80, H / 2);
    return;
  }

  const pad = { top: 20, right: 20, bottom: 40, left: 55 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const maxU = Math.max(...uncHistory) * 1.15;
  const minU = 0;

  // Grid
  ctx.strokeStyle = '#e8e8e8';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();
  }

  // Y-axis labels
  ctx.fillStyle = '#888';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = maxU - (maxU - minU) * (i / 4);
    const y = pad.top + (plotH / 4) * i;
    ctx.fillText(val.toFixed(3), pad.left - 8, y + 4);
  }

  // X-axis labels
  ctx.textAlign = 'center';
  const nPoints = uncHistory.length;
  const labelInterval = Math.max(1, Math.floor(nPoints / 8));
  for (let i = 0; i < nPoints; i += labelInterval) {
    const x = pad.left + (i / (nPoints - 1 || 1)) * plotW;
    ctx.fillText(i + 1, x, H - 8);
  }

  // Axis titles
  ctx.fillStyle = '#666';
  ctx.font = '12px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Step', pad.left + plotW / 2, H - 0);

  ctx.save();
  ctx.translate(12, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Epistemic Uncertainty', 0, 0);
  ctx.restore();

  // Area fill
  ctx.beginPath();
  uncHistory.forEach((u, i) => {
    const x = pad.left + (i / (nPoints - 1 || 1)) * plotW;
    const y = pad.top + plotH - ((u - minU) / (maxU - minU)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
  grad.addColorStop(0, 'rgba(0,102,192,0.2)');
  grad.addColorStop(1, 'rgba(0,102,192,0.02)');
  ctx.fillStyle = grad;
  ctx.fill();

  // Line
  ctx.beginPath();
  uncHistory.forEach((u, i) => {
    const x = pad.left + (i / (nPoints - 1 || 1)) * plotW;
    const y = pad.top + plotH - ((u - minU) / (maxU - minU)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#0066c0';
  ctx.lineWidth = 2.5;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // Dots at key points (first, last, min)
  const minIdx = uncHistory.indexOf(Math.min(...uncHistory));
  [0, minIdx, nPoints - 1].forEach(i => {
    if (i < 0 || i >= nPoints) return;
    const x = pad.left + (i / (nPoints - 1 || 1)) * plotW;
    const y = pad.top + plotH - ((uncHistory[i] - minU) / (maxU - minU)) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#0066c0';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  // Mark action types along the bottom with colored ticks
  if (steps && steps.length > 0) {
    const actionColors = { Search: '#8779a9', QueryUser: '#0066c0', Wait: '#c45500', Purchase: '#067d62' };
    steps.forEach((s, i) => {
      if (i >= nPoints) return;
      const x = pad.left + (i / (nPoints - 1 || 1)) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, pad.top + plotH);
      ctx.lineTo(x, pad.top + plotH + 6);
      ctx.strokeStyle = actionColors[s.action] || '#ccc';
      ctx.lineWidth = 2;
      ctx.stroke();
    });
  }
}

// ── Action donut chart (Canvas API) ────────────────────────
function drawActionChart(ac) {
  const canvas = $('#action-chart');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;

  const rect = canvas.parentElement.getBoundingClientRect();
  const W = Math.floor(rect.width - 48);
  const H = Math.floor(W * 0.55);
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx.scale(dpr, dpr);

  const cx = W / 2, cy = H / 2 - 5;
  const R = Math.min(W, H) * 0.35, r = R * 0.55;

  const slices = [
    { label: 'Search', count: ac.Search, color: '#8779a9' },
    { label: 'QueryUser', count: ac.QueryUser, color: '#0066c0' },
    { label: 'Wait', count: ac.Wait, color: '#c45500' },
    { label: 'Purchase', count: ac.Purchase, color: '#067d62' },
  ].filter(s => s.count > 0);

  const total = slices.reduce((s, x) => s + x.count, 0);
  if (total === 0) return;

  let startAngle = -Math.PI / 2;
  slices.forEach(slice => {
    const sweep = (slice.count / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx + r * Math.cos(startAngle), cy + r * Math.sin(startAngle));
    ctx.arc(cx, cy, R, startAngle, startAngle + sweep);
    ctx.arc(cx, cy, r, startAngle + sweep, startAngle, true);
    ctx.closePath();
    ctx.fillStyle = slice.color;
    ctx.fill();

    // Label
    const midAngle = startAngle + sweep / 2;
    const lx = cx + (R + 16) * Math.cos(midAngle);
    const ly = cy + (R + 16) * Math.sin(midAngle);
    ctx.fillStyle = '#444';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = midAngle > Math.PI / 2 && midAngle < Math.PI * 1.5 ? 'right' : 'left';
    ctx.fillText(`${slice.label} (${slice.count})`, lx, ly + 4);

    startAngle += sweep;
  });

  // Center text
  ctx.fillStyle = '#232f3e';
  ctx.font = '700 20px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(total, cx, cy + 4);
  ctx.fillStyle = '#888';
  ctx.font = '11px Inter, sans-serif';
  ctx.fillText('total steps', cx, cy + 18);
}

// ── Timeline builder ───────────────────────────────────────
function buildTimeline(steps) {
  const tl = $('#timeline');
  tl.innerHTML = '';

  steps.forEach(step => {
    const row = document.createElement('div');
    row.className = 'tl-step';

    let productsHTML = '';
    if (step.products && step.products.length > 0) {
      productsHTML = step.products.map(p => {
        let chipClass = 'tl-product-chip';
        if (step.target_item && p.id === step.target_item.id) {
          chipClass += step.action === 'Purchase' ? ' purchased' : ' target';
        }
        return `<span class="${chipClass}">${p.name} — $${p.price.toFixed(2)}</span>`;
      }).join('');
    }

    let metaHTML = '';
    if (step.epi_unc !== null) {
      metaHTML = `Uncertainty: ${step.epi_unc}`;
      if (step.exp_util !== null) metaHTML += ` · Exp. Utility: ${step.exp_util}`;
    }

    row.innerHTML = `
      <div class="tl-num ${step.action}">${step.step}</div>
      <div class="tl-body">
        <div class="tl-action ${step.action}">${step.action}</div>
        <div class="tl-reason">${step.reason}</div>
        ${productsHTML ? `<div class="tl-products">${productsHTML}</div>` : ''}
        ${metaHTML ? `<div class="tl-meta">${metaHTML}</div>` : ''}
      </div>
    `;
    tl.appendChild(row);
  });
}

// ─── Event Listeners ───────────────────────────────────────
runBtn.addEventListener('click', runEpisode);

closeOverlay.addEventListener('click', () => {
  resultOverlay.style.display = 'none';
  showAnalysis();
});

runAgainBtn.addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
  setTimeout(() => runEpisode(), 400);
});

// ─── Product Search ────────────────────────────────────────
async function performSearch(query) {
  if (!query || query.length < 1) {
    searchDropdown.classList.remove('open');
    return;
  }
  try {
    const resp = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const results = await resp.json();
    if (results.length === 0) {
      searchDropdown.innerHTML = '<div class="search-empty">No products found</div>';
    } else {
      searchDropdown.innerHTML = results.map(p => {
        const catClass = CATEGORY_CSS_CLASS[p.category] || '';
        const icon = CATEGORY_ICONS[p.category] || '';
        return `<div class="search-result" data-id="${p.id}">
          <div class="search-result-icon ${catClass}">${icon}</div>
          <div class="search-result-info">
            <div class="search-result-name">${p.name}</div>
            <div class="search-result-meta">${p.category} · ${renderStars(p.rating)} ${p.rating} · Quality: ${p.quality}%</div>
          </div>
          <div class="search-result-price">$${p.price.toFixed(2)}</div>
        </div>`;
      }).join('');
    }
    searchDropdown.classList.add('open');
  } catch (err) {
    searchDropdown.classList.remove('open');
  }
}

searchInput.addEventListener('input', () => {
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(() => {
    performSearch(searchInput.value.trim());
  }, 250);
});

searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    searchDropdown.classList.remove('open');
    searchInput.blur();
  }
});

document.addEventListener('click', (e) => {
  if (!e.target.closest('.nav-center')) {
    searchDropdown.classList.remove('open');
  }
});

searchInput.addEventListener('focus', () => {
  if (searchInput.value.trim().length >= 1) {
    performSearch(searchInput.value.trim());
  }
});
