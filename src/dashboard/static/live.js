// ── LIVE COMMAND CENTER LOGIC ────────────────────────────────────────────────

// ── Button state management ───────────────────────────────────────────────────

document.addEventListener('statusUpdated', (e) => {
    const status = e.detail;

    const setBtn = (id, isRunning, onText, offText, onClass = 'btn-danger', offClass = 'btn-ghost') => {
        const btn = document.getElementById(id);
        if (!btn) return;
        btn.textContent = isRunning ? onText : offText;
        btn.className   = `btn ${isRunning ? onClass : offClass}`;
    };

    setBtn('btn-locust',  status.locust,      'Stop Traffic Injection',  'Inject Traffic');
    setBtn('btn-agent',   status.live_agent,   'Stop RL Agent',           'Deploy RL Agent', 'btn-danger', 'btn-amber');

    // ── Boutique & Prometheus connectivity (NodePort, no process) ─────────
    const boutiqueLive = !!status.boutique_live;
    const promLive     = !!status.prometheus_live;

    // Prometheus indicator
    const promBtn = document.getElementById('btn-prom');
    if (promBtn) {
        promBtn.textContent = promLive ? '● Prometheus Connected' : '○ Prometheus Offline';
        promBtn.className   = `btn ${promLive ? 'btn-success' : 'btn-ghost'}`;
        promBtn.disabled    = true;  // Informational only — no process to toggle
    }

    // Boutique bridge — auto-show iframe when NodePort is live
    const boutiqueBtn = document.getElementById('btn-boutique');
    if (boutiqueBtn) {
        if (boutiqueLive) {
            // Use manual toggle state (user might have hidden it)
            if (_boutiqueWantVisible) {
                boutiqueBtn.textContent = 'Hide Boutique';
                boutiqueBtn.className   = 'btn btn-danger';
            } else {
                boutiqueBtn.textContent = 'Show Boutique';
                boutiqueBtn.className   = 'btn btn-ghost';
            }
        } else {
            boutiqueBtn.textContent = 'Boutique Offline';
            boutiqueBtn.className   = 'btn btn-ghost';
            boutiqueBtn.disabled    = true;
        }
        if (boutiqueLive) boutiqueBtn.disabled = false;
    }

    // Auto-show boutique iframe on first detection of live NodePort
    if (boutiqueLive && !_boutiqueEverLive) {
        _boutiqueEverLive = true;
        _boutiqueWantVisible = true;
        _loadBoutiqueIframe();
    }

    // If boutique just went offline, hide the iframe
    if (!boutiqueLive && _prevBoutiqueLive) {
        _hideBoutiqueIframe();
    }
    _prevBoutiqueLive = boutiqueLive;
});

// ── Boutique iframe control ───────────────────────────────────────────────────
// Use the Flask reverse-proxy route (/boutique/) so the iframe loads from
// the same origin as the dashboard — bypasses all browser cross-origin blocks.
const BOUTIQUE_PROXY = '/boutique/';

let _prevBoutiqueLive = false;
let _boutiqueEverLive = false;
let _boutiqueWantVisible = true;  // User toggle preference

function toggleBoutique() {
    // Simple visibility toggle — no process to start/stop
    _boutiqueWantVisible = !_boutiqueWantVisible;
    if (_boutiqueWantVisible) {
        _loadBoutiqueIframe();
    } else {
        _hideBoutiqueIframe();
    }
}

function _loadBoutiqueIframe() {
    const iframe  = document.getElementById('boutique-iframe');
    const offline = document.getElementById('boutique-offline');
    const pulse   = document.getElementById('boutique-pulse');
    const label   = document.getElementById('boutique-status-text');

    if (!iframe) return;

    // Only set src if it's not already pointing to our proxy (avoid reload flicker)
    if (!iframe.src.includes('/boutique/')) {
        iframe.src = BOUTIQUE_PROXY;
    }

    iframe.style.display  = 'block';
    if (offline) offline.style.display = 'none';
    if (pulse)   pulse.classList.add('active');
    if (label)   label.textContent = 'LIVE';
}

function _hideBoutiqueIframe() {
    const iframe  = document.getElementById('boutique-iframe');
    const offline = document.getElementById('boutique-offline');
    const pulse   = document.getElementById('boutique-pulse');
    const label   = document.getElementById('boutique-status-text');

    if (iframe) {
        iframe.src          = 'about:blank';
        iframe.style.display = 'none';
    }
    if (offline) offline.style.display = 'flex';
    if (pulse)   pulse.classList.remove('active');
    if (label)   label.textContent = 'OFFLINE';
}

// ── Live metrics polling (reads from live agent's CSV log) ────────────────────

let lastStep = -1;

async function fetchLiveMetrics() {
    try {
        const response = await fetch('/api/live_metrics');
        const data     = await response.json();

        if (data.error) return; // No active run — silent

        const step = parseInt(data.step);
        if (step === lastStep) return;
        lastStep = step;

        document.getElementById('step-badge').textContent = `Step ${step}`;

        // Traffic
        const rps = parseFloat(data.request_rate) || 0;
        document.getElementById('val-traffic').textContent    = rps.toFixed(1);
        document.getElementById('val-traffic-sub').textContent = `${rps.toFixed(0)} req/s from Locust`;

        // Latency & SLA
        const lat      = parseFloat(data.p99_latency) || 0;
        const breachEl = document.getElementById('val-breach');
        const latCard  = document.getElementById('card-latency');

        document.getElementById('val-latency').textContent = lat.toFixed(0);

        const breached = data.sla_breach === 'True' || lat > 200;
        if (breached) {
            breachEl.textContent = '⚠ SLA BREACHED';
            breachEl.className   = 'metric-sub bad';
            latCard.classList.add('breach');
            document.body.classList.add('sla-breach');
        } else {
            breachEl.textContent = 'Within SLA Target';
            breachEl.className   = 'metric-sub ok';
            latCard.classList.remove('breach');
            document.body.classList.remove('sla-breach');
        }

        // Replicas & Pod Grid
        const reps    = parseInt(data.replicas)      || 0;
        const pending = parseInt(data.pending_pods)  || 0;
        document.getElementById('val-replicas').textContent = reps;
        document.getElementById('val-pending').textContent  = `${pending} booting`;
        renderCluster('cluster-grid', reps, pending);

        // Agent action
        const delta    = parseInt(data.safe_delta) || 0;
        const actionEl = document.getElementById('val-action');
        const actSub   = document.getElementById('val-action-sub');

        if (delta > 0) {
            actionEl.textContent = `SCALE UP  +${delta}`;
            actSub.textContent   = `Proposed Δ: ${data.proposed_delta}`;
            actSub.className     = 'metric-sub ok';
            appendLog(step, `⬆ Scaled up by ${delta} pod${delta > 1 ? 's' : ''} → ${reps} total`, 'up');
        } else if (delta < 0) {
            actionEl.textContent = `SCALE DOWN ${delta}`;
            actSub.textContent   = `Proposed Δ: ${data.proposed_delta}`;
            actSub.className     = 'metric-sub warn';
            appendLog(step, `⬇ Scaled down by ${Math.abs(delta)} pod${Math.abs(delta) > 1 ? 's' : ''} → ${reps} total`, 'down');
        } else {
            actionEl.textContent = 'HOLD STEADY';
            actSub.textContent   = 'System stable';
            actSub.className     = 'metric-sub';
            appendLog(step, `◼ Hold — ${reps} pods, P99 ${lat.toFixed(0)} ms`, 'hold');
        }

    } catch (err) {
        console.error('Metrics poll failed:', err);
    }
}

function appendLog(step, msg, type) {
    const log = document.getElementById('action-log');
    if (!log) return;

    if (log.querySelector('.empty-state')) log.innerHTML = '';

    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <div class="log-step">[${step}]</div>
        <div class="log-action ${type}">${msg}</div>
    `;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;

    // Keep log from growing unbounded
    while (log.children.length > 200) log.removeChild(log.firstChild);
}

function clearLog() {
    const log = document.getElementById('action-log');
    if (log) log.innerHTML = '<div class="empty-state">Log cleared.</div>';
}

setInterval(fetchLiveMetrics, 2000);
