// ── LIVE COMMAND CENTER LOGIC ────────────────────────────────────────────────

// Handle button states based on global status
document.addEventListener('statusUpdated', (e) => {
    const status = e.detail;
    
    const setBtn = (id, isRunning, onText, offText, onClass = 'btn-danger', offClass = 'btn-ghost') => {
        const btn = document.getElementById(id);
        if(!btn) return;
        if(isRunning) {
            btn.textContent = onText;
            btn.className = `btn ${onClass}`;
        } else {
            btn.textContent = offText;
            btn.className = `btn ${offClass}`;
        }
    };

    setBtn('btn-prom', status.port_forward_prom, 'Stop Prometheus Bridge', 'Start Prometheus Bridge');
    setBtn('btn-podinfo', status.port_forward_podinfo, 'Stop Cluster Bridge', 'Start Cluster Bridge');
    setBtn('btn-locust', status.locust, 'Stop Chaos Injection', 'Inject Chaos');
    setBtn('btn-agent', status.live_agent, 'Stop RL Agent', 'Deploy RL Agent', 'btn-danger', 'btn-amber');
});

// Live metrics polling
let lastStep = -1;

async function fetchLiveMetrics() {
    try {
        const response = await fetch('/api/live_metrics');
        const data = await response.json();
        
        if (data.error) return; // Silent fail if no live run

        const step = parseInt(data.step);
        if (step === lastStep) return; // No new data
        lastStep = step;

        document.getElementById('step-badge').textContent = `Step ${step}`;
        
        // Traffic
        document.getElementById('val-traffic').textContent = formatNumber(data.request_rate);
        
        // Latency & SLA
        const lat = parseFloat(data.p99_latency);
        document.getElementById('val-latency').textContent = formatNumber(lat, 0);
        
        const breachEl = document.getElementById('val-breach');
        const latCard = document.getElementById('card-latency');
        
        if (data.sla_breach === "True" || lat > 200) {
            breachEl.textContent = "SLA BREACHED";
            breachEl.className = "metric-sub bad";
            latCard.classList.add('breach');
        } else {
            breachEl.textContent = "Within SLA Limits";
            breachEl.className = "metric-sub ok";
            latCard.classList.remove('breach');
        }

        // Replicas & Cluster
        const reps = parseInt(data.replicas) || 0;
        const pending = parseInt(data.pending_pods) || 0;
        document.getElementById('val-replicas').textContent = reps;
        document.getElementById('val-pending').textContent = `${pending} booting`;
        renderCluster('cluster-grid', reps, pending);

        // Action
        const delta = parseInt(data.safe_delta);
        const actionEl = document.getElementById('val-action');
        const actSub = document.getElementById('val-action-sub');
        
        if (delta > 0) {
            actionEl.textContent = `SCALE UP (+${delta})`;
            actSub.textContent = `Proposed: ${data.proposed_delta}`;
            actSub.className = "metric-sub ok";
            appendLog(step, `Scaled up by ${delta} pods.`, 'up');
        } else if (delta < 0) {
            actionEl.textContent = `SCALE DOWN (${delta})`;
            actSub.textContent = `Proposed: ${data.proposed_delta}`;
            actSub.className = "metric-sub warn";
            appendLog(step, `Scaled down by ${Math.abs(delta)} pods.`, 'down');
        } else {
            actionEl.textContent = `HOLD STEADY`;
            actSub.textContent = `System stable`;
            actSub.className = "metric-sub";
            appendLog(step, `No scaling action required.`, 'hold');
        }

    } catch (error) {
        console.error("Metrics poll failed", error);
    }
}

function appendLog(step, msg, type) {
    const log = document.getElementById('action-log');
    if(!log) return;
    
    // Remove empty state
    if(log.querySelector('.empty-state')) log.innerHTML = '';
    
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `
        <div class="log-step">[${step}]</div>
        <div class="log-action ${type}">${msg}</div>
    `;
    
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

function clearLog() {
    const log = document.getElementById('action-log');
    if(log) log.innerHTML = '<div class="empty-state">Log cleared.</div>';
}

setInterval(fetchLiveMetrics, 2000);
