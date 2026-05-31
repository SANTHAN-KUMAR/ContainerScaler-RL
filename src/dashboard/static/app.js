let currentStatus = {};
let currentReplicas = 0;

// Format numbers
function formatNumber(num, dec = 1) {
    const val = parseFloat(num);
    return isNaN(val) ? "0" : val.toFixed(dec);
}

// Global Status Fetch
async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        currentStatus = await response.json();

        // Autoscaler Toggle Sync
        const isAgentRunning = currentStatus['live_agent'] === true;
        const toggle = document.getElementById('toggle-autoscaler');
        if (toggle && toggle.checked !== isAgentRunning && !toggle.disabled) {
            toggle.checked = isAgentRunning;
        }

        // Global Status Indicator
        const anyRunning = Object.values(currentStatus).some(v => v);
        const dot = document.getElementById('global-status-dot');
        const text = document.getElementById('global-status-text');
        
        if (anyRunning) {
            dot.classList.add('live');
            text.textContent = 'System Active';
        } else {
            dot.classList.remove('live');
            text.textContent = 'Standby';
        }

        // Traffic Button
        const btnTraffic = document.getElementById('btn-traffic');
        if (btnTraffic) {
            if (currentStatus['locust']) {
                btnTraffic.textContent = 'Stop Traffic';
                btnTraffic.classList.remove('btn-primary');
                btnTraffic.classList.add('btn-danger');
            } else {
                btnTraffic.textContent = 'Inject Load';
                btnTraffic.classList.add('btn-primary');
                btnTraffic.classList.remove('btn-danger');
            }
        }

        // Eval Button
        const btnEval = document.getElementById('btn-eval');
        if (btnEval) {
            if (currentStatus['evaluation']) {
                btnEval.textContent = 'Running Benchmark...';
                btnEval.disabled = true;
            } else {
                btnEval.textContent = 'Run Benchmark';
                btnEval.disabled = false;
            }
        }

    } catch (error) {
        console.error("Error fetching status:", error);
    }
}

// Unified toggle for the RL Autoscaler infrastructure
async function handleAutoscalerToggle(checkbox) {
    const shouldStart = checkbox.checked;
    checkbox.disabled = true; // prevent spamming

    try {
        if (shouldStart) {
            // Sequence: Prom -> Podinfo -> Agent
            let res = await fetch('/api/start/port_forward_prom', { method: 'POST' });
            if (!res.ok) { let data = await res.json(); throw new Error(`Prometheus: ${data.error}`); }
            
            res = await fetch('/api/start/port_forward_podinfo', { method: 'POST' });
            if (!res.ok) { let data = await res.json(); throw new Error(`Podinfo: ${data.error}`); }
            
            // Small delay to let port forwards settle
            await new Promise(r => setTimeout(r, 1000));
            
            res = await fetch('/api/start/live_agent', { method: 'POST' });
            if (!res.ok) { let data = await res.json(); throw new Error(`Agent: ${data.error}`); }
        } else {
            await fetch('/api/stop/live_agent', { method: 'POST' });
            await fetch('/api/stop/port_forward_podinfo', { method: 'POST' });
            await fetch('/api/stop/port_forward_prom', { method: 'POST' });
        }
    } catch (error) {
        console.error("Error toggling autoscaler:", error);
        alert(`Failed to start autoscaler:\n${error.message}`);
        checkbox.checked = !shouldStart; // revert on fail
    } finally {
        checkbox.disabled = false;
        await updateStatus();
    }
}

// Generic toggle for other services
async function toggleService(service) {
    const isRunning = currentStatus[service];
    const endpoint = isRunning ? `/api/stop/${service}` : `/api/start/${service}`;
    
    try {
        let res = await fetch(endpoint, { method: 'POST' });
        if (!res.ok) {
            let data = await res.json();
            throw new Error(data.error);
        }
        await updateStatus();
        
        if (!isRunning && service === 'evaluation') {
            // Fetch plots frequently while evaluating
            setTimeout(fetchPlots, 5000);
            setTimeout(fetchPlots, 15000);
        }
    } catch (error) {
        console.error(`Error toggling ${service}:`, error);
        alert(`Failed to toggle ${service}:\n${error.message}`);
        await updateStatus();
    }
}

async function emergencyStop() {
    try {
        const toggle = document.getElementById('toggle-autoscaler');
        if(toggle) { toggle.checked = false; toggle.disabled = true; }
        
        await fetch('/api/stop_all', { method: 'POST' });
        setTimeout(async () => {
            await updateStatus();
            if(toggle) toggle.disabled = false;
        }, 1000);
    } catch (error) {
        console.error("Error stopping all:", error);
    }
}

// Visuals
function renderCluster(replicas) {
    const grid = document.getElementById('cluster-grid');
    if (replicas === currentReplicas) return;
    
    grid.innerHTML = '';
    for(let i=0; i<replicas; i++) {
        const node = document.createElement('div');
        node.className = 'pod-icon active';
        node.innerHTML = `<svg viewBox="0 0 24 24"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16zM12 22.27 4.5 18V9.73L12 14l7.5-4.27V18zM12 11.73 4.5 7.5 12 3.27l7.5 4.23z"/></svg>`;
        grid.appendChild(node);
    }
    
    currentReplicas = replicas;
}

// Live Metrics
async function fetchLiveMetrics() {
    try {
        const response = await fetch('/api/live_metrics');
        const data = await response.json();
        
        if (data.error) {
            return;
        }

        const stepBadge = document.getElementById('metrics-step');
        if(stepBadge) stepBadge.textContent = `Step ${data.step}`;

        document.getElementById('val-traffic').textContent = formatNumber(data.request_rate);
        
        const latency = parseFloat(data.p99_latency);
        document.getElementById('val-latency').textContent = formatNumber(latency, 0);
        
        const breachEl = document.getElementById('val-breach');
        if (data.sla_breach === "True" || latency > 200) {
            breachEl.textContent = "SLA Breached";
            breachEl.className = "metric-trend trend-bad";
        } else {
            breachEl.textContent = "Within SLA";
            breachEl.className = "metric-trend trend-good";
        }

        const replicas = parseInt(data.replicas) || 0;
        document.getElementById('val-replicas').textContent = replicas;
        document.getElementById('val-pending').textContent = `${data.pending_pods || 0} Pending`;
        renderCluster(replicas);

        const delta = parseInt(data.safe_delta);
        const actionEl = document.getElementById('val-action');
        if (delta > 0) {
            actionEl.textContent = `Scale Up (+${delta})`;
        } else if (delta < 0) {
            actionEl.textContent = `Scale Down (${delta})`;
        } else {
            actionEl.textContent = `Maintain`;
        }

        document.getElementById('val-proposed').textContent = `Proposed: ${data.proposed_delta}`;

    } catch (error) {
        console.error("Error fetching live metrics:", error);
    }
}

// Logs
async function fetchLogs() {
    try {
        const response = await fetch('/api/logs/live_agent');
        const data = await response.json();
        
        const terminalWindow = document.getElementById('terminal-output');
        if(!terminalWindow) return;

        const isScrolledToBottom = terminalWindow.scrollHeight - terminalWindow.clientHeight <= terminalWindow.scrollTop + 50;

        if (data.logs) {
            const lines = data.logs.split('\n').filter(l => l.trim().length > 0);
            let formattedHtml = '';
            for(let line of lines) {
                formattedHtml += `<div class="log-line">${line}</div>`;
            }
            terminalWindow.innerHTML = formattedHtml || '<div class="log-line">Waiting for system logs...</div>';
        }

        if (isScrolledToBottom) {
            terminalWindow.scrollTop = terminalWindow.scrollHeight;
        }
        
    } catch (error) {
        // fail silently
    }
}

// Plots
async function fetchPlots() {
    try {
        const response = await fetch('/api/plots');
        const plots = await response.json();
        
        const gallery = document.getElementById('plot-gallery');
        if(!gallery) return;
        
        if (plots.length === 0) {
            gallery.innerHTML = `
                <div style="color: var(--text-secondary); font-size: 14px; text-align: center; padding: 2rem;">
                    No recent benchmarks available.
                </div>
            `;
            return;
        }

        let html = '';
        for (const plot of plots) {
            html += `<div class="plot-wrapper"><img src="/plots/${plot}?t=${new Date().getTime()}" class="plot-image" alt="Benchmark Plot"></div>`;
        }
        gallery.innerHTML = html;
        
    } catch (error) {
        console.error("Error fetching plots:", error);
    }
}

// Init & Loops
updateStatus();
fetchPlots();
fetchLogs();
fetchLiveMetrics();

setInterval(updateStatus, 2000);
setInterval(fetchLogs, 2000);
setInterval(fetchLiveMetrics, 2000);
setInterval(fetchPlots, 30000);
