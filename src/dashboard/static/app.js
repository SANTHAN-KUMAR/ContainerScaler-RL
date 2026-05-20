const services = {
    'port_forward_prom': { id: 'btn-prom', defaultText: 'Start' },
    'port_forward_podinfo': { id: 'btn-podinfo', defaultText: 'Start' },
    'live_agent': { id: 'btn-agent', defaultText: 'Start Agent' },
    'locust': { id: 'btn-locust', defaultText: 'Generate Traffic' },
    'evaluation': { id: 'btn-evaluation', defaultText: 'Run Evaluation' }
};

// Global state
let currentStatus = {};

// Fetch and update status
async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        currentStatus = await response.json();

        for (const [service, isRunning] of Object.entries(currentStatus)) {
            const btn = document.getElementById(services[service].id);
            if (!btn) continue;

            if (isRunning) {
                btn.classList.add('running');
                btn.innerHTML = `<span class="material-icons" style="font-size:18px">stop</span> Stop`;
            } else {
                btn.classList.remove('running');
                btn.innerHTML = services[service].defaultText;
            }
        }
    } catch (error) {
        console.error("Error fetching status:", error);
    }
}

// Toggle a service
async function toggleService(service) {
    const isRunning = currentStatus[service];
    const endpoint = isRunning ? `/api/stop/${service}` : `/api/start/${service}`;
    
    // Optimistic UI update
    const btn = document.getElementById(services[service].id);
    btn.innerHTML = `<span class="material-icons" style="font-size:18px; animation: spin 1s linear infinite;">autorenew</span>`;

    try {
        await fetch(endpoint, { method: 'POST' });
        await updateStatus();
        
        // If we just started the agent, refresh plots shortly after
        if (!isRunning && service === 'live_agent') {
            setTimeout(fetchPlots, 2000);
        }
    } catch (error) {
        console.error(`Error toggling ${service}:`, error);
        await updateStatus();
    }
}

// Stop all
document.getElementById('btn-stop-all').addEventListener('click', async () => {
    try {
        await fetch('/api/stop_all', { method: 'POST' });
        await updateStatus();
    } catch (error) {
        console.error("Error stopping all:", error);
    }
});

// Fetch and render plots
async function fetchPlots() {
    try {
        const response = await fetch('/api/plots');
        const plots = await response.json();
        
        const gallery = document.getElementById('plot-gallery');
        
        if (plots.length === 0) {
            gallery.innerHTML = `
                <div class="empty-state">
                    <span class="material-icons empty-icon">insert_chart_outlined</span>
                    <p>No plots generated yet. Start the agent and click Generate Plot Now.</p>
                </div>
            `;
            return;
        }

        // Generate HTML for plots
        let html = '';
        for (const plot of plots) {
            // Append timestamp to prevent caching
            html += `<img src="/plots/${plot}?t=${new Date().getTime()}" class="plot-image" alt="Trajectory Plot">`;
        }
        gallery.innerHTML = html;
        
    } catch (error) {
        console.error("Error fetching plots:", error);
    }
}

// Fetch live agent logs
async function fetchLogs() {
    try {
        const response = await fetch('/api/logs/live_agent');
        const data = await response.json();
        
        const terminal = document.querySelector('.terminal-text');
        const terminalWindow = document.getElementById('terminal-output');
        
        // Auto-scroll if already at the bottom
        const isScrolledToBottom = terminalWindow.scrollHeight - terminalWindow.clientHeight <= terminalWindow.scrollTop + 50;

        if (data.logs) {
            terminal.textContent = data.logs;
        } else if (data.error) {
            terminal.textContent = "Error loading logs: " + data.error;
        }

        if (isScrolledToBottom) {
            terminalWindow.scrollTop = terminalWindow.scrollHeight;
        }
        
    } catch (error) {
        console.error("Error fetching logs:", error);
    }
}

// Generate plot immediately
async function generatePlotNow() {
    const btn = document.getElementById('btn-plot-now');
    const originalHtml = btn.innerHTML;
    btn.innerHTML = `<span class="material-icons" style="font-size:18px; animation: spin 1s linear infinite;">autorenew</span> Generating...`;
    btn.disabled = true;

    try {
        const response = await fetch('/api/plot_now', { method: 'POST' });
        const result = await response.json();
        
        if (response.ok) {
            await fetchPlots();
        } else {
            alert("Could not generate plot: " + (result.error || "Unknown error"));
        }
    } catch (error) {
        console.error("Error generating plot:", error);
        alert("Error generating plot. Check console.");
    } finally {
        btn.innerHTML = originalHtml;
        btn.disabled = false;
    }
}

// Fetch live metrics
async function fetchLiveMetrics() {
    try {
        const response = await fetch('/api/live_metrics');
        const data = await response.json();
        
        if (data.error) {
            document.getElementById('metrics-step').textContent = data.error;
            return;
        }

        // Update Step
        document.getElementById('metrics-step').textContent = `Step ${data.step}`;

        // Traffic & CPU
        document.getElementById('val-traffic').textContent = parseFloat(data.request_rate).toFixed(1);
        document.getElementById('val-cpu').textContent = parseFloat(data.cpu_util).toFixed(2);

        // Latency
        const latency = parseFloat(data.p99_latency);
        const latEl = document.getElementById('val-latency');
        latEl.textContent = latency.toFixed(0) + " ms";
        if (latency > 200) {
            latEl.classList.add('danger');
            latEl.classList.remove('success');
        } else {
            latEl.classList.add('success');
            latEl.classList.remove('danger');
        }

        // Replicas
        document.getElementById('val-replicas').textContent = data.replicas;
        document.getElementById('val-pending').textContent = `${data.pending_pods} Pending`;

        // Action
        const delta = parseInt(data.safe_delta);
        const actionEl = document.getElementById('val-action');
        if (delta > 0) {
            actionEl.innerHTML = `<span class="material-icons" style="font-size:24px; vertical-align:-4px;">arrow_upward</span> Scaled Up (+${delta})`;
            actionEl.className = 'metric-value success';
        } else if (delta < 0) {
            actionEl.innerHTML = `<span class="material-icons" style="font-size:24px; vertical-align:-4px;">arrow_downward</span> Scaled Down (${delta})`;
            actionEl.className = 'metric-value danger';
        } else {
            actionEl.innerHTML = `<span class="material-icons" style="font-size:24px; vertical-align:-4px;">horizontal_rule</span> Maintained`;
            actionEl.className = 'metric-value';
        }

        document.getElementById('val-proposed').textContent = `Proposed by RL: ${data.proposed_delta}`;

        // Breach
        document.getElementById('val-breach').textContent = data.sla_breach === "True" ? "Yes" : "No";

    } catch (error) {
        console.error("Error fetching live metrics:", error);
    }
}

function toggleConsole() {
    const el = document.getElementById('terminal-output');
    if (el.style.display === 'none') {
        el.style.display = 'block';
    } else {
        el.style.display = 'none';
    }
}

// Add CSS keyframes dynamically for the spinner
const style = document.createElement('style');
style.innerHTML = `
@keyframes spin { 100% { transform: rotate(360deg); } }
`;
document.head.appendChild(style);

// Initialization
updateStatus();
fetchPlots();
fetchLogs();
fetchLiveMetrics();

// Polling intervals
setInterval(updateStatus, 2000);
setInterval(fetchLogs, 2000); // Poll terminal logs every 2 seconds
setInterval(fetchLiveMetrics, 2000); // Poll metrics
setInterval(fetchPlots, 30000);
