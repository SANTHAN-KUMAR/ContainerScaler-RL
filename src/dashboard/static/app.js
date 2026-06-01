// ── SHARED UTILITIES ───────────────────────────────────────────────────────────

let currentStatus = {};

// Format numbers securely
function formatNumber(num, dec = 1) {
    const val = parseFloat(num);
    return isNaN(val) ? "0" : val.toFixed(dec);
}

// Render Pod Grid
function renderCluster(containerId, replicas, pending = 0) {
    const grid = document.getElementById(containerId);
    if (!grid) return;
    
    // Quick dirty diffing
    const currentTotal = grid.children.length;
    const targetTotal = replicas + pending;
    
    if (currentTotal === targetTotal && grid.dataset.lastReps == replicas) return;
    grid.dataset.lastReps = replicas;
    
    grid.innerHTML = '';
    
    // Ready pods
    for(let i=0; i<replicas; i++) {
        const node = document.createElement('div');
        node.className = 'pod ready';
        node.innerHTML = `<svg viewBox="0 0 24 24"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16zM12 22.27 4.5 18V9.73L12 14l7.5-4.27V18zM12 11.73 4.5 7.5 12 3.27l7.5 4.23z"/></svg>`;
        grid.appendChild(node);
    }
    
    // Pending pods
    for(let i=0; i<pending; i++) {
        const node = document.createElement('div');
        node.className = 'pod pending';
        node.innerHTML = `<svg viewBox="0 0 24 24"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16zM12 22.27 4.5 18V9.73L12 14l7.5-4.27V18zM12 11.73 4.5 7.5 12 3.27l7.5 4.23z"/></svg>`;
        grid.appendChild(node);
    }
}

// ── GLOBAL STATUS POLL ───────────────────────────────────────────────────────

async function updateStatus() {
    try {
        const response = await fetch('/api/status');
        currentStatus = await response.json();

        const anyRunning = Object.values(currentStatus).some(v => v);
        const dot = document.getElementById('global-status-dot');
        const text = document.getElementById('global-status-text');
        
        if (dot && text) {
            if (anyRunning) {
                dot.classList.add('live');
                text.textContent = 'System Active';
            } else {
                dot.classList.remove('live');
                text.textContent = 'Standby';
            }
        }
        
        // Dispatch event so specific pages can update their buttons
        document.dispatchEvent(new CustomEvent('statusUpdated', { detail: currentStatus }));
    } catch (error) {
        console.error("Status fetch failed", error);
    }
}

// Toggle backend service
async function toggleService(service) {
    const isRunning = currentStatus[service];
    const endpoint = isRunning ? `/api/stop/${service}` : `/api/start/${service}`;
    
    let options = { method: 'POST' };
    
    // Inject parameters from UI if we are starting a service
    if (!isRunning) {
        let payload = {};
        if (service === 'locust') {
            const profile = document.getElementById('trafficProfile');
            if (profile) payload.traffic_profile = profile.value;
        } else if (service === 'live_agent') {
            const model = document.getElementById('agentModel');
            if (model) payload.model = model.value;
        }
        
        if (Object.keys(payload).length > 0) {
            options.headers = { 'Content-Type': 'application/json' };
            options.body = JSON.stringify(payload);
        }
    }
    
    try {
        let res = await fetch(endpoint, options);
        if (!res.ok) {
            let data = await res.json();
            throw new Error(data.error);
        }
        await updateStatus();
    } catch (error) {
        alert(`Action failed:\n${error.message}`);
    }
}

// Run once on load, then poll
updateStatus();
setInterval(updateStatus, 2000);
