// ── SIMULATOR ARENA LOGIC ────────────────────────────────────────────────────

let currentScenario = 'flash_crowd';
let currentAgent = 'ensemble';
let eventSource = null;
let simTimeout = null;

// UI Setup
document.addEventListener('DOMContentLoaded', () => {
    // Scenario selection
    document.querySelectorAll('.scenario-card').forEach(card => {
        card.addEventListener('click', () => {
            document.querySelectorAll('.scenario-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            currentScenario = card.dataset.scenario;
        });
    });

    // Agent selection
    document.querySelectorAll('.agent-option').forEach(opt => {
        opt.addEventListener('click', () => {
            document.querySelectorAll('.agent-option').forEach(o => o.classList.remove('selected'));
            opt.classList.add('selected');
            currentAgent = opt.dataset.agent;
        });
    });
});

// Run Simulation (Stream or Fast-Forward)
async function startSim(fastForward = false) {
    // Reset UI
    document.getElementById('setup-panel').style.display = 'none';
    document.getElementById('scorecard-panel').style.display = 'none';
    document.getElementById('benchmark-panel').style.display = 'none';
    document.getElementById('active-panel').style.display = 'block';
    
    document.getElementById('sim-title').textContent = `RUNNING: ${currentScenario.toUpperCase()} / ${currentAgent.toUpperCase()}`;
    document.getElementById('sim-progress').style.width = '0%';
    document.getElementById('sim-log').innerHTML = '';
    
    if (eventSource) {
        eventSource.close();
    }
    
    if (fastForward) {
        await runFastForward();
    } else {
        runStream();
    }
}

async function runFastForward() {
    try {
        const res = await fetch('/api/simulate/run', {
            method: 'POST',
            body: JSON.stringify({
                scenario: currentScenario,
                agent: currentAgent,
                seed: Math.floor(Math.random() * 1000)
            })
        });
        
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        
        // Playback the trajectory smoothly
        let step = 0;
        const traj = data.trajectory;
        
        function tick() {
            if (step >= traj.length) {
                showScorecard(data.scorecard);
                return;
            }
            updateSimUI(traj[step]);
            step++;
            simTimeout = setTimeout(tick, 50); // Fast playback speed
        }
        tick();
        
    } catch (e) {
        alert("Sim error: " + e.message);
        resetSim();
    }
}

function runStream() {
    const speed = (200 - document.getElementById('speed-slider').value) / 1000.0; // convert to seconds
    const seed = Math.floor(Math.random() * 1000);
    
    eventSource = new EventSource(`/api/simulate/stream?scenario=${currentScenario}&agent=${currentAgent}&seed=${seed}&speed=${speed}`);
    
    eventSource.onmessage = (e) => {
        const data = JSON.parse(e.data);
        
        if (data.type === 'step') {
            updateSimUI(data);
        } else if (data.type === 'done') {
            eventSource.close();
            showScorecard(data.scorecard);
        } else if (data.type === 'cancelled') {
            eventSource.close();
            resetSim();
        }
    };
    
    eventSource.onerror = () => {
        eventSource.close();
        alert("Streaming connection lost.");
        resetSim();
    };
}

async function cancelSim() {
    if (eventSource) {
        eventSource.close();
    }
    clearTimeout(simTimeout);
    await fetch('/api/simulate/cancel', { method: 'POST' });
    resetSim();
}

function resetSim() {
    document.getElementById('setup-panel').style.display = 'grid';
    document.getElementById('active-panel').style.display = 'none';
    document.getElementById('scorecard-panel').style.display = 'none';
    document.getElementById('benchmark-panel').style.display = 'none';
    document.getElementById('bench-status').className = 'label muted';
}

function startBenchmark() {
    document.getElementById('setup-panel').style.display = 'none';
    document.getElementById('active-panel').style.display = 'none';
    document.getElementById('scorecard-panel').style.display = 'none';
    document.getElementById('benchmark-panel').style.display = 'block';
    
    document.getElementById('bench-agent-name').textContent = currentAgent.toUpperCase();
    document.getElementById('bench-progress').style.width = '0%';
    document.getElementById('bench-status').textContent = 'Initializing benchmark...';
    document.getElementById('bench-tbody').innerHTML = '';
    
    if (eventSource) {
        eventSource.close();
    }
    
    eventSource = new EventSource(`/api/simulate/benchmark?agent=${currentAgent}`);
    
    eventSource.onmessage = (e) => {
        const data = JSON.parse(e.data);
        
        if (data.type === 'init') {
            document.getElementById('bench-status').textContent = `Starting evaluation across ${data.scenarios.length} scenarios...`;
        } else if (data.type === 'scenario_start') {
            document.getElementById('bench-status').textContent = `Evaluating on: ${data.scenario.toUpperCase()}`;
        } else if (data.type === 'scenario_done') {
            const m = data.metrics;
            const tr = document.createElement('tr');
            
            // Format SLA
            const slaClass = m.sla_compliance >= 99 ? 'good' : (m.sla_compliance < 95 ? 'bad' : '');
            
            tr.innerHTML = `
                <td><strong>${data.scenario.toUpperCase()}</strong></td>
                <td class="${slaClass}">${m.sla_compliance}%</td>
                <td>$${m.avg_cost}/hr</td>
                <td>${m.max_latency}ms</td>
                <td>${m.over_provisioning}%</td>
                <td><strong>${m.composite}</strong></td>
            `;
            document.getElementById('bench-tbody').appendChild(tr);
            
            // Progress
            const currentRows = document.getElementById('bench-tbody').children.length;
            document.getElementById('bench-progress').style.width = `${(currentRows / 4) * 100}%`;
            
        } else if (data.type === 'done') {
            eventSource.close();
            document.getElementById('bench-status').textContent = 'Benchmark Complete.';
            document.getElementById('bench-status').className = 'label green';
        } else if (data.type === 'cancelled') {
            eventSource.close();
            resetSim();
        }
    };
    
    eventSource.onerror = () => {
        eventSource.close();
        alert("Benchmark connection lost.");
        resetSim();
    };
}

function updateSimUI(data) {
    // Progress
    const pct = ((data.step + 1) / 120) * 100;
    document.getElementById('sim-progress').style.width = `${pct}%`;
    document.getElementById('sim-time').textContent = `Step ${data.step} / 120`;
    
    // Metrics
    document.getElementById('sim-val-latency').textContent = data.p99_latency;
    document.getElementById('sim-val-cost').textContent = data.cost_rate;
    
    const latCard = document.getElementById('sim-card-latency');
    if (data.sla_breach) {
        latCard.classList.add('breach');
    } else {
        latCard.classList.remove('breach');
    }
    
    // Cluster
    renderCluster('sim-cluster-grid', data.replicas, data.pending_pods);
    
    // Log
    const log = document.getElementById('sim-log');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    
    let type = 'hold';
    if (data.action > 0) type = 'up';
    if (data.action < 0) type = 'down';
    
    entry.innerHTML = `
        <div class="log-step">[${data.step}]</div>
        <div class="log-action ${type}">${data.action_label}</div>
        <div class="muted" style="width: 80px; text-align: right;">${data.p99_latency}ms</div>
    `;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

function showScorecard(score) {
    document.getElementById('active-panel').style.display = 'none';
    document.getElementById('scorecard-panel').style.display = 'block';
    
    const gradeEl = document.getElementById('score-grade');
    gradeEl.textContent = score.grade;
    gradeEl.className = `grade-display grade-${score.grade}`;
    
    document.getElementById('score-sla').textContent = score.sla_compliance_pct;
    document.getElementById('score-breaches').textContent = `${score.sla_breaches} Breaches`;
    
    if (score.sla_breaches > 0) {
        document.getElementById('score-breaches').className = 'metric-sub bad';
    } else {
        document.getElementById('score-breaches').className = 'metric-sub ok';
    }
    
    document.getElementById('score-cost').textContent = score.avg_cost_per_hour;
    document.getElementById('score-max-lat').textContent = score.max_latency_ms;
}
