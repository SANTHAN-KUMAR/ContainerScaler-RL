// ── ANALYTICS PAGE LOGIC ─────────────────────────────────────────────────────

async function loadSessions() {
    const list = document.getElementById('session-list');
    list.innerHTML = '<div class="empty-state">Loading...</div>';
    
    try {
        const res = await fetch('/api/analytics/sessions');
        const sessions = await res.json();
        
        if (sessions.length === 0) {
            list.innerHTML = '<div class="empty-state">No simulation data found.<br>Run simulations in the Simulator Arena first.</div>';
            return;
        }
        
        list.innerHTML = '';
        sessions.forEach(s => {
            const el = document.createElement('div');
            el.className = 'session-item';
            el.onclick = () => loadSessionDetail(s.session_id);
            
            const date = new Date(s.timestamp * 1000).toLocaleString();
            
            el.innerHTML = `
                <div class="session-grade grade-${s.scorecard.grade}">${s.scorecard.grade}</div>
                <div class="session-info">
                    <div class="session-name">${s.scenario.toUpperCase()} / ${s.agent.toUpperCase()}</div>
                    <div class="session-meta">${date}</div>
                </div>
                <div class="muted" style="font-family: var(--font-mono); font-size: 11px;">
                    ${s.scorecard.sla_compliance_pct}% SLA
                </div>
            `;
            list.appendChild(el);
        });
        
    } catch (e) {
        list.innerHTML = `<div class="empty-state" style="color:var(--red)">Failed to load data.</div>`;
    }
}

async function loadSessionDetail(id) {
    document.getElementById('empty-panel').style.display = 'none';
    const panel = document.getElementById('detail-panel');
    panel.style.display = 'block';
    
    try {
        const res = await fetch(`/api/analytics/session/${id}`);
        const data = await res.json();
        
        const score = data.scorecard;
        
        document.getElementById('det-agent').textContent = data.agent.toUpperCase();
        document.getElementById('det-sla').textContent = score.sla_compliance_pct;
        document.getElementById('det-cost').textContent = score.avg_cost_per_hour;
        
        const gradeEl = document.getElementById('det-grade');
        gradeEl.textContent = score.grade;
        gradeEl.className = `metric-value grade-${score.grade}`;
        
        // Render Trajectory
        const log = document.getElementById('det-log');
        log.innerHTML = '';
        
        data.trajectory.forEach(step => {
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            
            let type = 'hold';
            if (step.action > 0) type = 'up';
            if (step.action < 0) type = 'down';
            
            const latColor = step.sla_breach ? 'color: var(--red);' : 'color: var(--text-2);';
            
            entry.innerHTML = `
                <div class="log-step">[${step.step}]</div>
                <div class="log-action ${type}">${step.action_label}</div>
                <div style="width: 80px; text-align: right; ${latColor}">${step.p99_latency}ms</div>
                <div class="muted" style="width: 60px; text-align: right;">$${step.cost_rate}</div>
            `;
            log.appendChild(entry);
        });
        
    } catch (e) {
        console.error("Detail load failed", e);
    }
}

// Init
loadSessions();
