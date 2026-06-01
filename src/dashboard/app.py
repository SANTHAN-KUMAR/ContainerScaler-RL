"""
ContainerScaler-RL Dashboard — Mission Control API

Routes:
  /               → Mission Briefing page
  /live           → Live Command Center
  /simulator      → Simulation Arena
  /analytics      → Debrief & Analytics

  /api/status             → process status
  /api/start/<svc>        → start a service process
  /api/stop/<svc>         → stop a service process
  /api/stop_all           → stop everything
  /api/logs/<svc>         → service log tail
  /api/live_metrics       → latest live metrics from CSV

  /api/simulate/run       → run full episode, return trajectory JSON
  /api/simulate/stream    → SSE streaming of simulation steps
  /api/analytics/sessions → list completed simulation sessions
  /api/analytics/session/<id> → get full session data
"""

import csv
import json
import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from io import StringIO
from pathlib import Path
from typing import Any, Generator

import psutil
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from src.live.observer import PrometheusObserver
from src.safety.safety_filter import ClusterState

# Global observer for live metrics when agent is not running
_global_observer = None

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Process registry ─────────────────────────────────────────────────────────
processes: dict[str, Any] = {
    "live_agent": None,
    "locust": None,
    "evaluation": None,
}

# NodePort endpoints — no port-forwards needed
PROM_NODEPORT    = "http://127.0.0.1:30090"
BOUTIQUE_NODEPORT = "http://127.0.0.1:30808"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
DASHBOARD_LOGS_DIR = PROJECT_ROOT / "logs" / "dashboard"
SIM_SESSIONS_DIR = PROJECT_ROOT / "logs" / "sim_sessions"
DASHBOARD_LOGS_DIR.mkdir(parents=True, exist_ok=True)
SIM_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Active streaming simulation state (one at a time)
_sim_lock = threading.Lock()
_active_sim: dict[str, Any] = {}


# ── Utilities ─────────────────────────────────────────────────────────────────

def kill_process_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def kill_dangling_port_forwards(svc_name: str | None = None) -> None:
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline")
            if cmdline and "port-forward" in cmdline:
                if svc_name is None or svc_name in cmdline:
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass


def is_running(proc: Any) -> bool:
    return proc is not None and proc.poll() is None


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("mission.html")


@app.route("/live")
def live_page():
    return render_template("live.html")


@app.route("/simulator")
def simulator_page():
    return render_template("simulator.html")


@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html")


# ── Service control API ───────────────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def get_status():
    status = {key: is_running(proc) for key, proc in processes.items()}
    # Add connectivity status for NodePort services (no process needed)
    status["boutique_live"] = _check_nodeport(BOUTIQUE_NODEPORT)
    status["prometheus_live"] = _check_nodeport(PROM_NODEPORT)
    return jsonify(status)


def _check_nodeport(url: str) -> bool:
    """Quick TCP-level check if a NodePort is reachable.
    Relaxes timeout requirements to prevent false 'Offline' alarms during heavy DDOS load.
    """
    import socket
    from urllib.parse import urlparse
    parsed = urlparse(url)
    try:
        s = socket.create_connection((parsed.hostname, parsed.port), timeout=5)
        s.close()
        return True
    except socket.timeout:
        # If it times out, the port is open but the server is too congested to reply quickly.
        # This means the cluster is ALIVE and struggling, which is exactly what we want to show!
        return True
    except ConnectionRefusedError:
        # Port is actually closed (Boutique is not deployed or NodePort is wrong).
        return False
    except OSError:
        return False


@app.route("/api/start/<service>", methods=["POST"])
def start_service(service: str):
    if service not in processes:
        return jsonify({"error": "Unknown service"}), 400

    if is_running(processes[service]):
        return jsonify({"message": f"{service} is already running"}), 200

    env = os.environ.copy()
    cwd = str(PROJECT_ROOT)
    log_file_path = DASHBOARD_LOGS_DIR / f"{service}.log"
    log_file = open(log_file_path, "w")

    try:
        python_cmd = sys.executable
        
        # Parse optional JSON payload for configuration
        payload = {}
        if request.is_json:
            payload = request.get_json()

        if service == "live_agent":
            model = payload.get("model", "ensemble")
            cmd = [
                python_cmd, "-m", "src.live.live_agent",
                "--prom", PROM_NODEPORT,
                "--namespace", "default",
                "--deployment", "frontend",   # Online Boutique frontend
                "--model", model,
                "--steps", "720",
                "--interval", "5",
                "--name", "live_test_run",
            ]
        elif service == "locust":
            traffic_profile = payload.get("traffic_profile", "steady")
            env["TRAFFIC_PROFILE"] = traffic_profile
            # Realistic e-commerce user journeys against the boutique via Traefik Ingress
            cmd = [
                python_cmd, "-m", "locust", "-f", "deploy/locustfile.py",
                "--headless", "-u", "200", "-r", "30",
                "--run-time", "10m", "--host", "http://127.0.0.1:80",
            ]
        elif service == "evaluation":
            cmd = [python_cmd, "-m", "src.evaluation.live_experiment", "--mode", "sim"]
        else:
            return jsonify({"error": "Unknown service"}), 400

        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes[service] = proc
        return jsonify({"message": f"Started {service}"}), 200
    except Exception as e:
        logger.error(f"Error starting {service}: {e}")
        log_file.close()
        return jsonify({"error": str(e)}), 500


@app.route("/api/stop/<service>", methods=["POST"])
def stop_service(service: str):
    if service not in processes:
        return jsonify({"error": "Unknown service"}), 400
    proc = processes[service]
    if proc and is_running(proc):
        kill_process_tree(proc.pid)
        processes[service] = None
        return jsonify({"message": f"Stopped {service}"}), 200
    return jsonify({"message": f"{service} is not running"}), 200


@app.route("/api/stop_all", methods=["POST"])
def stop_all():
    for service, proc in processes.items():
        if proc and is_running(proc):
            kill_process_tree(proc.pid)
            processes[service] = None
    kill_dangling_port_forwards()

    with _sim_lock:
        _active_sim["cancelled"] = True

    return jsonify({"message": "Stopped all services"}), 200


@app.route("/api/logs/<service>", methods=["GET"])
def get_logs(service: str):
    log_file_path = DASHBOARD_LOGS_DIR / f"{service}.log"
    if not log_file_path.exists():
        return jsonify({"logs": "No logs yet."})
    try:
        with open(log_file_path) as f:
            lines = f.readlines()
            return jsonify({"logs": "".join(lines[-100:])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/live_metrics", methods=["GET"])
def get_live_metrics():
    """Return live metrics for the dashboard.
    
    If the RL agent is running, we return the latest row from its CSV log 
    so the dashboard shows exact agent decisions (proposed delta, safe delta).
    
    If the agent is NOT running, we fetch metrics directly from the cluster 
    via a local PrometheusObserver so the dashboard stays "Live" permanently!
    """
    global _global_observer
    
    agent_running = is_running(processes.get("live_agent"))
    live_logs_dir = PROJECT_ROOT / "logs" / "live"
    
    latest_csv = None
    if live_logs_dir.exists():
        csv_files = [p for p in live_logs_dir.glob("live_test_run_*.csv")
                     if not p.name.endswith("_meta.json")]
        if csv_files:
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)

    # If agent is running AND we have a fresh CSV, use it for exact logs
    if agent_running and latest_csv and (time.time() - latest_csv.stat().st_mtime < 30):
        try:
            with open(latest_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    return jsonify(rows[-1])
        except Exception as e:
            logger.error(f"Failed to read live CSV: {e}")
            
    # Otherwise, the agent is OFF or starting up. Fetch metrics live directly!
    try:
        if _global_observer is None:
            _global_observer = PrometheusObserver(prom_url=PROM_NODEPORT)
            
        obs = _global_observer.get_state()
        state = ClusterState.from_obs(obs)
        
        # Construct a synthetic row that mimics the CSV format so the UI works
        synthetic_row = {
            "step": "-",
            "replicas": state.replicas,
            "p99_latency": state.p99_latency,
            "request_rate": state.request_rate,
            "cpu_util": state.cpu_util,
            "mem_util": state.mem_util,
            "queue_depth": state.queue_depth,
            "proposed_delta": 0,
            "safe_delta": 0,
            "source": "none",
            "reward": 0.0
        }
        return jsonify(synthetic_row)
    except Exception as e:
        return jsonify({"error": f"Failed to fetch direct live metrics: {e}"}), 500


@app.route("/plots/<path:filename>")
def serve_plot(filename: str):
    return send_from_directory(str(PROJECT_ROOT / "plots"), filename)


@app.route("/boutique/", defaults={"path": ""}, methods=["GET", "POST", "HEAD"])
@app.route("/boutique/<path:path>", methods=["GET", "POST", "HEAD"])
def boutique_proxy(path: str):
    """Reverse-proxy the Online Boutique frontend through Flask.

    Uses the NodePort (30808) directly — no port-forward process needed.
    Routing through this Flask server (same origin) bypasses browser
    cross-origin iframe restrictions.
    """
    import urllib.request
    import urllib.error

    target_url = f"{BOUTIQUE_NODEPORT}/{path}"
    if request.query_string:
        target_url += f"?{request.query_string.decode('utf-8', errors='replace')}"

    _OFFLINE_HTML = (
        "<html><body style='background:#0a0a14;color:#888;font-family:sans-serif;"
        "display:flex;height:100vh;align-items:center;justify-content:center;'>"
        "<div style='text-align:center'><h2>🛍️ Boutique Offline</h2>"
        "<p>Deploy the Online Boutique to your K3s cluster first:</p>"
        "<code style='color:#60a5fa'>kubectl apply -f deploy/online-boutique.yaml</code></div></body></html>"
    )

    if not _check_nodeport(BOUTIQUE_NODEPORT):
        return _OFFLINE_HTML, 503

    # ── Proxy request ─────────────────────────────────────────────────────────
    try:
        req = urllib.request.Request(
            target_url,
            data=request.get_data() or None,
            method=request.method,
            headers={k: v for k, v in request.headers if k.lower() not in
                     ("host", "transfer-encoding", "connection")},
        )
        try:
            resp_ctx = urllib.request.urlopen(req, timeout=15)
            status = resp_ctx.status
            content = resp_ctx.read()
            content_type = resp_ctx.headers.get("Content-Type", "text/html")
        except urllib.error.HTTPError as http_err:
            # Pass the boutique's own error pages through (e.g. 404, 500)
            status = http_err.code
            content = http_err.read()
            content_type = http_err.headers.get("Content-Type", "text/html")

        # Rewrite absolute URLs in HTML so links stay within the proxy
        if "text/html" in content_type:
            content = content.replace(
                b'action="/', b'action="/boutique/'
            ).replace(
                b'href="/', b'href="/boutique/'
            ).replace(
                b'src="/', b'src="/boutique/'
            )

        from flask import Response as FlaskResponse
        return FlaskResponse(content, status=status, content_type=content_type)

    except urllib.error.URLError as e:
        logger.error("Boutique proxy connection error for %s: %s", target_url, e)
        return _OFFLINE_HTML, 502
    except Exception as e:
        logger.error("Boutique proxy unexpected error: %s", e)
        return _OFFLINE_HTML, 500


@app.route("/api/plots", methods=["GET"])
def get_plots():
    plot_dir = PROJECT_ROOT / "plots"
    if not plot_dir.exists():
        return jsonify([])
    plots = [f.name for f in plot_dir.iterdir() if f.suffix == ".png"]
    plots.sort(key=lambda x: (plot_dir / x).stat().st_mtime, reverse=True)
    return jsonify(plots)


# ── Simulation API ────────────────────────────────────────────────────────────

def _action_label(delta: int) -> str:
    if delta > 0:
        return f"Scale Up +{delta}"
    if delta < 0:
        return f"Scale Down {delta}"
    return "Hold Steady"


def _run_episode(scenario: str, agent_type: str, seed: int, fast: bool = False) -> list[dict]:
    """Run a full 120-step simulation episode and return the trajectory."""
    from src.env.k8s_sim import K8sSimEnv

    env = K8sSimEnv(workload_pattern=scenario, seed=seed)
    obs, info = env.reset()

    # Try to load the RL model
    rl_model = None
    if agent_type in ("ppo", "dqn", "qrdqn"):
        try:
            from stable_baselines3 import PPO, DQN
            model_map = {
                "ppo": PROJECT_ROOT / "ppo_autoscaler",
                "dqn": PROJECT_ROOT / "dqn_autoscaler",
            }
            if agent_type in model_map:
                model_path = model_map[agent_type]
                if agent_type == "ppo":
                    rl_model = PPO.load(str(model_path))
                elif agent_type == "dqn":
                    rl_model = DQN.load(str(model_path))
        except Exception as e:
            logger.warning(f"Could not load {agent_type} model: {e} — falling back to HPA")

    # HPA baseline agent
    def hpa_action(env_state: K8sSimEnv) -> int:
        """Simple HPA: scale based on CPU utilization threshold."""
        cpu = env_state.cpu_util
        reps = env_state.replicas
        if cpu > 0.80:
            delta = min(3, max(1, int((cpu - 0.80) * 10)))
        elif cpu < 0.30 and reps > env_state.min_replicas:
            delta = -1
        else:
            delta = 0
        return delta + 3  # map to action space [0..6]

    trajectory = []
    cumulative_reward = 0.0
    sla_breaches = 0

    for step in range(env.episode_length):
        # Choose action
        if rl_model is not None:
            action, _ = rl_model.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = hpa_action(env)

        obs, reward, terminated, truncated, info = env.step(action)
        delta = action - 3
        cumulative_reward += reward
        if info["sla_breach"]:
            sla_breaches += 1

        step_data = {
            "step": step,
            "request_rate": round(info["request_rate"], 1),
            "p99_latency": round(info["p99_latency"], 1),
            "replicas": info["replicas"],
            "pending_pods": info["pending_pods"],
            "cpu_util": round(info["cpu_util"] * 100, 1),
            "mem_util": round(info["mem_util"] * 100, 1),
            "queue_depth": round(info["queue_depth"], 1),
            "cost_rate": round(info["cost_rate"], 3),
            "action": delta,
            "action_label": _action_label(delta),
            "sla_breach": info["sla_breach"],
            "reward": round(reward, 4),
            "cumulative_reward": round(cumulative_reward, 4),
            "sla_breaches_so_far": sla_breaches,
            "workload_pattern": info["workload_pattern"],
            "agent_type": agent_type,
        }
        trajectory.append(step_data)

        if terminated or truncated:
            break

    return trajectory


def _compute_scorecard(trajectory: list[dict], agent_type: str, scenario: str) -> dict:
    """Compute a scorecard from the trajectory."""
    if not trajectory:
        return {}

    sla_breaches = sum(1 for s in trajectory if s["sla_breach"])
    total_steps = len(trajectory)
    sla_pct = (1 - sla_breaches / total_steps) * 100
    avg_cost = sum(s["cost_rate"] for s in trajectory) / total_steps
    avg_latency = sum(s["p99_latency"] for s in trajectory) / total_steps
    max_latency = max(s["p99_latency"] for s in trajectory)
    avg_replicas = sum(s["replicas"] for s in trajectory) / total_steps
    total_reward = trajectory[-1]["cumulative_reward"]

    # Grade: S/A/B/C/F
    if sla_pct >= 99 and avg_cost < 1.5:
        grade = "S"
    elif sla_pct >= 95:
        grade = "A"
    elif sla_pct >= 85:
        grade = "B"
    elif sla_pct >= 70:
        grade = "C"
    else:
        grade = "F"

    return {
        "grade": grade,
        "sla_compliance_pct": round(sla_pct, 1),
        "sla_breaches": sla_breaches,
        "avg_cost_per_hour": round(avg_cost, 3),
        "avg_latency_ms": round(avg_latency, 1),
        "max_latency_ms": round(max_latency, 1),
        "avg_replicas": round(avg_replicas, 1),
        "total_reward": round(total_reward, 4),
        "total_steps": total_steps,
        "agent_type": agent_type,
        "scenario": scenario,
    }


@app.route("/api/simulate/run", methods=["POST"])
def simulate_run():
    """Run a full simulation episode and return the complete trajectory."""
    data = request.get_json(force=True) or {}
    scenario = data.get("scenario", "diurnal")
    agent_type = data.get("agent", "ppo")
    seed = int(data.get("seed", 42))

    try:
        trajectory = _run_episode(scenario, agent_type, seed, fast=True)
        scorecard = _compute_scorecard(trajectory, agent_type, scenario)

        # Save session
        session_id = str(uuid.uuid4())[:8]
        session_data = {
            "session_id": session_id,
            "scenario": scenario,
            "agent": agent_type,
            "seed": seed,
            "timestamp": time.time(),
            "scorecard": scorecard,
            "trajectory": trajectory,
        }
        session_file = SIM_SESSIONS_DIR / f"{session_id}.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f)

        return jsonify({
            "session_id": session_id,
            "scorecard": scorecard,
            "trajectory": trajectory,
        })
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulate/stream")
def simulate_stream():
    """SSE stream — runs simulation step by step, emitting each step as an event."""
    scenario = request.args.get("scenario", "diurnal")
    agent_type = request.args.get("agent", "ppo")
    seed = int(request.args.get("seed", "42"))
    speed = float(request.args.get("speed", "0.05"))  # seconds per step

    def generate() -> Generator[str, None, None]:
        from src.env.k8s_sim import K8sSimEnv

        session_id = str(uuid.uuid4())[:8]
        yield f"data: {json.dumps({'type': 'init', 'session_id': session_id, 'scenario': scenario, 'agent': agent_type})}\n\n"

        env = K8sSimEnv(workload_pattern=scenario, seed=seed)
        obs, _ = env.reset()

        ensemble_model = None
        rl_model = None
        if agent_type == "ensemble":
            try:
                from src.agents.ensemble_agent import EnsembleMetaAgent
                ensemble_model = EnsembleMetaAgent()
            except Exception as e:
                logger.warning(f"Ensemble load failed: {e}")
        elif agent_type in ("ppo", "dqn"):
            try:
                from stable_baselines3 import PPO, DQN
                model_map = {
                    "ppo": PROJECT_ROOT / "ppo_autoscaler",
                    "dqn": PROJECT_ROOT / "dqn_autoscaler",
                }
                mp = model_map.get(agent_type)
                if mp and mp.exists():
                    rl_model = PPO.load(str(mp)) if agent_type == "ppo" else DQN.load(str(mp))
            except Exception as e:
                logger.warning(f"Model load failed: {e}")

        def hpa_action(e: K8sSimEnv) -> int:
            cpu = e.cpu_util
            reps = e.replicas
            if cpu > 0.80:
                return min(3, max(1, int((cpu - 0.80) * 10))) + 3
            elif cpu < 0.30 and reps > e.min_replicas:
                return 2  # delta -1
            return 3  # hold

        trajectory = []
        cumulative_reward = 0.0
        sla_breaches = 0

        with _sim_lock:
            _active_sim["cancelled"] = False

        for step in range(env.episode_length):
            with _sim_lock:
                if _active_sim.get("cancelled"):
                    yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                    return

            if ensemble_model:
                delta = ensemble_model.decide(obs, step)
                action = delta + 3
            elif rl_model:
                action = int(rl_model.predict(obs, deterministic=True)[0])
            else:
                action = hpa_action(env)

            obs, reward, _, truncated, info = env.step(action)
            delta = action - 3
            cumulative_reward += reward
            if info["sla_breach"]:
                sla_breaches += 1

            step_data = {
                "type": "step",
                "step": step,
                "request_rate": round(info["request_rate"], 1),
                "p99_latency": round(info["p99_latency"], 1),
                "replicas": info["replicas"],
                "pending_pods": info["pending_pods"],
                "cpu_util": round(info["cpu_util"] * 100, 1),
                "mem_util": round(info["mem_util"] * 100, 1),
                "queue_depth": round(info["queue_depth"], 1),
                "cost_rate": round(info["cost_rate"], 3),
                "action": delta,
                "action_label": _action_label(delta),
                "sla_breach": info["sla_breach"],
                "reward": round(reward, 4),
                "cumulative_reward": round(cumulative_reward, 4),
                "sla_breaches_so_far": sla_breaches,
                "workload_pattern": info["workload_pattern"],
            }
            trajectory.append(step_data)
            yield f"data: {json.dumps(step_data)}\n\n"

            if speed > 0:
                time.sleep(speed)

            if truncated:
                break

        scorecard = _compute_scorecard(trajectory, agent_type, scenario)

        # Save session
        session_data = {
            "session_id": session_id,
            "scenario": scenario,
            "agent": agent_type,
            "seed": seed,
            "timestamp": time.time(),
            "scorecard": scorecard,
            "trajectory": trajectory,
        }
        session_file = SIM_SESSIONS_DIR / f"{session_id}.json"
        with open(session_file, "w") as f:
            json.dump(session_data, f)

        yield f"data: {json.dumps({'type': 'done', 'scorecard': scorecard, 'session_id': session_id})}\n\n"

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route("/api/simulate/cancel", methods=["POST"])
def simulate_cancel():
    with _sim_lock:
        _active_sim["cancelled"] = True
    return jsonify({"message": "Simulation cancelled"})


@app.route("/api/simulate/benchmark", methods=["GET"])
def simulate_benchmark():
    agent_type = request.args.get("agent", "ensemble")

    def generate() -> Generator[str, None, None]:
        scenarios = ["flash_crowd", "diurnal", "sawtooth", "double_peak"]
        yield f"data: {json.dumps({'type': 'init', 'scenarios': scenarios})}\n\n"

        from src.env.k8s_sim import K8sSimEnv
        from src.evaluation.benchmark_runner import run_episode_extended
        from src.agents.hpa_baseline import RealisticHPA

        agent = None
        if agent_type == "ensemble":
            from src.agents.ensemble_agent import EnsembleMetaAgent
            agent = EnsembleMetaAgent()
        elif agent_type == "hpa":
            agent = RealisticHPA()
        else:
            from src.agents.agent import ContainerScaleAgent
            try:
                agent = ContainerScaleAgent(model_path=f"{agent_type}_autoscaler")
            except:
                agent = RealisticHPA() # fallback

        results = {}

        for scen in scenarios:
            with _sim_lock:
                if _active_sim.get("cancelled"):
                    yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                    return

            env = K8sSimEnv(workload_pattern=scen, seed=42)
            env.reset(seed=42)
            if hasattr(agent, 'reset'):
                agent.reset()

            yield f"data: {json.dumps({'type': 'scenario_start', 'scenario': scen})}\n\n"

            # Use existing evaluation code cleanly as requested
            eval_type = "rl" if hasattr(agent, 'decide') else "hpa"
            metrics = run_episode_extended(env, agent, agent_type=eval_type)

            results[scen] = {
                "sla_compliance": round(metrics.get("sla_compliance", 0), 1),
                "avg_cost": round(metrics.get("avg_cost", 0), 2),
                "max_latency": round(metrics.get("max_latency", 0), 0),
                "over_provisioning": round(metrics.get("over_provisioning", 0), 1),
                "composite": round(metrics.get("composite", 0), 1)
            }

            yield f"data: {json.dumps({'type': 'scenario_done', 'scenario': scen, 'metrics': results[scen]})}\n\n"
            # Small delay so UI can show the transition
            time.sleep(0.5)

        yield f"data: {json.dumps({'type': 'done', 'results': results})}\n\n"

    return Response(generate(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


# ── Analytics API ─────────────────────────────────────────────────────────────

@app.route("/api/analytics/sessions", methods=["GET"])
def list_sessions():
    sessions = []
    for f in sorted(SIM_SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(f) as fp:
                data = json.load(fp)
            sessions.append({
                "session_id": data.get("session_id"),
                "scenario": data.get("scenario"),
                "agent": data.get("agent"),
                "timestamp": data.get("timestamp"),
                "scorecard": data.get("scorecard"),
            })
        except Exception:
            pass
    return jsonify(sessions[:20])  # last 20 sessions


@app.route("/api/analytics/session/<session_id>", methods=["GET"])
def get_session(session_id: str):
    session_file = SIM_SESSIONS_DIR / f"{session_id}.json"
    if not session_file.exists():
        return jsonify({"error": "Session not found"}), 404
    try:
        with open(session_file) as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
