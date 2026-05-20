import os
import subprocess
import signal
import psutil
from flask import Flask, jsonify, request, render_template, send_from_directory
from pathlib import Path
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track running processes
processes = {
    "port_forward_prom": None,
    "port_forward_podinfo": None,
    "live_agent": None,
    "locust": None,
    "evaluation": None,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_LOGS_DIR = PROJECT_ROOT / "logs" / "dashboard"
DASHBOARD_LOGS_DIR.mkdir(parents=True, exist_ok=True)

def kill_process_tree(pid):
    """Kills a process and all its children."""
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def is_running(proc):
    return proc is not None and proc.poll() is None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    status = {key: is_running(proc) for key, proc in processes.items()}
    return jsonify(status)

@app.route('/api/start/<service>', methods=['POST'])
def start_service(service):
    if service not in processes:
        return jsonify({"error": "Unknown service"}), 400
    
    if is_running(processes[service]):
        return jsonify({"message": f"{service} is already running"}), 200

    env = os.environ.copy()
    cwd = str(PROJECT_ROOT)
    
    log_file_path = DASHBOARD_LOGS_DIR / f"{service}.log"
    log_file = open(log_file_path, "w")

    try:
        if service == "port_forward_prom":
            subprocess.run(["pkill", "-f", "port-forward svc/prometheus"], stderr=subprocess.DEVNULL)
            cmd = ["/usr/local/bin/kubectl", "port-forward", "svc/prometheus", "30090:9090"]
        elif service == "port_forward_podinfo":
            subprocess.run(["pkill", "-f", "port-forward svc/podinfo"], stderr=subprocess.DEVNULL)
            cmd = ["/usr/local/bin/kubectl", "port-forward", "svc/podinfo", "9898:9898"]
        elif service == "live_agent":
            cmd = [
                "./venv/bin/python", "-m", "src.live.live_agent",
                "--prom", "http://localhost:30090",
                "--namespace", "default",
                "--deployment", "podinfo",
                "--model", "ppo_autoscaler",
                "--steps", "120",
                "--name", "live_test_run"
            ]
        elif service == "locust":
            cmd = [
                "./venv/bin/locust", "-f", "deploy/locustfile.py",
                "--headless", "-u", "100", "-r", "10",
                "--run-time", "15m", "--host", "http://localhost:9898"
            ]
        elif service == "evaluation":
            cmd = [
                "./venv/bin/python", "-m", "src.evaluation.live_experiment",
                "--mode", "sim"
            ]

        proc = subprocess.Popen(
            cmd, 
            cwd=cwd, 
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        processes[service] = proc
        return jsonify({"message": f"Started {service}"}), 200
    except Exception as e:
        logger.error(f"Error starting {service}: {e}")
        log_file.close()
        return jsonify({"error": str(e)}), 500

@app.route('/api/stop/<service>', methods=['POST'])
def stop_service(service):
    if service not in processes:
        return jsonify({"error": "Unknown service"}), 400
    
    proc = processes[service]
    if proc and is_running(proc):
        kill_process_tree(proc.pid)
        processes[service] = None
        return jsonify({"message": f"Stopped {service}"}), 200
    
    return jsonify({"message": f"{service} is not running"}), 200

@app.route('/api/stop_all', methods=['POST'])
def stop_all():
    for service, proc in processes.items():
        if proc and is_running(proc):
            kill_process_tree(proc.pid)
            processes[service] = None
    subprocess.run(["pkill", "-f", "port-forward"], stderr=subprocess.DEVNULL)
    return jsonify({"message": "Stopped all services"}), 200

@app.route('/api/logs/<service>', methods=['GET'])
def get_logs(service):
    log_file_path = DASHBOARD_LOGS_DIR / f"{service}.log"
    if not log_file_path.exists():
        return jsonify({"logs": "No logs yet."})
    
    try:
        # Read the last 100 lines
        with open(log_file_path, "r") as f:
            lines = f.readlines()
            return jsonify({"logs": "".join(lines[-100:])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/live_metrics', methods=['GET'])
def get_live_metrics():
    live_logs_dir = PROJECT_ROOT / "logs" / "live"
    if not live_logs_dir.exists():
        return jsonify({"error": "No live logs found."})
        
    csv_files = list(live_logs_dir.glob("live_test_run_*.csv"))
    if not csv_files:
        return jsonify({"error": "No CSV logs found."})
        
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    try:
        import csv
        with open(latest_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return jsonify({"error": "CSV is empty."})
            return jsonify(rows[-1]) # Return the last row as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/plot_now', methods=['POST'])
def plot_now():
    live_logs_dir = PROJECT_ROOT / "logs" / "live"
    if not live_logs_dir.exists():
        return jsonify({"error": "No live logs directory found"}), 404
        
    csv_files = list(live_logs_dir.glob("live_test_run_*.csv"))
    if not csv_files:
        return jsonify({"error": "No CSV logs found"}), 404
        
    # Get the most recent CSV
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    try:
        from src.evaluation.visualize import plot_live_run
        plot_live_run(latest_csv, PROJECT_ROOT / "plots")
        return jsonify({"message": "Plot generated successfully"})
    except Exception as e:
        logger.error(f"Error plotting: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    plot_dir = PROJECT_ROOT / "plots"
    return send_from_directory(str(plot_dir), filename)

@app.route('/api/plots', methods=['GET'])
def get_plots():
    plot_dir = PROJECT_ROOT / "plots"
    if not plot_dir.exists():
        return jsonify([])
    plots = [f.name for f in plot_dir.iterdir() if f.suffix == '.png']
    plots.sort(key=lambda x: (plot_dir / x).stat().st_mtime, reverse=True)
    return jsonify(plots)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
