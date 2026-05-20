"""
Regret Analysis — Evaluates the impact of the Safety Filter (Experiment 5).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

def compute_regret(safe_deltas: list[int], proposed_deltas: list[int]) -> dict[str, float]:
    """Calculate regret metrics between proposed and applied actions.
    
    Parameters
    ----------
    safe_deltas : list[int]
        Actions actually applied to the environment.
    proposed_deltas : list[int]
        Actions proposed by the agent before filtering.
        
    Returns
    -------
    dict
        Intervention rate and average magnitude of modification.
    """
    if not safe_deltas or len(safe_deltas) != len(proposed_deltas):
        return {"intervention_rate": 0.0, "avg_modification": 0.0}
        
    interventions = sum(1 for s, p in zip(safe_deltas, proposed_deltas) if s != p)
    rate = (interventions / len(safe_deltas)) * 100.0
    
    mod_sum = sum(abs(s - p) for s, p in zip(safe_deltas, proposed_deltas))
    avg_mod = mod_sum / len(safe_deltas)
    
    return {
        "intervention_rate": rate,
        "avg_modification": avg_mod
    }

def analyze_logs(csv_path: str) -> None:
    """Analyze a live run's CSV log to compute safety filter regret."""
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"Could not find log file: {csv_path}")
        return
        
    if "safe_delta" not in df.columns or "proposed_delta" not in df.columns:
        logger.error("Log file missing required delta columns.")
        return
        
    regret = compute_regret(df["safe_delta"].tolist(), df["proposed_delta"].tolist())
    
    print("\n=== Safety Filter Regret Analysis ===")
    print(f"Total steps: {len(df)}")
    print(f"Intervention Rate: {regret['intervention_rate']:.2f}%")
    print(f"Average Modification (replicas): {regret['avg_modification']:.2f}")
    
    # Analyze SLA vs interventions
    breaches = df[df["sla_breach"]]
    if len(breaches) > 0:
        breach_interventions = len(breaches[breaches["safe_delta"] != breaches["proposed_delta"]])
        print(f"Interventions during SLA breach: {breach_interventions} / {len(breaches)}")
