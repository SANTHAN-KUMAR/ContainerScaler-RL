import sys
import logging
logging.basicConfig(level=logging.DEBUG)

from src.env.k8s_sim import K8sSimEnv
from src.evaluation.benchmark_runner import run_episode_extended
from src.agents.ensemble_agent import EnsembleMetaAgent

print("Loading ensemble...")
agent = EnsembleMetaAgent()
env = K8sSimEnv(workload_pattern="flash_crowd", seed=42)
env.reset(seed=42)
print("Running episode...")
metrics = run_episode_extended(env, agent, agent_type="rl")
print(metrics)
