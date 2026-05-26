"""
Kubernetes RL Autoscaling Visual Simulator (Pygame).

Provides a real-time, interactive, graphical visualizer for the K8sSimEnv simulation.
Allows comparing the RL Agent, HPA Baseline, and manual user controls.

Usage:
    pip install pygame
    python src/evaluation/pygame_visualizer.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import random

# Core simulation imports
from src.env.k8s_sim import K8sSimEnv
from src.agents.agent import ContainerScaleAgent
from src.agents.hpa_baseline import RealisticHPA
from src.safety.safety_filter import ClusterState, SafetyFilter

# Check for pygame dependency
try:
    import pygame
except ImportError:
    print("\n" + "="*80)
    print(" Pygame is required for the visual simulator!")
    print(" Please install it by running:")
    print("     pip install pygame")
    print("="*80 + "\n")
    sys.exit(1)


# Color Palette (Glassmorphism / Sleek Dark Theme)
BG_COLOR = (15, 18, 25)
CARD_COLOR = (26, 32, 44)
CARD_BORDER = (45, 55, 72)
TEXT_PRIMARY = (240, 242, 245)
TEXT_MUTED = (160, 174, 192)

GREEN = (72, 187, 120)       # Ready Pod
YELLOW = (236, 201, 75)     # Pending Pod / Warming Up
RED = (245, 101, 101)       # Latency Breach / Queue Overflow
BLUE = (66, 153, 225)       # Incoming Requests / Normal state
PURPLE = (159, 122, 234)    # RL decision indicator
GRAY = (74, 85, 104)        # Muted nodes / empty slots

# Window Settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800


class Particle:
    """Represents a request flowing from traffic source to queue/pods."""
    def __init__(self, start_x: float, start_y: float, target_x: float, target_y: float, speed: float = 8.0):
        self.x = start_x
        self.y = start_y
        self.target_x = target_x
        self.target_y = target_y
        self.speed = speed
        self.active = True
        
        # Calculate velocity vector
        dx = target_x - start_x
        dy = target_y - start_y
        dist = (dx**2 + dy**2)**0.5
        if dist > 0:
            self.vx = (dx / dist) * speed
            self.vy = (dy / dist) * speed
        else:
            self.vx = 0
            self.vy = 0

    def update(self) -> None:
        self.x += self.vx
        self.y += self.vy
        
        # Check if reached target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = (dx**2 + dy**2)**0.5
        if dist < self.speed:
            self.active = False


class VisualSimulator:
    def __init__(self):
        # 1. Initialize RL Model first (this is slow and blocks the thread)
        self.agent_rl = None
        self.has_rl = False
        print("Loading RL Model (this might take a few seconds)...")
        try:
            # Check if model exists
            model_zip = Path("ppo_autoscaler.zip")
            if model_zip.exists() or Path("checkpoints/ppo_autoscaler_500000_steps.zip").exists():
                model_path = "ppo_autoscaler" if model_zip.exists() else "checkpoints/ppo_autoscaler_500000_steps"
                self.agent_rl = ContainerScaleAgent(model_path=model_path)
                self.has_rl = True
        except Exception as e:
            print(f"Warning: Failed to load RL agent: {e}. RL mode will fall back to HPA.")
            
        self.agent_hpa = RealisticHPA()
        self.safety_filter = SafetyFilter()
        
        # 2. Now initialize pygame and open the display window (will be responsive immediately)
        pygame.init()
        pygame.display.set_caption("ContainerScaler-RL — Interactive Simulator")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("Outfit", 28, bold=True)
        self.font_section = pygame.font.SysFont("Outfit", 20, bold=True)
        self.font_body = pygame.font.SysFont("Outfit", 16)
        self.font_body_bold = pygame.font.SysFont("Outfit", 16, bold=True)
        self.font_large = pygame.font.SysFont("Outfit", 36, bold=True)
        
        # Sim Mode: 'manual', 'hpa', 'rl'
        self.mode = 'rl'
        self.paused = False
        self.step_delay_ms = 400  # speed of auto-step
        self.last_step_time = pygame.time.get_ticks()
        
        self.particles: list[Particle] = []
        self.reset_sim()

    def reset_sim(self) -> None:
        self.env = K8sSimEnv(workload_pattern="diurnal", seed=random.randint(0, 1000))
        self.obs, _ = self.env.reset()
        if self.agent_rl:
            self.agent_rl.reset()
        self.agent_hpa.reset()
        self.safety_filter.reset()
        
        self.history_traffic = []
        self.history_replicas = []
        self.history_latency = []
        
        self.total_cost = 0.0
        self.total_breaches = 0
        self.last_action_desc = "Initialized"
        self.last_action_source = "system"
        self.particles.clear()
        
        # Prefill history with initial values
        self.record_history()

    def record_history(self) -> None:
        self.history_traffic.append(self.env.request_rate)
        self.history_replicas.append(self.env.replicas)
        self.history_latency.append(self.env.p99_latency)
        if len(self.history_traffic) > 120:
            self.history_traffic.pop(0)
            self.history_replicas.pop(0)
            self.history_latency.pop(0)

    def step(self, manual_delta: int = 0) -> None:
        if self.env.step_count >= self.env.episode_length:
            return  # End of episode
            
        # Get action based on mode
        state = ClusterState.from_obs(self.obs)
        if self.mode == 'rl' and self.has_rl:
            decision = self.agent_rl.decide_with_info(self.obs, self.env.step_count)
            delta = decision["delta"]
            self.last_action_source = decision["source"]
        elif self.mode == 'hpa':
            proposed = self.agent_hpa.act(state)
            delta = self.safety_filter.check(state, proposed, self.env.step_count)
            self.last_action_source = "hpa"
        else:  # manual
            delta = self.safety_filter.check(state, manual_delta, self.env.step_count)
            self.last_action_source = "user"
            
        # Map delta to action discrete [0..6] (delta + 3)
        action = delta + 3
        
        # Execute simulator step
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update metrics
        self.total_cost += (self.env.cost_rate * (self.env.step_duration / 3600.0))
        if info["sla_breach"]:
            self.total_breaches += 1
            
        action_sign = "+" if delta > 0 else ""
        self.last_action_desc = f"{action_sign}{delta} Replicas ({self.last_action_source.upper()})"
        
        self.record_history()
        self.spawn_particles()

    def spawn_particles(self) -> None:
        # Spawn some request particles based on current rate
        # Let's map request_rate (e.g. 0-500) to particle count
        num_particles = min(15, int(self.env.request_rate / 30) + 1)
        
        # Spawn traffic particles (Left source -> Queue center)
        # Queue center is around x=300, y=550
        for _ in range(num_particles):
            self.particles.append(Particle(
                start_x=80, start_y=550 + random.randint(-15, 15),
                target_x=280 + random.randint(-10, 10), target_y=550 + random.randint(-20, 20),
                speed=random.uniform(5, 9)
            ))

    def update_particles(self) -> None:
        active_particles = []
        for p in self.particles:
            p.update()
            if p.active:
                active_particles.append(p)
            else:
                # If a particle finished its route to the Queue,
                # let's route it from Queue to one of the active Pod positions!
                if p.target_x < 400: # It reached the queue
                    # Route to a pod
                    if self.env.replicas > 0:
                        pod_index = random.randint(0, self.env.replicas - 1)
                        # Compute pod coordinate
                        node_idx = pod_index // 10
                        slot_idx = pod_index % 10
                        node_x = 650 + node_idx * 180
                        node_y = 150 + slot_idx * 32
                        
                        self.particles.append(Particle(
                            start_x=p.target_x, start_y=p.target_y,
                            target_x=node_x + 25, target_y=node_y + 15,
                            speed=random.uniform(6, 10)
                        ))
        self.particles = active_particles

    def draw_rounded_rect(self, surface, rect, color, radius=8, border=0):
        pygame.draw.rect(surface, color, rect, border, radius)

    def render(self) -> None:
        self.screen.fill(BG_COLOR)
        
        # Draw Header
        header_rect = (20, 20, WINDOW_WIDTH - 40, 60)
        self.draw_rounded_rect(self.screen, header_rect, CARD_COLOR)
        pygame.draw.rect(self.screen, CARD_BORDER, header_rect, 1, 8)
        
        title_text = self.font_title.render("Kubernetes RL Autoscaler Simulator", True, TEXT_PRIMARY)
        self.screen.blit(title_text, (40, 35))
        
        # Mode Indicators (Top Right)
        modes = ['rl', 'hpa', 'manual']
        mode_labels = {'rl': "AI Agent (RL)", 'hpa': "K8s HPA Baseline", 'manual': "Manual Control"}
        mode_colors = {'rl': PURPLE, 'hpa': BLUE, 'manual': YELLOW}
        
        for i, m in enumerate(modes):
            mx = 680 + i * 160
            my = 35
            mw = 140
            mh = 30
            
            m_color = CARD_BORDER if self.mode != m else mode_colors[m]
            self.draw_rounded_rect(self.screen, (mx, my, mw, mh), m_color, radius=6)
            
            lbl = self.font_body_bold.render(mode_labels[m], True, TEXT_PRIMARY if self.mode == m else TEXT_MUTED)
            self.screen.blit(lbl, (mx + (mw - lbl.get_width()) // 2, my + (mh - lbl.get_height()) // 2))

        # --- LEFT PANEL: TRAFFIC WAVES & QUEUE VISUALIZER ---
        # Traffic Panel
        traffic_rect = (20, 100, 540, 320)
        self.draw_rounded_rect(self.screen, traffic_rect, CARD_COLOR)
        pygame.draw.rect(self.screen, CARD_BORDER, traffic_rect, 1, 8)
        
        self.screen.blit(self.font_section.render("Workload Profile (Diurnal)", True, TEXT_PRIMARY), (40, 115))
        
        # Draw diurnal workload wave
        # Step range 0 to 120. Width 480px, Height 180px, starting at (40, 300)
        chart_x, chart_y = 50, 170
        chart_w, chart_h = 480, 200
        
        # Y axis max request rate: 500 RPS
        points = []
        for s in range(121):
            # Compute diurnal rate
            phase = 2 * 3.14159 * s / 120.0
            base = 200.0 + 150.0 * -math_cos(phase)
            # Add some minor noise for visual curve shape
            y_val = chart_y + chart_h - int((base / 500.0) * chart_h)
            x_val = chart_x + int((s / 120.0) * chart_w)
            points.append((x_val, y_val))
            
        if len(points) > 1:
            pygame.draw.lines(self.screen, GRAY, False, points, 2)
            
        # Draw current steps progress line
        curr_step = self.env.step_count
        curr_x = chart_x + int((curr_step / 120.0) * chart_w)
        pygame.draw.line(self.screen, BLUE, (curr_x, chart_y), (curr_x, chart_y + chart_h), 2)
        
        # Draw live traffic line in history
        history_points = []
        for i, val in enumerate(self.history_traffic):
            step_idx = max(0, curr_step - len(self.history_traffic) + i + 1)
            x_val = chart_x + int((step_idx / 120.0) * chart_w)
            y_val = chart_y + chart_h - int((val / 500.0) * chart_h)
            history_points.append((x_val, y_val))
        if len(history_points) > 1:
            pygame.draw.lines(self.screen, BLUE, False, history_points, 3)

        # Traffic stats
        traffic_txt = self.font_body.render(f"Incoming Load: {self.env.request_rate:.0f} RPS", True, TEXT_PRIMARY)
        self.screen.blit(traffic_txt, (350, 115))

        # Queue depth panel
        queue_panel_rect = (20, 440, 540, 340)
        self.draw_rounded_rect(self.screen, queue_panel_rect, CARD_COLOR)
        pygame.draw.rect(self.screen, CARD_BORDER, queue_panel_rect, 1, 8)
        self.screen.blit(self.font_section.render("Live Traffic Queue & Buffer", True, TEXT_PRIMARY), (40, 455))
        
        # Draw generator
        pygame.draw.circle(self.screen, BLUE, (80, 550), 30)
        gen_lbl = self.font_body_bold.render("TRAFFIC", True, TEXT_PRIMARY)
        self.screen.blit(gen_lbl, (80 - gen_lbl.get_width()//2, 540))
        
        # Draw queue tube
        queue_box = (250, 500, 80, 100)
        self.draw_rounded_rect(self.screen, queue_box, CARD_BORDER, radius=10)
        
        # Fill queue box proportional to queue depth (max capacity visual 3000)
        fill_ratio = min(1.0, self.env.queue_depth / 2000.0)
        if fill_ratio > 0:
            fill_h = int(fill_ratio * 96)
            self.draw_rounded_rect(self.screen, (252, 598 - fill_h, 76, fill_h), RED if self.env.queue_depth > 50 else BLUE, radius=8)
            
        q_lbl = self.font_body_bold.render("QUEUE", True, TEXT_PRIMARY)
        self.screen.blit(q_lbl, (290 - q_lbl.get_width()//2, 475))
        q_val = self.font_body.render(f"{int(self.env.queue_depth)} requests", True, RED if self.env.queue_depth > 0 else TEXT_MUTED)
        self.screen.blit(q_val, (290 - q_val.get_width()//2, 610))

        # --- RIGHT PANEL: KUBERNETES CLUSTER NODES ---
        cluster_rect = (580, 100, 600, 480)
        self.draw_rounded_rect(self.screen, cluster_rect, CARD_COLOR)
        pygame.draw.rect(self.screen, CARD_BORDER, cluster_rect, 1, 8)
        self.screen.blit(self.font_section.render("Kubernetes Cluster (Simulated Nodes)", True, TEXT_PRIMARY), (600, 115))
        
        # Render 3 Nodes
        # Node capacity: 10 pods each
        for node_idx in range(3):
            nx = 600 + node_idx * 185
            ny = 145
            nw = 175
            nh = 350
            
            # Highlight node if active
            # Node is active if pods occupy it. CPU limit is 4.0, cpu_per_pod is 0.25 -> max 16 pods per node theoretically but config says 10 packing limit
            # Let's count pods on this node
            total_pods = self.env.replicas + len(self.env.pending_pods)
            pods_on_node = max(0, min(10, total_pods - (node_idx * 10)))
            
            node_border_color = BLUE if pods_on_node > 0 else CARD_BORDER
            self.draw_rounded_rect(self.screen, (nx, ny, nw, nh), BG_COLOR, radius=6)
            pygame.draw.rect(self.screen, node_border_color, (nx, ny, nw, nh), 2, 6)
            
            # Node header
            node_title = self.font_body_bold.render(f"Sim-Node-{node_idx+1}", True, TEXT_PRIMARY if pods_on_node > 0 else TEXT_MUTED)
            self.screen.blit(node_title, (nx + 10, ny + 8))
            
            # Render pod slots (10 slots)
            for slot_idx in range(10):
                pod_x = nx + 15
                pod_y = ny + 35 + slot_idx * 30
                slot_id = node_idx * 10 + slot_idx
                
                # Check status of this slot
                if slot_id < self.env.replicas:
                    # Running Pod
                    pygame.draw.circle(self.screen, GREEN, (pod_x + 12, pod_y + 12), 10)
                    # Simple indicator inside circle
                    pygame.draw.circle(self.screen, CARD_COLOR, (pod_x + 12, pod_y + 12), 4)
                    
                    # Draw capacity utilization (e.g. CPU util)
                    util_bar_w = int(self.env.cpu_util * 80)
                    pygame.draw.rect(self.screen, GRAY, (pod_x + 35, pod_y + 8, 80, 8))
                    pygame.draw.rect(self.screen, GREEN, (pod_x + 35, pod_y + 8, util_bar_w, 8))
                    
                elif slot_id < self.env.replicas + len(self.env.pending_pods):
                    # Pending Pod (Warming up)
                    pending_idx = slot_id - self.env.replicas
                    timer = self.env.pending_pods[pending_idx]
                    # timer is remaining seconds (e.g., 30 to 180). Max clip is 180.
                    progress_ratio = max(0.0, min(1.0, 1.0 - (timer / 120.0))) # approx range
                    
                    # Pulse effect
                    pulse = int(10 + 2 * math_sin(pygame.time.get_ticks() / 150.0))
                    pygame.draw.circle(self.screen, YELLOW, (pod_x + 12, pod_y + 12), pulse, 2)
                    
                    # Progress bar
                    pygame.draw.rect(self.screen, GRAY, (pod_x + 35, pod_y + 8, 80, 8))
                    pygame.draw.rect(self.screen, YELLOW, (pod_x + 35, pod_y + 8, int(progress_ratio * 80), 8))
                    lbl_start = self.font_body.render(f"starting {int(timer)}s", True, TEXT_MUTED)
                    self.screen.blit(lbl_start, (pod_x + 35, pod_y + 16))
                else:
                    # Empty Slot
                    pygame.draw.circle(self.screen, GRAY, (pod_x + 12, pod_y + 12), 6, 1)
                    
        # Legend
        leg_x = 600
        leg_y = 515
        pygame.draw.circle(self.screen, GREEN, (leg_x + 10, leg_y + 8), 6)
        self.screen.blit(self.font_body.render("Active Pod (Ready)", True, TEXT_PRIMARY), (leg_x + 25, leg_y))
        
        pygame.draw.circle(self.screen, YELLOW, (leg_x + 180, leg_y + 8), 6)
        self.screen.blit(self.font_body.render("Pending Pod (Cold Start)", True, TEXT_PRIMARY), (leg_x + 195, leg_y))
        
        # --- BOTTOM RIGHT: SYSTEM METRICS & ACTIONS ---
        metrics_panel_rect = (580, 600, 600, 180)
        self.draw_rounded_rect(self.screen, metrics_panel_rect, CARD_COLOR)
        pygame.draw.rect(self.screen, CARD_BORDER, metrics_panel_rect, 1, 8)
        
        # Row 1 metrics
        # Step count
        self.screen.blit(self.font_body_bold.render("Episode Step:", True, TEXT_MUTED), (600, 620))
        self.screen.blit(self.font_section.render(f"{self.env.step_count} / 120", True, TEXT_PRIMARY), (720, 617))
        
        # Replicas
        self.screen.blit(self.font_body_bold.render("Active Replicas:", True, TEXT_MUTED), (600, 650))
        self.screen.blit(self.font_section.render(f"{self.env.replicas} Pods", True, GREEN if self.env.replicas > 2 else TEXT_PRIMARY), (720, 647))
        
        # P99 Latency
        self.screen.blit(self.font_body_bold.render("P99 Latency:", True, TEXT_MUTED), (840, 620))
        lat_color = RED if self.env.p99_latency > self.env.sla_target else GREEN
        self.screen.blit(self.font_section.render(f"{self.env.p99_latency:.1f} ms", True, lat_color), (960, 617))
        
        # Cost accumulator
        self.screen.blit(self.font_body_bold.render("SLA Violations:", True, TEXT_MUTED), (840, 650))
        self.screen.blit(self.font_section.render(f"{self.total_breaches} steps", True, RED if self.total_breaches > 0 else GREEN), (960, 647))

        # Accumulated Cost
        self.screen.blit(self.font_body_bold.render("Total cost:", True, TEXT_MUTED), (840, 680))
        self.screen.blit(self.font_section.render(f"${self.total_cost:.3f}", True, YELLOW), (960, 677))

        # Last action
        self.screen.blit(self.font_body_bold.render("Last scaling action:", True, TEXT_MUTED), (600, 680))
        self.screen.blit(self.font_section.render(self.last_action_desc, True, PURPLE if self.last_action_source == "rl" else (BLUE if self.last_action_source == "hpa" else YELLOW)), (600, 700))

        # Status Footer
        status_y = 745
        footer_msg = "SPACE: Pause/Play  |  R: Reset Simulation  |  Keys 1-3: Switch Mode"
        if self.mode == 'manual':
            footer_msg += "  |  UP/DOWN Arrow: Manual Scale +/- 1"
        lbl_foot = self.font_body.render(footer_msg, True, TEXT_MUTED)
        self.screen.blit(lbl_foot, (600, status_y))

        # Display simulation paused indicator
        if self.paused:
            pause_lbl = self.font_large.render("PAUSED", True, YELLOW)
            self.screen.blit(pause_lbl, (WINDOW_WIDTH // 2 - pause_lbl.get_width() // 2, WINDOW_HEIGHT // 2 - pause_lbl.get_height() // 2))

        # Update and draw particles
        self.update_particles()
        for p in self.particles:
            pygame.draw.circle(self.screen, BLUE, (int(p.x), int(p.y)), 3)
            
        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            # Event Loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.reset_sim()
                    elif event.key == pygame.K_1:
                        self.mode = 'rl'
                        print("Switched to RL Agent Mode.")
                    elif event.key == pygame.K_2:
                        self.mode = 'hpa'
                        print("Switched to HPA Baseline Mode.")
                    elif event.key == pygame.K_3:
                        self.mode = 'manual'
                        print("Switched to Manual Control Mode.")
                    elif event.key == pygame.K_UP and self.mode == 'manual':
                        self.step(manual_delta=1)
                    elif event.key == pygame.K_DOWN and self.mode == 'manual':
                        self.step(manual_delta=-1)
                    elif event.key == pygame.K_n and self.paused:
                        # step manual clock
                        self.step()

            # Automatic simulation ticks
            if not self.paused:
                now = pygame.time.get_ticks()
                if now - self.last_step_time > self.step_delay_ms:
                    self.step()
                    self.last_step_time = now

            self.render()
            self.clock.tick(60)

        pygame.quit()


# Helper Math wrapper to bypass raw math call issues
def math_sin(val: float) -> float:
    import math
    return math.sin(val)

def math_cos(val: float) -> float:
    import math
    return math.cos(val)


if __name__ == "__main__":
    sim = VisualSimulator()
    sim.run()
