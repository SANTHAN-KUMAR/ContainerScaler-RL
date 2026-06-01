"""
Online Boutique Load Generator — Shaped User Journeys

Simulates realistic e-commerce shoppers hitting the Online Boutique frontend.
Each task weight is calibrated to the typical traffic distribution observed
in the original Google load-generator (which uses Locust under the hood):
  - 70 % browse/discover (catalogue, product pages)
  - 20 % cart interactions
  - 10 % checkout

Traffic shape: flash-crowd by default (sharp spike → plateau → cool-down),
switchable to diurnal for longer observation runs.
"""
from __future__ import annotations

import math
import random
from locust import HttpUser, between, task, LoadTestShape


# Product IDs shipped with the demo catalogue
_PRODUCT_IDS = [
    "OLJCESPC7Z",
    "66VCHSJNUP",
    "1YMWWN1N4O",
    "L9ECAV7KIM",
    "2ZYFJ3GM2N",
    "0PUK6V6EV0",
    "LS4PSXUNUM",
    "9SIQT8TOJO",
    "6E92ZMYYFZ",
]


class BoutiqueShopperUser(HttpUser):
    """Realistic online-boutique shopper that browses, adds to cart, and checks out."""

    # Think-time between requests: simulates real human reading/clicking
    wait_time = between(0.5, 2.0)

    # ── Browse & discover (weight 7) ──────────────────────────────────────────

    @task(5)
    def view_homepage(self):
        self.client.get("/", name="/")

    @task(2)
    def view_product(self):
        pid = random.choice(_PRODUCT_IDS)
        self.client.get(f"/product/{pid}", name="/product/[id]")

    # ── Cart interactions (weight 2) ──────────────────────────────────────────

    @task(2)
    def add_to_cart_and_view(self):
        pid = random.choice(_PRODUCT_IDS)
        self.client.post(
            "/cart",
            data={"product_id": pid, "quantity": random.randint(1, 3)},
            name="/cart [POST]",
        )
        self.client.get("/cart", name="/cart [GET]")

    # ── Checkout (weight 1) ───────────────────────────────────────────────────

    @task(1)
    def checkout(self):
        # Add an item first, then complete checkout
        pid = random.choice(_PRODUCT_IDS)
        self.client.post(
            "/cart",
            data={"product_id": pid, "quantity": 1},
            name="/cart [POST]",
        )
        self.client.post(
            "/cart/checkout",
            data={
                "email": "test@example.com",
                "street_address": "1600 Amphitheatre Pkwy",
                "zip_code": "94043",
                "city": "Mountain View",
                "state": "CA",
                "country": "United States",
                "credit_card_number": "4432801561520454",
                "credit_card_expiration_month": "1",
                "credit_card_expiration_year": "2030",
                "credit_card_cvv": "672",
            },
            name="/cart/checkout [POST]",
        )


# ── Traffic shapes ─────────────────────────────────────────────────────────────
import os
profile = os.environ.get("TRAFFIC_PROFILE", "steady")

if profile == "flash":
    class FlashCrowdShape(LoadTestShape):
        """
        Flash-crowd pattern — mirrors the 'flash_crowd' scenario in K8sSimEnv:
          Phase 1 (0 – 2 min): ramp from 20 → 200 users
          Phase 2 (2 – 8 min): sustain high load at 200 users
          Phase 3 (8 – 10 min): cool-down back to 20
        """
        stages = [
            {"duration": 120, "users": 20,  "spawn_rate": 10},   # ramp-up
            {"duration": 480, "users": 200, "spawn_rate": 30},   # flash crowd
            {"duration": 120, "users": 20,  "spawn_rate": 10},   # cool-down
        ]

        def tick(self):
            run_time = self.get_run_time()
            elapsed = 0
            for stage in self.stages:
                elapsed += stage["duration"]
                if run_time <= elapsed:
                    return stage["users"], stage["spawn_rate"]
            return None

elif profile == "diurnal":
    class DiurnalShape(LoadTestShape):
        """
        Diurnal sine-wave — 1-hour episode matching the WorkloadGenerator:
          rate = 125 + 75 * sin(2π * t / 3600)
        """
        time_limit = 3600

        def tick(self):
            run_time = self.get_run_time()
            if run_time > self.time_limit:
                return None
            rate = 125.0 + 75.0 * math.sin(2.0 * math.pi * run_time / 3600.0)
            return int(rate), 10

elif profile == "ddos":
    class DDOSShape(LoadTestShape):
        """
        Instant massive spike to 500 users to force a crash/throttle.
        """
        stages = [
            {"duration": 30, "users": 20, "spawn_rate": 10},
            {"duration": 600, "users": 500, "spawn_rate": 500},
        ]
        def tick(self):
            run_time = self.get_run_time()
            elapsed = 0
            for stage in self.stages:
                elapsed += stage["duration"]
                if run_time <= elapsed:
                    return stage["users"], stage["spawn_rate"]
            return None

elif profile == "step":
    class StepLoadShape(LoadTestShape):
        """
        Harsh step increments: 50 -> 100 -> 150 -> 200 users.
        """
        step_time = 60
        step_load = 50
        spawn_rate = 20
        time_limit = 600

        def tick(self):
            run_time = self.get_run_time()
            if run_time > self.time_limit:
                return None
            current_step = math.floor(run_time / self.step_time) + 1
            return int(current_step * self.step_load), self.spawn_rate

# If profile == "steady", no LoadTestShape is defined, so Locust falls back to 
# the constant --users and --spawn-rate CLI arguments, providing a steady baseline.
