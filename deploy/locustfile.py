import math
import time
from locust import HttpUser, task, LoadTestShape

class PodinfoUser(HttpUser):
    @task
    def delay_request(self):
        # We hit an endpoint that simulates some CPU work or delay
        self.client.get("/delay/1")

class DiurnalShape(LoadTestShape):
    """
    Simulates a diurnal sine wave pattern matching the WorkloadGenerator's
    diurnal pattern: 125 + 75 * sin(2pi * t / 3600)
    """
    time_limit = 3600 # 1 hour
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time > self.time_limit:
            return None

        # 1 episode = 120 steps * 30s = 3600s
        # 125 baseline + 75 amplitude
        rate = 125.0 + 75.0 * math.sin(2.0 * math.pi * run_time / 3600.0)
        
        # In Locust, we control user count and spawn rate.
        # Assuming 1 user roughly = 1 RPS for simplicity
        user_count = int(rate)
        spawn_rate = 10
        
        return (user_count, spawn_rate)
