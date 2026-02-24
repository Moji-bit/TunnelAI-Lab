# sim/scenario_generator.py
from __future__ import annotations
import json
import os
import numpy as np
from dataclasses import asdict
from sim.event_generator import Scenario


def generate_random_scenarios(
    out_dir: str = "scenarios",
    n: int = 8,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    incident_types = ["collision", "stalled_vehicle", "wrong_way_driver", "vehicle_fire"]
    weather_types = ["clear", "rain", "fog", "snow"]

    scenarios = []

    for i in range(n):
        duration_s = int(rng.integers(5400, 9000))
        weather_start = int(rng.integers(600, max(1200, duration_s // 2)))
        weather_end = int(min(duration_s - 1, weather_start + int(rng.integers(600, 2200))))

        sc = Scenario(
            scenario_id=f"stau_case_{i:02d}",
            duration_s=duration_s,  # 1.5h – 2.5h
            q_in_base_veh_per_h=float(rng.uniform(1800, 2600)),
            q_in_peak_amp_veh_per_h=float(rng.uniform(800, 1500)),
            q_in_peak_period_s=int(rng.integers(1200, 2400)),
            incident_start_s=int(rng.integers(1500, 4000)),
            incident_end_s=int(rng.integers(4000, 6000)),
            incident_capacity_drop=float(rng.uniform(0.20, 0.60)),
            incident_type=str(rng.choice(incident_types)),
            incident_severity=float(rng.uniform(0.35, 0.95)),
            weather_type=str(rng.choice(weather_types)),
            weather_start_s=weather_start,
            weather_end_s=weather_end,
            weather_visibility_drop_pct=float(rng.uniform(0, 35)),
            weather_speed_drop_kmh=float(rng.uniform(0, 25)),
            vms_speed_limit_kmh=int(rng.choice([80, 90])),
            fan_stage=int(rng.choice([1, 2, 3])),
        )

        # ensure end > start
        if sc.incident_end_s <= sc.incident_start_s:
            sc.incident_end_s = sc.incident_start_s + 900

        path = os.path.join(out_dir, f"{sc.scenario_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(sc), f, indent=4)

        scenarios.append(path)

    print("✅ Generated scenarios:")
    for p in scenarios:
        print(" ", p)

    return scenarios


if __name__ == "__main__":
    generate_random_scenarios(n=8)