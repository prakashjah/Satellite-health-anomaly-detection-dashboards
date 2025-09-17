# main.py
import threading
import time
from collections import deque
from typing import Dict, Any, List
import random
import math

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

app = FastAPI(title="Satellite Health - Telemetry & Anomaly API")

# Serve frontend files placed in ./static
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# telemetry buffer (latest N points)
BUFFER_SIZE = 300
telemetry_buffer = deque(maxlen=BUFFER_SIZE)

# Simulation control
SIM_INTERVAL = 1.0  # seconds between telemetry points

# Anomaly rules (simple thresholds)
THRESHOLDS = {
    "battery_voltage": {"min": 3.2, "max": 4.3},
    "thermal_temp": {"min": -10.0, "max": 85.0},
    "comm_signal": {"min": -90.0, "max": -20.0},  # dBm
}

# ML model (IsolationForest) - will be trained on initial "normal" simulated data
ml_model = None
model_ready = False

# For failure probability heuristic
ANOMALY_WINDOW = 12  # last N points to consider (~12 sec if SIM_INTERVAL=1)
ANOMALY_COUNT_THRESHOLD = 4  # if >= this anomalies in window -> high risk


def generate_normal_point(t: int) -> Dict[str, Any]:
    """
    Generate a realistic normal telemetry sample with slight noise.
    """
    # battery slowly varies
    battery_base = 3.9 + 0.05 * math.sin(t / 300.0)
    battery = round(random.normalvariate(battery_base, 0.03), 3)

    # thermal: base around 30C with small variations
    thermal = round(random.normalvariate(30 + 3 * math.sin(t / 120.0), 1.2), 2)

    # comm signal: vary between -65 and -35 dBm
    comm = round(random.normalvariate(-50 + 5 * math.sin(t / 60.0), 4.0), 2)

    return {"t": int(time.time()), "battery_voltage": battery, "thermal_temp": thermal, "comm_signal": comm}


def inject_anomaly(sample: Dict[str, Any], t: int) -> Dict[str, Any]:
    # randomly decide to inject anomaly (low prob). Increased for demo.
    r = random.random()
    # moderate chance for demo; tweak for faster/slower injections
    if r < 0.15:
        typ = random.choice(["battery_drop", "overheat", "comm_loss", "spike"])
        if typ == "battery_drop":
            sample["battery_voltage"] = round(sample["battery_voltage"] - random.uniform(0.6, 1.3), 3)
        elif typ == "overheat":
            sample["thermal_temp"] = round(sample["thermal_temp"] + random.uniform(30, 60), 2)
        elif typ == "comm_loss":
            sample["comm_signal"] = round(sample["comm_signal"] - random.uniform(30, 60), 2)
        elif typ == "spike":
            sample["battery_voltage"] += random.uniform(0.5, 1.0)
            sample["thermal_temp"] += random.uniform(10, 20)
    # smaller chance of a strong multi-metric event
    if r < 0.05:
        sample["battery_voltage"] = round(sample["battery_voltage"] - random.uniform(1.4, 2.2), 3)
        sample["comm_signal"] = round(sample["comm_signal"] - random.uniform(40, 70), 2)
    return sample


def is_rule_anomaly(point: Dict[str, Any]) -> Dict[str, bool]:
    flags = {}
    flags["battery"] = not (THRESHOLDS["battery_voltage"]["min"] <= point["battery_voltage"] <= THRESHOLDS["battery_voltage"]["max"])
    flags["thermal"] = not (THRESHOLDS["thermal_temp"]["min"] <= point["thermal_temp"] <= THRESHOLDS["thermal_temp"]["max"])
    flags["comm"] = not (THRESHOLDS["comm_signal"]["min"] <= point["comm_signal"] <= THRESHOLDS["comm_signal"]["max"])
    return flags


def train_ml_model(initial_samples: List[Dict[str, Any]]):
    global ml_model, model_ready
    if len(initial_samples) < 50:
        model_ready = False
        return
    df = pd.DataFrame(initial_samples)[["battery_voltage", "thermal_temp", "comm_signal"]]
    # train IsolationForest for novelty detection
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(df.values)
    ml_model = model
    model_ready = True
    print("[MODEL] IsolationForest trained on", len(df), "samples")


def compute_failure_probability(recent_points: List[Dict[str, Any]]) -> float:
    """Simple heuristic: ratio of anomaly counts + ML anomaly contributions"""
    if not recent_points:
        return 0.0
    rule_anomalies = 0
    ml_anomaly_score = 0.0
    for p in recent_points:
        flags = p.get("rule_flags", {})
        if any(flags.values()):
            rule_anomalies += 1
        ml_score = p.get("ml_score", 0.0)
        ml_anomaly_score += max(0.0, ml_score)

    rule_frac = rule_anomalies / len(recent_points)
    ml_frac = (ml_anomaly_score / len(recent_points)) if len(recent_points) else 0.0

    prob = 0.6 * rule_frac + 0.4 * ml_frac
    if rule_anomalies >= ANOMALY_COUNT_THRESHOLD:
        prob = max(prob, 0.6 + 0.1 * (rule_anomalies - ANOMALY_COUNT_THRESHOLD))
    return round(min(1.0, prob), 3)


def recommended_action(prob: float, last_flags: Dict[str, bool]) -> str:
    if prob >= 0.75:
        return "Enter SAFE MODE immediately and perform load shedding."
    if prob >= 0.4:
        if last_flags.get("battery"):
            return "Prioritize battery diagnostics; reduce power consumption and non-critical loads."
        if last_flags.get("thermal"):
            return "Schedule thermal checks; reduce operational load or orient for cooling."
        if last_flags.get("comm"):
            return "Switch to redundant comm channel and retransmit queued telemetry."
        return "Investigate anomalies; consider temporary operational limits."
    return "No critical action required; continue monitoring."


def telemetry_worker():
    """Background thread: generate telemetry and run detection continuously."""
    global ml_model
    t_counter = 0
    initial_training_samples = []
    while True:
        t_counter += 1
        sample = generate_normal_point(t_counter)
        sample = inject_anomaly(sample, t_counter)

        flags = is_rule_anomaly(sample)
        sample["rule_flags"] = flags

        if model_ready and ml_model is not None:
            X = np.array([[sample["battery_voltage"], sample["thermal_temp"], sample["comm_signal"]]])
            score = -ml_model.decision_function(X)[0]  # invert so higher=more anomalous
            ml_score = 1 / (1 + math.exp(-6 * (score - 0.02)))
            sample["ml_score"] = float(round(ml_score, 4))
        else:
            sample["ml_score"] = 0.0

        telemetry_buffer.append(sample)

        # collect data for initial model training
        if len(initial_training_samples) < 200:
            initial_training_samples.append({"battery_voltage": sample["battery_voltage"], "thermal_temp": sample["thermal_temp"], "comm_signal": sample["comm_signal"]})
        elif not model_ready:
            try:
                train_ml_model(initial_training_samples)
            except Exception as e:
                print("Model training failed:", e)

        time.sleep(SIM_INTERVAL)


@app.on_event("startup")
def start_background_tasks():
    t = threading.Thread(target=telemetry_worker, daemon=True)
    t.start()
    print("Telemetry simulator started.")


@app.get("/api/latest")
def api_latest():
    if not telemetry_buffer:
        return JSONResponse({"ok": False, "msg": "No data yet"}, status_code=503)
    latest = telemetry_buffer[-1]
    recent = list(telemetry_buffer)[-ANOMALY_WINDOW:]
    failure_prob = compute_failure_probability(recent)
    action = recommended_action(failure_prob, latest.get("rule_flags", {}))
    return {
        "ok": True,
        "latest": latest,
        "recent_count": len(recent),
        "failure_probability": failure_prob,
        "recommended_action": action,
    }


@app.get("/api/history")
def api_history(n: int = 120):
    n = max(1, min(n, BUFFER_SIZE))
    data = list(telemetry_buffer)[-n:]
    return {"ok": True, "count": len(data), "data": data}


@app.post("/api/inject")
def api_inject(type: str = Query("battery_drop")):
    """
    Manual injection endpoint for demo: type in {battery_drop, overheat, comm_loss, spike}
    Appends a few anomalous samples to the buffer.
    """
    if not telemetry_buffer:
        return {"ok": False, "msg": "no data yet"}
    latest = telemetry_buffer[-1]
    added = 0
    for i in range(3):
        s = latest.copy()
        s["t"] = int(time.time()) + i
        if type == "battery_drop":
            s["battery_voltage"] = round(max(0.3, s["battery_voltage"] - random.uniform(0.8, 1.6)), 3)
        elif type == "overheat":
            s["thermal_temp"] = round(s["thermal_temp"] + random.uniform(30, 60), 2)
        elif type == "comm_loss":
            s["comm_signal"] = round(s["comm_signal"] - random.uniform(30, 70), 2)
        elif type == "spike":
            s["battery_voltage"] = round(s["battery_voltage"] + random.uniform(0.6, 1.2), 3)
            s["thermal_temp"] = round(s["thermal_temp"] + random.uniform(10, 20), 2)
        s["rule_flags"] = is_rule_anomaly(s)
        s["ml_score"] = 1.0
        telemetry_buffer.append(s)
        added += 1
    return {"ok": True, "injected": type, "samples_added": added}
