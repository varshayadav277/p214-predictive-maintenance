"""Message checking, InfluxDB point building and alert rules (no network needed)."""
import json

# sensible limits: anything outside is rejected as a broken reading
RANGES = {
    "rpm": (0, 4000),
    "load_pct": (0, 120),
    "ambient_temp_c": (-20, 70),
    "bearing_temp_c": (-20, 200),
    "motor_temp_c": (-20, 250),
    "vib_x_mms": (0, 60),
    "vib_y_mms": (0, 60),
    "vib_z_mms": (0, 60),
    "vib_peak_g": (0, 50),
    "vib_kurtosis": (0, 50),
    "motor_current_a": (0, 200),
    "power_kw": (0, 200),
}

REQUIRED = ["machine_id", "timestamp"] + list(RANGES)
MEASUREMENT = "machine_sensors"

def topic_for(machine_id):
    return f"plant/{machine_id}/telemetry"

def validate_message(raw):
    """Return (True, data) for a good message or (False, reason) for a bad one."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return False, "broken JSON"
    if not isinstance(data, dict):
        return False, "message is not a JSON object"
    for key in REQUIRED:
        if key not in data or data[key] is None:
            return False, f"missing field: {key}"
    for key, (lo, hi) in RANGES.items():
        try:
            value = float(data[key])
        except (ValueError, TypeError):
            return False, f"{key} is not a number"
        if not lo <= value <= hi:
            return False, f"{key}={value} is outside {lo}..{hi}"
    return True, data

def to_point(data):
    """Convert a checked message into an InfluxDB point description."""
    return {
        "measurement": MEASUREMENT,
        "tags": {"machine_id": data["machine_id"]},
        "fields": {k: float(data[k]) for k in RANGES},
        "time": data["timestamp"],
    }

def alert_level(health_class, failure_prob, rul_hours):
    """HEALTHY / WARNING / CRITICAL - the worst of the three signals wins."""
    if health_class == "Critical" or failure_prob >= 0.50 or rul_hours <= 48:
        return "CRITICAL"
    if health_class == "Warning" or failure_prob >= 0.25 or rul_hours <= 120:
        return "WARNING"
    return "HEALTHY"

def alert_message(machine_id, level, rul_hours, failure_prob):
    actions = {
        "HEALTHY": "No action needed",
        "WARNING": "Plan maintenance in the next shift or two",
        "CRITICAL": "Stop or service the machine as soon as possible"
    }
    return (
        f"{machine_id}: {level} - predicted RUL {rul_hours:.0f} h, "
        f"48 h failure risk {failure_prob*100:.0f}%. {actions[level]}."
    )
