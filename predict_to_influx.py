"""Read recent sensor data from InfluxDB, predict health, write predictions and alerts."""
import time
import joblib
import pandas as pd
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from features import add_features, SENSORS
from pipeline import alert_level, alert_message

INFLUX_URL, TOKEN, ORG, BUCKET = "http://localhost:8086", "p214-token", "hcl", "machines"
CLASSES = ["Normal", "Warning", "Critical"]
MODEL_VERSION = "v1"

FLUX = f'''from(bucket: "{BUCKET}") |> range(start: -72h)
  |> filter(fn: (r) => r._measurement == "machine_sensors")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'''

def run_once(db, writer, models):
    raw = db.query_api().query_data_frame(FLUX, org=ORG)
    raw = pd.concat(raw) if isinstance(raw, list) else raw
    if raw.empty:
        return
    raw = raw.rename(columns={"_time": "timestamp"}).sort_values(["machine_id", "timestamp"])
    raw[SENSORS] = raw.groupby("machine_id")[SENSORS].transform(lambda s: s.interpolate(limit_direction="both"))
    feats = add_features(raw)
    last = feats.groupby("machine_id").tail(1)  # latest hour of each machine
    X = last[models["features"]]
    cls = models["class"].predict(X)
    prob = models["fail48"].predict_proba(X)[:, 1]
    rul = models["rul"].predict(X).clip(0)
    for i, (_, row) in enumerate(last.iterrows()):
        level = alert_level(CLASSES[int(cls[i])], prob[i], rul[i])
        pt = (
            Point("machine_predictions").tag("machine_id", row.machine_id)
            .tag("model_version", MODEL_VERSION).time(row.timestamp)
            .field("health_class", CLASSES[int(cls[i])]).field("failure_prob_48h", float(prob[i]))
            .field("rul_hours", float(rul[i])).field("health_score", float(min(rul[i] / 300, 1) * 100))
            .field("status", level)
        )
        writer.write(bucket=BUCKET, record=pt)
        if level != "HEALTHY":
            alert = (
                Point("maintenance_alerts").tag("machine_id", row.machine_id).tag("severity", level)
                .time(row.timestamp).field("message", alert_message(row.machine_id, level, rul[i], prob[i]))
                .field("acknowledged", False)
            )
            writer.write(bucket=BUCKET, record=alert)

def main(every_seconds=5):
    models = joblib.load("models/p214_models.joblib")
    db = InfluxDBClient(url=INFLUX_URL, token=TOKEN, org=ORG)
    writer = db.write_api(write_options=SYNCHRONOUS)
    while True:
        run_once(db, writer, models)
        time.sleep(every_seconds)

if __name__ == "__main__":
    main()
