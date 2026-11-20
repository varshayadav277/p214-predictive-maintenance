"""Replay the dataset like a live sensor: one hour of data per second, all machines."""
import json
import time
import pandas as pd
import paho.mqtt.client as mqtt
from pipeline import topic_for

BROKER, PORT = "localhost", 1883
SEND = [
    "machine_id", "timestamp", "rpm", "load_pct", "ambient_temp_c", "bearing_temp_c",
    "motor_temp_c", "vib_x_mms", "vib_y_mms", "vib_z_mms", "vib_peak_g",
    "vib_kurtosis", "motor_current_a", "power_kw"
]

def make_client():
    try:  # paho-mqtt 2.x
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:  # paho-mqtt 1.x
        return mqtt.Client()

def main(delay=1.0):
    df = pd.read_csv("predictive_maintenance_dataset.csv")
    df["hour"] = df.groupby("machine_id").cumcount()
    client = make_client()
    client.connect(BROKER, PORT)
    client.loop_start()
    for hour in range(int(df.hour.max()) + 1):
        for _, row in df[df.hour == hour].iterrows():
            msg = {k: (None if pd.isna(row[k]) else row[k]) for k in SEND}
            # a missing sensor value is sent as null; the subscriber will reject it and log it
            msg["timestamp"] = pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%dT%H:%M:%SZ")
            client.publish(topic_for(row["machine_id"]), json.dumps(msg, default=float))
        print(f"sent hour {hour}", end="\r")
        time.sleep(delay)
    client.loop_stop()

if __name__ == "__main__":
    main()
