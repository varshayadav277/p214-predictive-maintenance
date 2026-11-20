"""MQTT -> InfluxDB. Checks every message; bad ones go to rejected_messages.log."""
import logging
import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from pipeline import validate_message, to_point

BROKER, PORT = "localhost", 1883
INFLUX_URL, TOKEN, ORG, BUCKET = "http://localhost:8086", "p214-token", "hcl", "machines"

logging.basicConfig(
    filename="rejected_messages.log",
    level=logging.WARNING,
    format="%(asctime)s %(message)s"
)

db = InfluxDBClient(url=INFLUX_URL, token=TOKEN, org=ORG)
writer = db.write_api(write_options=SYNCHRONOUS)

def on_connect(client, userdata, flags, rc, *extra):
    client.subscribe("plant/+/telemetry")  # runs again after every reconnect
    print("connected, listening on plant/+/telemetry")

def on_message(client, userdata, msg):
    ok, result = validate_message(msg.payload)
    if not ok:
        logging.warning("rejected %s: %s", msg.topic, result)
        return  # keep running
    p = to_point(result)
    point = Point(p["measurement"]).time(p["time"])
    for k, v in p["tags"].items():
        point = point.tag(k, v)
    for k, v in p["fields"].items():
        point = point.field(k, v)
    writer.write(bucket=BUCKET, record=point)

def main():
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()
    client.on_connect, client.on_message = on_connect, on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    client.connect(BROKER, PORT)
    client.loop_forever(retry_first_connection=True)

if __name__ == "__main__":
    main()
