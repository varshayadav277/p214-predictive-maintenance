"""Build dashboard/index.html: a self-contained live-replay dashboard for all 8 machines.
Each machine is predicted by models that never saw that machine (leave-one-machine-out).
Results are cached per machine in dashboard/cache/, so the script can be stopped and re-run."""
import json
import os
import sys
from features import load_and_clean, add_features, feature_columns
from train import fit_models, predict

CACHE = "dashboard/cache"

def machine_data(df, cols, m):
    path = f"{CACHE}/{m}.json"
    if os.path.exists(path):
        return json.load(open(path))
    models = fit_models(df[df.machine_id != m], cols)  # leave this machine out
    g = df[df.machine_id == m]
    cls, prob, rul = predict(models, g)
    r = lambda s, n=2: [round(float(v), n) for v in s]
    d = {
        "type": g.machine_type.iloc[0], "zone": g.zone.iloc[0], "fault": g.fault_type.iloc[0],
        "start": str(g.timestamp.iloc[0]), "actual_rul": [int(v) for v in g.rul_hours],
        "vx": r(g.vib_x_mms), "vy": r(g.vib_y_mms), "vz": r(g.vib_z_mms),
        "brg": r(g.bearing_temp_c, 1), "mot": r(g.motor_temp_c, 1), "kurt": r(g.vib_kurtosis),
        "p_rul": r(rul, 1), "risk": r(prob, 3), "cls": [int(v) for v in cls]
    }
    json.dump(d, open(path, "w"), separators=(",", ":"))
    return d

def main(only=None):
    os.makedirs(CACHE, exist_ok=True)
    df = add_features(load_and_clean())
    cols = feature_columns(df)
    machines = sorted(df.machine_id.unique())
    out = {}
    for m in machines:
        if only and m not in only:
            continue
        out[m] = machine_data(df, cols, m)
        print(m, "done", flush=True)
    if only:
        return
    html = open("dashboard/template.html").read().replace(
        "/*DATA*/", "const DATA = " + json.dumps(out, separators=(",", ":")) + ";"
    )
    open("dashboard/index.html", "w").write(html)
    json.dump(out, open("dashboard/dashboard_data.json", "w"), separators=(",", ":"))
    print("wrote dashboard/index.html")

if __name__ == "__main__":
    main(sys.argv[1:] or None)
