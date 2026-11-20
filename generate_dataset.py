"""Generate simulated run-to-failure data for 8 rotating machines (seed 2026)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)

# machine_id, type, zone, rated rpm, fault type, hours-to-failure
MACHINES = [
    ("RM01", "Centrifugal Pump", "Water Plant", 1480, "Bearing Wear", 452),
    ("RM02", "Induction Motor", "Assembly Line", 1450, "Overheating", 487),
    ("RM03", "Cooling Fan", "HVAC Room", 960, "Imbalance", 415),
    ("RM04", "Air Compressor", "Utility Block", 2900, "Bearing Wear", 509),
    ("RM05", "Gearbox Drive", "Packing Unit", 1750, "Misalignment", 466),
    ("RM06", "Conveyor Motor", "Dispatch Bay", 1200, "Overheating", 431),
    ("RM07", "Boiler Feed Pump", "Boiler House", 2950, "Misalignment", 498),
    ("RM08", "Exhaust Blower", "Paint Shop", 1440, "Imbalance", 443),
]

# how strongly each fault shows up on each signal (multipliers)
SIGN = {
    "Bearing Wear": dict(vx=1.0, vy=1.0, vz=0.8, peak=1.4, kurt=1.5, brg=1.0, mot=0.6, cur=0.8),
    "Imbalance": dict(vx=1.4, vy=1.3, vz=0.5, peak=0.8, kurt=0.5, brg=0.6, mot=0.5, cur=0.7),
    "Misalignment": dict(vx=0.9, vy=0.9, vz=1.5, peak=0.9, kurt=0.8, brg=0.8, mot=0.7, cur=0.9),
    "Overheating": dict(vx=0.6, vy=0.6, vz=0.5, peak=0.6, kurt=0.4, brg=1.2, mot=1.6, cur=1.4),
}

COLS = [
    "timestamp", "machine_id", "machine_type", "zone", "fault_type", "rpm", "load_pct",
    "ambient_temp_c", "bearing_temp_c", "motor_temp_c", "vib_x_mms", "vib_y_mms", "vib_z_mms",
    "vib_peak_g", "vib_kurtosis", "motor_current_a", "power_kw", "operating_hours",
    "health_index", "rul_hours", "failure_within_48h", "maintenance_class"
]

def build():
    rows = []
    start = pd.Timestamp("2026-03-01 00:00:00")
    for mid, mtype, zone, rpm0, fault, L in MACHINES:
        s = SIGN[fault]
        for t in range(L):
            rul = L - t
            # degradation stays tiny for most of the life, then speeds up
            d = float(np.clip(1 - rul / L, 0, 1)) ** 2.2
            load = float(np.clip(rng.normal(68, 9) + 7 * np.sin(2 * np.pi * (t % 24) / 24), 30, 100))
            amb = 27 + 5 * np.sin(2 * np.pi * ((t % 24) - 8) / 24) + rng.normal(0, 0.7)
            rpm = rpm0 * (1 + rng.normal(0, 0.006)) - 30 * d
            brg = amb + 20 + 0.13 * load + 36 * d * s["brg"] + rng.normal(0, 0.9)
            mot = amb + 28 + 0.19 * load + 32 * d * s["mot"] + rng.normal(0, 1.1)
            vx = 1.0 + 0.011 * load + 7.2 * d * s["vx"] + abs(rng.normal(0, 0.15))
            vy = 0.9 + 0.010 * load + 6.6 * d * s["vy"] + abs(rng.normal(0, 0.15))
            vz = 0.7 + 0.008 * load + 5.4 * d * s["vz"] + abs(rng.normal(0, 0.12))
            peak = 0.3 + 2.4 * d * s["peak"] + abs(rng.normal(0, 0.06))
            kurt = 3.0 + 4.2 * (d ** 1.4) * s["kurt"] + rng.normal(0, 0.17)
            cur = 7.5 + 0.055 * load + 3.0 * d * s["cur"] + rng.normal(0, 0.17)
            power = 1.732 * 415 * cur * 0.85 / 1000  # 3-phase, 415 V, pf 0.85
            hi = float(np.clip(100 * (1 - d) + rng.normal(0, 1.6), 0, 100))
            cls = "Normal" if rul > 120 else ("Warning" if rul > 48 else "Critical")
            rows.append([
                (start + pd.Timedelta(hours=t)).strftime("%Y-%m-%d %H:%M:%S"),
                mid, mtype, zone, fault, round(rpm, 1), round(load, 1), round(amb, 1),
                round(brg, 1), round(mot, 1), round(vx, 2), round(vy, 2), round(vz, 2),
                round(peak, 2), round(kurt, 2), round(cur, 2), round(power, 2), t + 1,
                round(hi, 1), rul, int(rul <= 48), cls
            ])
    df = pd.DataFrame(rows, columns=COLS)
    # real sensors drop readings sometimes: blank out ~0.6% of a few sensor cells
    for col in ["bearing_temp_c", "vib_x_mms", "motor_current_a", "vib_kurtosis"]:
        idx = rng.choice(df.index, size=int(0.006 * len(df)), replace=False)
        df.loc[idx, col] = np.nan
    return df

if __name__ == "__main__":
    df = build()
    df.to_csv("predictive_maintenance_dataset.csv", index=False)
    print(df.shape)
    print(df["maintenance_class"].value_counts())
    print("missing:", df.isna().sum()[df.isna().sum() > 0].to_dict())
    print(df.describe().T[["min", "mean", "max", "std"]].round(2))
