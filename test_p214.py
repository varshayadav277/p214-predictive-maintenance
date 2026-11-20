"""24 automated tests (TC01-TC24) for the P_214 project. Run: python test_p214.py"""
import json
import unittest
import numpy as np
import pandas as pd
import generate_dataset as gd
import pipeline as pl
from features import (
    load_and_clean, add_features, feature_columns, make_labels,
    LEAKY, SENSORS
)
from train import fit_models, evaluate, predict, split, lead_time, TEST_MACHINES

RAW = RES = MODELS = TEST = DF = COLS = None

def setUpModule():
    global RAW, DF, COLS, MODELS, TEST, RES
    RAW = pd.read_csv("predictive_maintenance_dataset.csv")
    DF = add_features(load_and_clean())
    COLS = feature_columns(DF)
    train_df, TEST = split(DF)
    MODELS = fit_models(train_df, COLS)
    RES = evaluate(MODELS, TEST)

GOOD = {
    "machine_id": "RM05", "timestamp": "2026-03-01T00:00:00Z", "rpm": 1750, "load_pct": 66,
    "ambient_temp_c": 27, "bearing_temp_c": 50, "motor_temp_c": 62, "vib_x_mms": 2.0,
    "vib_y_mms": 1.8, "vib_z_mms": 1.4, "vib_peak_g": 0.4, "vib_kurtosis": 3.0,
    "motor_current_a": 11.0, "power_kw": 6.7
}

class TestDataset(unittest.TestCase):
    def test_tc01_file_size(self):
        self.assertEqual(RAW.shape[1], 22)
        self.assertEqual(RAW.machine_id.nunique(), 8)
        self.assertGreater(len(RAW), 3500)

    def test_tc02_healthy_phase(self):
        h = RAW[RAW.rul_hours > 300]
        self.assertLess(h.vib_x_mms.max(), 4.5)
        self.assertLess(h.bearing_temp_c.max(), 80)

    def test_tc03_vibration_near_failure(self):
        for _, g in RAW.groupby("machine_id"):
            self.assertGreater(g.vib_x_mms.tail(24).mean(), 2 * g.vib_x_mms.head(24).mean())

    def test_tc04_labels_match_rul(self):
        cls = np.where(RAW.rul_hours > 120, "Normal", np.where(RAW.rul_hours > 48, "Warning", "Critical"))
        self.assertTrue((cls == RAW.maintenance_class).all())
        self.assertTrue(((RAW.rul_hours <= 48).astype(int) == RAW.failure_within_48h).all())

    def test_tc05_same_seed_same_shape(self):
        self.assertEqual(gd.build().shape, RAW.shape)

    def test_tc06_missing_readings(self):
        self.assertGreater(RAW[SENSORS].isna().sum().sum(), 0)
        self.assertEqual(DF[SENSORS].isna().sum().sum(), 0)

class TestFeatures(unittest.TestCase):
    def test_tc07_rolling_average(self):
        g = DF[DF.machine_id == "RM01"].reset_index(drop=True)
        self.assertAlmostEqual(g.vib_x_mms_m6[50], g.vib_x_mms[45:51].mean(), places=9)

    def test_tc08_no_leakage(self):
        self.assertFalse(set(COLS) & set(LEAKY))

    def test_tc09_first_row_trend_zero(self):
        first = DF.groupby("machine_id").head(1)
        self.assertTrue((first.vib_x_mms_d12 == 0).all())

class TestMqttChecker(unittest.TestCase):
    def test_tc10_valid_message(self):
        ok, _ = pl.validate_message(json.dumps(GOOD))
        self.assertTrue(ok)

    def test_tc11_broken_json(self):
        ok, why = pl.validate_message("{not json")
        self.assertFalse(ok)
        self.assertIn("JSON", why)

    def test_tc12_missing_field(self):
        bad = {k: v for k, v in GOOD.items() if k != "rpm"}
        ok, why = pl.validate_message(json.dumps(bad))
        self.assertFalse(ok)
        self.assertIn("rpm", why)

    def test_tc13_impossible_value(self):
        ok, _ = pl.validate_message(json.dumps(dict(GOOD, bearing_temp_c=900)))
        self.assertFalse(ok)

    def test_tc14_influx_point(self):
        p = pl.to_point(GOOD)
        self.assertEqual(p["measurement"], "machine_sensors")
        self.assertEqual(p["tags"], {"machine_id": "RM05"})
        self.assertEqual(len(p["fields"]), 12)  # all 12 sensor fields of Section 5

    def test_tc15_topic_name(self):
        self.assertEqual(pl.topic_for("RM05"), "plant/RM05/telemetry")

class TestAlerts(unittest.TestCase):
    def test_tc16_healthy(self):
        self.assertEqual(pl.alert_level("Normal", 0.02, 300), "HEALTHY")

    def test_tc17_warning(self):
        self.assertEqual(pl.alert_level("Warning", 0.10, 100), "WARNING")

    def test_tc18_model_worry_wins(self):
        self.assertEqual(pl.alert_level("Normal", 0.80, 300), "CRITICAL")

class TestModel(unittest.TestCase):
    def test_tc19_f1(self):
        self.assertGreaterEqual(RES["f1"], 0.90)

    def test_tc20_critical_recall(self):
        self.assertGreaterEqual(RES["per_class"]["Critical"][1], 0.90)

    def test_tc21_rul_error(self):
        self.assertLess(RES["rul_mae"], 15)

    def test_tc22_risk_levels(self):
        _, prob, _ = predict(MODELS, TEST)
        self.assertLess(prob[(TEST.rul_hours > 300).values].mean(), 0.05)
        self.assertGreater(prob[(TEST.rul_hours <= 24).values].mean(), 0.90)

    def test_tc23_early_warning(self):
        for m in TEST_MACHINES:
            self.assertGreaterEqual(RES["lead"][m], 72)

    def test_tc24_test_machines_unseen(self):
        self.assertEqual(sorted(TEST.machine_id.unique()), TEST_MACHINES)

if __name__ == "__main__":
    unittest.main(verbosity=2)
