# Predictive Maintenance for Rotating Machinery Using IoT Data (P_214)

*Submitted By:* Varsha Yadav (Roll No: 2400320310200)  
*Institute:* ABES Engineering College, Ghaziabad  
*Domain:* Embedded IoT - Industrial Maintenance  

## Overview
End-to-end predictive maintenance pipeline using simulated sensor data (vibration, temperature, current, RPM, load) for 8 rotating machines. Data flows via *MQTT (Mosquitto)* into *InfluxDB, is analyzed by **XGBoost* models (health classification, 48-hour failure risk, and Remaining Useful Life), and is visualized on *Grafana* and an interactive web dashboard.

## How to Run
1. Start services: docker compose up -d
2. Generate dataset: python generate_dataset.py
3. Start subscriber: python subscriber.py
4. Start publisher: python publisher.py
5. Train models: python train.py
6. Run automated tests: python test_p214.py
7. 
