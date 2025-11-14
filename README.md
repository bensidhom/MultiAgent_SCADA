# MultiAgent_SCADA

# 3D Printer Real-Time Monitoring Dashboard  
**Acoustic Emission (AE) • Computer Vision (CV) • Temperature • OctoPrint Control • AI Agents**

![GitHub](https://img.shields.io/badge/license-MIT-blue.svg)  
![Python](https://img.shields.io/badge/python-3.9%2B-blue)  
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-orange)

---

## Overview

This repository implements a real-time monitoring and control system for FDM 3D printers using OctoPrint. It integrates Acoustic Emission (AE), Computer Vision (CV), temperature monitoring, and full OctoPrint control into a unified Streamlit dashboard. A multi-agent AI system powered by Qwen3-Coder-30B enables natural-language interaction for querying data, analyzing defects, and automating printer control.

Repository URL: [https://github.com/bensidhom/MultiAgent_SCADA.git](https://github.com/bensidhom/MultiAgent_SCADA.git)

---

## Repository Structure

├── models/
│ ├── class_bi.h5
│ └── yolov5/
├── uploads/
├── database/
├── data.db
├── app.py
├── db.py
├── spotwave_ae.py
├── transfer_pi.py
├── server.crt / server.key
├── session_config.json
├── requirements.txt
└── README.md


*(If there are any additional folders or files in the repo, adjust this section to match exactly.)*

---

## Features

| Feature | Description |
|---------|-------------|
| AE Monitoring | 10 MHz sampling, feature extraction, Keras classification, auto-pause on 5 defects. |
| CV Monitoring | YOLOv5 detection for extrusion defects; auto-pause on 3+ detections. |
| Temperature Monitoring | Real-time nozzle/bed tracking with dynamic plots. |
| 3D Path Visualization | Parses G-code, streams coordinates to Godot, shows trajectory with defect markers. |
| OctoPrint Integration | Upload, start, pause, resume, cancel, set temps, adjust flow/feed. |
| AI Multi-Agent System | Natural-language control using smolagents + Qwen3-Coder-30B. |
| Database Logging | Stores AE waveforms, CV detections, temps, coordinates, timestamps, job IDs. |

---

## Multi-Agent Architecture

The system uses smolagents for code-executing agents:

### Database Agent
- SQL queries on `data.db`
- Returns DataFrames, tables, and plots

### Printing Agent
- Full OctoPrint control through OctoRest

### Master Agent
- Routes user queries
- Performs multi-step logic (e.g., detect → analyze → pause)

All agents run on Qwen3-Coder-30B (via Ollama or LiteLLM).

---

## Setup

### 1. Clone

```bash
git clone https://github.com/bensidhom/MultiAgent_SCADA.git
cd MultiAgent_SCADA

2. Create Environment

python -m venv myenv
source myenv/bin/activate         # Linux/macOS
myenv\Scripts\activate            # Windows


3. Install Dependencies

pip install -r requirements.txt
(Optional):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

4. SSL Certificates (Optional)

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout server.key -out server.crt

Database Initialization
python db.py

Creates:

time_series

date_time

job_id

time

class

probability

amplitude

duration

energy

rms

rise_time

counts

wave (JSON)

computer_vision

date_time

job_id

time

class

probability

image_path


Raspberry Pi (AE Acquisition)
1. Transfer Files

python transfer_pi.py --pi-ip <PI_IP> --username pi --password 1234

2. Install Dependencies
ssh pi@<PI_IP>
pip install numpy
3. Configure AE Parameters
SAMPLING_RATE = 10_000_000
THRESHOLD = 0.1
DURATION_US = 1000

streamlit run app.py


First-Time Setup (UI)

Enter OctoPrint IP

Ensure Pi is streaming AE

Ensure Qwen3-Coder-30B is running in Ollama

Click Start Monitoring Systems

Natural Language Examples

“Show AE defects from the last job.”

“Pause printing if CV finds 3 defects.”

“Start printing bracket.gcode at 215°C.”
