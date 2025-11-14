# MultiAgent_SCADA

A multi-agent SCADA system built with **Streamlit**, **Flask**, **Python**, and **OctoPrint**, integrating **Acoustic Emission (AE)** data, **computer vision**, **3D printer telemetry**, and **machine-learning inference** for real-time monitoring of additive manufacturing processes.

This repository includes:
- A real-time SCADA dashboard (`inference_deployement/app3.py`)
- A Flask backend for printer control and inference APIs
- Integration with **OctoPrint**, **SpotWave AE sensors (via Raspberry Pi)**, and **OpenCV**
- ML pipeline for AE-based defect detection
- Utilities for database logging, plotting, and AE/CV streaming

---

## 📁 Repository Structure

```
MultiAgent_SCADA/
│
├── inference_deployement/
│   ├── app3.py                # Main Streamlit SCADA dashboard
│   ├── AEPlotter.py           # Real-time AE plots
│   ├── CVPlotter.py           # Live CV plots + video frames
│   ├── utils.py               # Shared helpers
│   ├── session_config.json    # Stores IPs + printer/device configs
│
├── inference_server/
│   ├── app.py                 # Flask backend (inference + printer control)
│   ├── octorest_manager.py    # OctoPrint REST API control
│   ├── ml_model.py            # Loads Keras AE model
│   ├── ae_stream.py           # Live AE data processing
│   ├── config.py              # Server settings
│
├── cv/
│   ├── all_cv.py              # Camera streaming + CV inference
│
├── ae/
│   ├── all_ae.py              # AE streaming from Raspberry Pi
│
├── printer/
│   ├── printer_log_reader.py  # Printer logs (coords, temps, extrusion)
│
├── database/
│   ├── db_manager.py          # SQL logging
│   ├── schema.sql             # DB schema
│
└── README.md
```

---

## 🚀 Features

### **Streamlit SCADA Dashboard**
- AE waveform & spectrogram  
- Real-time camera feed  
- Printer telemetry (extrusion, coordinates, temperature)  
- AE-based defect detection (with model auto-pause)  
- Full printer controls: Start / Pause / Resume / Cancel  
- 1-second live refresh without rerendering everything  

### **Flask Backend**
REST API for:
| Action | Route |
|--------|-------|
| Start print | `/start` |
| Pause | `/pause` |
| Cancel | `/cancel` |
| Status | `/status` |
| ML inference | `/infer` |
| AE data streaming | `/ae` |

### **AE System**
- Streams from Raspberry Pi  
- Saves each burst  
- Classifies in real-time  
- Sends decisions to dashboard  
- Optional auto-stop  

### **Computer Vision**
- OpenCV camera capture  
- Real-time CV plotting  
- Inference-ready pipeline  

### **OctoPrint**
- OctoRest API integration  
- Status / Temp / Job progress  
- Full remote printer control  

---

## 🛠️ Installation

### Clone the repository
```bash
git clone https://github.com/bensidhom/MultiAgent_SCADA.git
cd MultiAgent_SCADA
```

### Create virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate      # Windows
source myenv/bin/activate   # macOS/Linux
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the System

### 1. Start the **Flask API server**
```bash
cd inference_server
python app.py
```

### 2. Start the **Streamlit SCADA dashboard**
```bash
cd inference_deployement
streamlit run app3.py
```

### 3. (Optional) Start AE streaming
```bash
cd ae
python all_ae.py
```

### 4. (Optional) Start CV streaming
```bash
cd cv
python all_cv.py
```

---

## ⚙️ Configuration

Edit:
```
inference_deployement/session_config.json
```

Example:
```json
{
  "ip": "192.168.1.25",
  "octoprint_url": "http://192.168.1.25:5000",
  "octoprint_api_key": "YOUR_API_KEY"
}
```

Used by:
- Streamlit dashboard  
- AE streamer  
- CV streamer  
- Flask API  

---

## 🗄️ Database Logging

Run:
```bash
sqlite3 scada.db < database/schema.sql
```

Automatically logs:
- AE bursts  
- CV detections  
- Printer telemetry  
- Inference results  

---

## 📌 Status

Currently under active development:
- Multi-agent orchestrator (LangGraph)  
- Cloud logging  
- Unified message bus  
- Real-time 3D visualizations  

---

## 🤝 Contributing

Fork → Make changes → Pull request  
Issues welcome.

---

## 📜 License
MIT License.

