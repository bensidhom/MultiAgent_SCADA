# 3D Printing Monitoring System with Acoustic Emission and Computer Vision

This repository contains a complete pipeline for monitoring 3D printing processes using acoustic emission (AE) sensors, computer vision (CV), and integration with OctoPrint for printer control. The system runs on an AMD Ryzen AI setup, with data collection potentially offloaded to a Raspberry Pi for AE sensing. It uses SQLite for data storage, Streamlit for the user interface, and AI agents for intelligent querying and control.

The pipeline includes:
- Database creation and management for storing AE and CV data.
- AE data collection from a SpotWave device (running on Raspberry Pi).
- File transfer to Raspberry Pi via SSH.
- A Streamlit app that orchestrates real-time monitoring, data visualization, defect detection, and printer control using AI agents.
<img width="700" height="817" alt="image" src="https://github.com/user-attachments/assets/fec1612f-85d1-4b35-9ec7-fb311b2749cc" />
## Features
- **Acoustic Emission Monitoring**: Collects real-time AE waveforms from a SpotWave sensor, classifies them using a pre-trained Keras model, and detects defects based on thresholds.
- **Computer Vision Integration**: Processes images (e.g., from a camera) for defect detection and stores results in the database.
- **Real-Time Data Visualization**: Plots AE waveforms, classification history, temperature trends, printer coordinates in 3D, and defect points using Matplotlib and Plotly.
- **Database Management**: SQLite database with tables for computer vision and time-series AE data, including waveform storage as JSON.
- **OctoPrint Integration**: Controls 3D printers via OctoPrint API (e.g., start/pause prints, set temperatures, adjust flow/feed rates).
- **AI Agents**: Uses SmolAgents with LiteLLM for natural language querying of the database and printer control. Includes a master agent that routes requests to specialized agents.
- **Defect Detection and Response**: Automatically pauses printing on consecutive defects and visualizes defect points in 3D space.
- **File Transfer**: SSH-based transfer of scripts (e.g., AE collector) to Raspberry Pi.
- **Real-Time Streaming**: Manages in-memory data for AE, CV, temperatures, and printer coordinates with thread-safe access.
<img width="1215" height="531" alt="image" src="https://github.com/user-attachments/assets/8bcc8962-0c78-466d-8d5a-8b981faa1cb5" />


## Requirements
- **Hardware**:
  - AMD Ryzen AI system for running the main app.
  - Raspberry Pi (e.g., for running AE data collection).
  - SpotWave device for AE sensing (connected via USB to Raspberry Pi).
  - 3D Printer with OctoPrint server.
  - Optional: Camera for CV monitoring.
<img width="1066" height="550" alt="image" src="https://github.com/user-attachments/assets/91bd5fba-7455-40c4-9c16-5a001a1d0ad9" />

- **Software**:
  - Python 3.12+.
  - Libraries: Install via `pip install -r requirements.txt` (create this file based on imports; see below for key ones).
    - Key dependencies: `streamlit`, `paramiko`, `sqlite3`, `keras`, `tensorflow` (for model), `numpy`, `pandas`, `matplotlib`, `plotly`, `octorest`, `requests`, `opencv-python` (cv2), `torch`, `waveline`, `smolagents`, `langchain-community`, `sqlglot`, `pillow`.
  - Pre-trained Keras model: `models/class_bi.h5` (for AE classification; provide or train your own).
  - OctoPrint API key and URL (configured in code).

- **Environment Setup**:
  - Ensure Raspberry Pi is accessible via SSH (IP: 150.250.210.249, username: "pi", password: "1234" – update as needed).
  - SpotWave library must be installed on the Raspberry Pi for AE collection.

## Installation
1. Clone the repository:

git clone https://github.com/bensidhom/MultiAgent_SCADA.git

2. Install dependencies:
pip install -r requirements.txt
3. Prepare the database:
Run `db.py` to create the SQLite database
4. Transfer AE script to Raspberry Pi (if using Pi for AE collection):
Ensure `spotwave_ae.py` is in the same directory as `transfer_pi.py`
6. Configure OctoPrint:
- Set `API_KEY` in `app.py` to your OctoPrint API key.
- Update IP addresses (e.g., printer IP in session config).

## Usage
### Running the Main Application
The core of the system is `app.py`, a Streamlit app that handles monitoring, visualization, and control.

1. Start the Streamlit app:
streamlit run app.py

- Access the UI at `http://localhost:8501` (or the provided URL).

2. **Session Configuration**:
- Load or save printer IP (default: "150.250.211.169").
- Set job name, timestamp, and folder for organizing data.
![alt text](<Screenshot 2025-11-13 120945.png>)
control ​

Sends commands to OctoPrint via REST API. Handles start, pause, resume, cancel, and temperature/flow adjustments. Polls printer status and temperature every few seconds.
3. **Real-Time Monitoring**:
- The app initializes a `RealTimeDataManager` for in-memory storage of AE waveforms, classifications, CV detections, temperatures, and printer coordinates.
- An `AEMonitor` class loads the Keras model, processes AE data, classifies waveforms (e.g., "defect" if probability > 0.5), and detects defects.
- Defect detection: Pauses the print if 3+ consecutive defects are detected.
- Visualizations:
  - AE waveform and classification history plots.
  - 3D printer path with defect points (red markers).
  - Temperature trends (bed and tool).
  - CV images with detections.
![alt text](<Screenshot 2025-11-13 121108.png>)
Ai​

 Runs two models: Keras classifier on AE features to detect print defects, and YOLOv5 on webcam images to spot visual flaws like stringing or under-extrusion.
4. **Database Interactions**:
- Data is inserted into `data.db` via `DatabaseManager`.
  - `computer_vision` table: Stores CV results (job_id, class, probability, image_path).
  - `time_series` table: Stores AE results (job_id, class, probability, waveform as JSON, features like amplitude, duration).
- Query the database using natural language via the AI agents in the UI.
![alt text](<Screenshot 2025-11-13 121158.png>)
Manages real-time data:​

 SSH streams AE and G-code, HTTP grabs webcam frames, in-memory buffers hold recent data, and SQLite logs all events
5. **AI Agents Interface**:
- In the Streamlit UI, enter queries like:
  - "Show failed prints last week" (queries database).
  - "Start printing bracket.gcode" (controls printer).
  - "Check the last print’s temperature. If above 220°C, pause the printer." (combines query and control).
- The master agent routes to `database_agent` (SQL queries) or `printing_agent` (OctoPrint commands).
- Displays results as DataFrames, images, waveforms, or status messages.
![alt text](<Screenshot 2025-11-13 121242.png>)
Agents​

 LLM-powered agents (master, database, printing) interpret natural language, query SQLite logs, and control the printer based on sensor data and rules.
 <img width="1020" height="562" alt="image" src="https://github.com/user-attachments/assets/b5041408-6a33-469a-81b1-e46e152a4901" />

6. **Printer Control**:
- Use the UI or agents for commands: list files, start/pause/resume/cancel prints, set temps, adjust flow/feed.
- Get real-time status and temperatures.
![alt text](<Screenshot 2025-11-13 121447.png>)
GODOT​

 Receives live X/Y/Z coordinates via TCP (127.0.0.2:50003) and renders the 3D print path with defect markers in real time.
### File Explanations
- **db.py**: Creates the SQLite database (`data.db`) with two tables:
- `computer_vision`: Stores CV data (date_time, job_id, time, class, probability, image_path).
- `time_series`: Stores AE time-series data (date_time, job_id, time, class, probability, amplitude, duration, energy, rms, rise_time, counts, wave as JSON).
- Inserts example data for verification.
- Run this first to initialize the DB.
<img width="876" height="691" alt="image" src="https://github.com/user-attachments/assets/0016ff69-207a-4456-b561-091af6318ea8" />

- **spotwave_ae.py**: AE data collector script, designed to run on Raspberry Pi.
- Uses the `waveline` library to connect to a SpotWave device via USB.
- Configures settings: threshold, filter (100kHz-450kHz), transient recording.
- Merges AE and transient records into `HitRecord` dataclasses.
- Outputs JSON-formatted records (including waveform as list) to stdout.
- Handles errors and prints status messages to stderr.
<img width="892" height="558" alt="image" src="https://github.com/user-attachments/assets/0e0796b6-a803-45aa-9243-38d1d8d5f659" />

- **transfer_pi.py**: SSH file transfer script.
- Transfers `spotwave_ae.py` from local machine to Raspberry Pi (IP: 150.250.210.249).
- Uses Paramiko for SSH connection (username: "pi", password: "1234").
- Sets remote file permissions to executable (0o755).
- Run this to deploy the AE collector to the Pi.

- **app.py**: The main Streamlit application.
- Manages session config (IP, job details) in `session_config.json`.
- Integrates OctoPrint for printer control (start/pause, temps, status).
- Real-time data management with threading locks.
- AE monitoring: Loads Keras model, classifies waveforms, extracts features (amplitude, rms, etc.), inserts to DB.
- CV processing: Inserts detections to DB.
- Visualizations: Real-time plots for AE, temps, 3D paths.
- AI Agents: Uses SmolAgents for database querying (SQL via LangChain) and printer control (OctoPrint API).
- UI for natural language inputs, displaying results, schemas, and samples.


## Contributing
- Fork the repo and submit pull requests.
- Report issues for bugs or enhancements.
- Ensure code follows PEP8 and includes docstrings.

## License
MIT License – feel free to use and modify.
