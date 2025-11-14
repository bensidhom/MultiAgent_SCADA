# 3D Printing Monitoring System with Acoustic Emission and Computer Vision

This repository contains a complete pipeline for monitoring 3D printing processes using acoustic emission (AE) sensors, computer vision (CV), and integration with OctoPrint for printer control. The system runs on an AMD Ryzen AI setup, with data collection potentially offloaded to a Raspberry Pi for AE sensing. It uses SQLite for data storage, Streamlit for the user interface, and AI agents for intelligent querying and control.

The pipeline includes:
- Database creation and management for storing AE and CV data.
- AE data collection from a SpotWave device (running on Raspberry Pi).
- File transfer to Raspberry Pi via SSH.
- A Streamlit app that orchestrates real-time monitoring, data visualization, defect detection, and printer control using AI agents.

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

## Requirements
- **Hardware**:
  - AMD Ryzen AI system for running the main app.
  - Raspberry Pi (e.g., for running AE data collection).
  - SpotWave device for AE sensing (connected via USB to Raspberry Pi).
  - 3D Printer with OctoPrint server.
  - Optional: Camera for CV monitoring.

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

3. **Real-Time Monitoring**:
- The app initializes a `RealTimeDataManager` for in-memory storage of AE waveforms, classifications, CV detections, temperatures, and printer coordinates.
- An `AEMonitor` class loads the Keras model, processes AE data, classifies waveforms (e.g., "defect" if probability > 0.5), and detects defects.
- Defect detection: Pauses the print if 3+ consecutive defects are detected.
- Visualizations:
  - AE waveform and classification history plots.
  - 3D printer path with defect points (red markers).
  - Temperature trends (bed and tool).
  - CV images with detections.

4. **Database Interactions**:
- Data is inserted into `data.db` via `DatabaseManager`.
  - `computer_vision` table: Stores CV results (job_id, class, probability, image_path).
  - `time_series` table: Stores AE results (job_id, class, probability, waveform as JSON, features like amplitude, duration).
- Query the database using natural language via the AI agents in the UI.

5. **AI Agents Interface**:
- In the Streamlit UI, enter queries like:
  - "Show failed prints last week" (queries database).
  - "Start printing bracket.gcode" (controls printer).
  - "Check the last print’s temperature. If above 220°C, pause the printer." (combines query and control).
- The master agent routes to `database_agent` (SQL queries) or `printing_agent` (OctoPrint commands).
- Displays results as DataFrames, images, waveforms, or status messages.

6. **Printer Control**:
- Use the UI or agents for commands: list files, start/pause/resume/cancel prints, set temps, adjust flow/feed.
- Get real-time status and temperatures.

### File Explanations
- **db.py**: Creates the SQLite database (`data.db`) with two tables:
- `computer_vision`: Stores CV data (date_time, job_id, time, class, probability, image_path).
- `time_series`: Stores AE time-series data (date_time, job_id, time, class, probability, amplitude, duration, energy, rms, rise_time, counts, wave as JSON).
- Inserts example data for verification.
- Run this first to initialize the DB.

- **spotwave_ae.py**: AE data collector script, designed to run on Raspberry Pi.
- Uses the `waveline` library to connect to a SpotWave device via USB.
- Configures settings: threshold, filter (100kHz-450kHz), transient recording.
- Merges AE and transient records into `HitRecord` dataclasses.
- Outputs JSON-formatted records (including waveform as list) to stdout.
- Handles errors and prints status messages to stderr.

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