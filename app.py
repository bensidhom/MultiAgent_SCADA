import streamlit as st
import os
import time
import json
import threading
import logging
import io
import re
import csv
import socket
import struct
import math
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
import keras
import requests
import numpy as np
import pandas as pd
import paramiko
import cv2
import torch
from streamlit_autorefresh import st_autorefresh
# Matplotlib: force non-GUI backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image, ImageFile
from octorest import OctoRest

ImageFile.LOAD_TRUNCATED_IMAGES = True
#######################################
# streamlit_app.py
import re
import time
import ast
import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from PIL import Image
import os
import sqlite3
import pandas as pd
import sqlglot
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.config import RunnableConfig
from PIL import Image
import io
import base64
from pathlib import Path
from smolagents import LiteLLMModel, CodeAgent, tool
from octorest import OctoRest
import requests
import time
import json
# smolagents
DB_PATH = r".\data.db"

##############################
# =================== CONFIGURATION ===================
UPLOAD_FOLDER = r'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CONFIG_PATH = r"session_config.json"
API_KEY = "0B280554DA16426CB85536D88A82B672"

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler("app.log", maxBytes=5_000_000, backupCount=3)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
logger.handlers = [file_handler]

# =================== SHARED SESSION MANAGEMENT ===================
def get_saved_ip():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f).get("ip", "150.250.211.169")
    return "150.250.211.169"

def save_session_config(ip, job_name=None, timestamp=None, job_folder=None):
    config = {"ip": ip}
    if job_name and timestamp and job_folder:
        config.update({
            "job_name": job_name,
            "timestamp": timestamp,
            "job_folder": job_folder
        })
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)

def load_session_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"ip": get_saved_ip()}

# =================== OCTOPRINT CLIENT ===================
def make_octoprint_client(url, apikey):
    try:
        return OctoRest(url=url, apikey=apikey)
    except Exception as ex:
        logger.error(f"OctoPrint connection failed: {ex}")
        return None

def send_octoprint_command(command, action=None):
    session_config = load_session_config()
    ip = session_config["ip"]
    try:
        payload = {"command": command}
        if action:
            payload["action"] = action
        response = requests.post(
            f"http://{ip}/api/job",
            headers={"Content-Type": "application/json", "X-Api-Key": API_KEY},
            json=payload,
            timeout=10
        )
        logger.info(f"Command '{command}' sent. Status: {response.status_code}")
        return response.status_code == 204
    except Exception as e:
        logger.error(f"Error sending OctoPrint command: {e}")
        return False

def get_job_status():
    session_config = load_session_config()
    ip = session_config["ip"]
    try:
        response = requests.get(
            f"http://{ip}/api/job",
            headers={"X-Api-Key": API_KEY},
            timeout=5
        )
        return response.json()
    except Exception as e:
        logger.error(f"Error retrieving job status: {e}")
        return {}

def get_octoprint_files():
    session_config = load_session_config()
    ip = session_config["ip"]
    client = make_octoprint_client(f"http://{ip}", API_KEY)
    try:
        if client:
            files = client.files()['files']
            return [file['name'] for file in files]
    except Exception as e:
        logger.error(f"Error retrieving OctoPrint files: {e}")
    return []

def get_printer_temperature():
    """Get current printer temperature directly from API"""
    session_config = load_session_config()
    ip = session_config["ip"]
    try:
        response = requests.get(
            f"http://{ip}/api/printer",
            headers={"X-Api-Key": API_KEY},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            temp_data = data.get("temperature", {})
            return {
                "bed_actual": temp_data.get("bed", {}).get("actual", 0),
                "bed_target": temp_data.get("bed", {}).get("target", 0),
                "tool_actual": temp_data.get("tool0", {}).get("actual", 0),
                "tool_target": temp_data.get("tool0", {}).get("target", 0)
            }
    except Exception as e:
        logger.error(f"Error getting temperature: {e}")
    return {"bed_actual": 0, "bed_target": 0, "tool_actual": 0, "tool_target": 0}

# =================== REAL-TIME DATA STREAMING ===================
class RealTimeDataManager:
    def __init__(self):
        self.ae_waveform_data = []
        self.ae_classification_data = []
        self.cv_detection_data = []
        self.temperature_data = []
        self.printer_coordinates = []
        self.lock = threading.Lock()
        
        # Initialize plot data structures
        self.max_data_points = 100  # Keep last 100 points for performance
        
    def add_ae_data(self, waveform, classification, probability, timestamp):
        """Add AE data and track elapsed time"""
        if not hasattr(self, 'plot_start_time'):
            self.plot_start_time = time.time()
        elapsed_sec = time.time() - self.plot_start_time
        """Add AE data directly to memory"""
        with self.lock:
            self.ae_waveform_data.append({
                'timestamp': timestamp,
                'waveform': waveform,
                'classification': classification,
                'probability': probability,
                'elapsed_sec': elapsed_sec
            })
            # Keep only recent data
 #           if len(self.ae_waveform_data) > self.max_data_points:
   #             self.ae_waveform_data.pop(0)
    
    def add_cv_data(self, image, detections, timestamp):
        """Add CV detection data directly to memory"""
        with self.lock:
            self.cv_detection_data.append({
                'timestamp': timestamp,
                'image': image,
                'detections': detections
            })
            if len(self.cv_detection_data) > 10:  # Keep fewer images
                self.cv_detection_data.pop(0)
    
    def add_temperature_data(self, bed_temp, tool_temp, timestamp):
        """Add temperature data directly to memory"""
        with self.lock:
            self.temperature_data.append({
                'timestamp': timestamp,
                'bed_temp': bed_temp,
                'tool_temp': tool_temp
            })

         #if len(self.temperature_data) > self.max_data_points:
          #      self.temperature_data.pop(0)

    def add_printer_coordinates(self, x, y, z, timestamp):
        """Add printer coordinates directly to memory"""
        with self.lock:
            self.printer_coordinates.append({
                'timestamp': timestamp,
                'x': x,
                'y': y,
                'z': z
            })
          #  if len(self.printer_coordinates) > self.max_data_points:
           #     self.printer_coordinates.pop(0)
    
    def get_latest_ae_data(self):
        """Get latest AE data for plotting"""
        with self.lock:
            return self.ae_waveform_data[-1] if self.ae_waveform_data else None
    
    def get_latest_cv_data(self):
        """Get latest CV data for display"""
        with self.lock:
            return self.cv_detection_data[-1] if self.cv_detection_data else None
    
    def get_temperature_history(self):
        """Get temperature history for plotting"""
        with self.lock:
            return self.temperature_data.copy()
    
    def get_coordinate_history(self):
        """Get coordinate history for 3D plotting"""
        with self.lock:
            return self.printer_coordinates.copy()
###########################################################

class DatabaseManager:
    def __init__(self, db_path=r".\data.db"):
        self.db_path = db_path

    def insert_computer_vision(self, job_id, class_name, probability, image_path, time_val=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        date_time = datetime.now().isoformat()
        cursor.execute("""
            INSERT OR REPLACE INTO computer_vision (date_time, job_id, time, class, probability, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            date_time, job_id, time_val or 0.0, class_name, probability, image_path
        ))
        conn.commit()
        conn.close()

    def insert_time_series(self, job_id, class_name, probability, waveform, sample, time_val=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        date_time = datetime.now().isoformat(timespec='microseconds')

        cursor.execute("""
            INSERT OR REPLACE INTO time_series (
                date_time, job_id, time, class, probability,
                amplitude, duration, energy, rms, rise_time, counts, wave
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date_time,
            job_id,
            time_val if time_val is not None else 0.0,
            class_name,
            float(probability),
            float(sample.get("amplitude", 0.0)),
            float(sample.get("duration", 0.0)),
            float(sample.get("energy", 0.0)),
            float(sample.get("rms", 0.0)),
            float(sample.get("rise_time", 0.0)),
            int(sample.get("counts", 0)),
            json.dumps(waveform)
        ))

        conn.commit()
        conn.close()

Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
db = DatabaseManager(db_path=DB_PATH)

####################################################
# =================== AE MONITORING SYSTEM ===================
class AEMonitor:
    def __init__(self, data_manager, db):
        self.session_config = load_session_config()
        self.data_manager = data_manager
        self.model_path = r"models\class_bi.h5"
        self.model = None
        self.last_z = 0.0
        self.defect_points = []
        self.consecutive_defects = 0
        self.db = db
        
        
        # Initialize real-time plots
        self.fig1, self.axs = plt.subplots(2, 1, figsize=(8, 6))
        self.fig2 = plt.figure(figsize=(8, 6))
        self.ax3d = self.fig2.add_subplot(111, projection='3d')
        
        self.desired_order = [
            'timestamp', 'trai', 'amplitude', 'duration', 'energy', 'rms',
            'rise_time', 'counts', 'samples', 'waveform', 'class', 'probability'
        ]
        
        self.load_model()
    
    def load_model(self):
        """Load the AE classification model."""
        try:
            
            self.model = keras.models.load_model(self.model_path, compile=False)
            logger.info("AE model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load AE model: {e}")
    
    def _predict_scalar(self, inputs):
        """Run model.predict and return a single float scalar."""
        if not self.model:
            return 0.0
        try:
            y = self.model.predict(inputs, verbose=0)
            if isinstance(y, (list, tuple)):
                y = y[0]
            y = np.asarray(y)
            if y.size == 0:
                return 0.0
            return float(y.squeeze())
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0
    
    def extract_xyz_from_gcode(self, gcode_line: str):
        """Parse X/Y/Z floats from a G-code line."""
        _num = r"-?\d+(?:\.\d+)?"
        x = y = None
        
        mx = re.search(rf"\bX({_num})\b", gcode_line)
        my = re.search(rf"\bY({_num})\b", gcode_line)
        mz = re.search(rf"\bZ({_num})\b", gcode_line)

        if mx:
            try: x = float(mx.group(1))
            except Exception: x = None
        if my:
            try: y = float(my.group(1))
            except Exception: y = None
        if mz:
            try: self.last_z = float(mz.group(1))
            except Exception: pass
        return x, y, self.last_z
    


    def create_ae_plot(self, sampling_rate=10e6):
        """Create real-time AE plot with waveform in ms and history in seconds since plotting started"""
        
        # Initialize plot start time
        if not hasattr(self, 'plot_start_time'):
            self.plot_start_time = time.time()
        
        latest_data = self.data_manager.get_latest_ae_data()
        if not latest_data:
            return self.fig1

        waveform = latest_data['waveform']
        classification = latest_data['classification']
        probability = latest_data['probability']

        self.axs[0].clear()
        self.axs[1].clear()

        # -------------------- Plot latest waveform in ms --------------------
        if waveform:
            num_samples = len(waveform)
            time_axis_ms = np.arange(num_samples) / sampling_rate * 1e3  # convert to ms
            color = 'red' if classification == 'defected' else 'green'
            self.axs[0].plot(time_axis_ms, waveform, color=color, linewidth=1)
            self.axs[0].set_title(f"AE Waveform (Confidence: {probability:.2f})")
            self.axs[0].set_xlabel("Time (ms)")
            self.axs[0].set_ylabel("Amplitude")
            self.axs[0].grid(True, alpha=0.3)

        # -------------------- Plot AE history as scatter in seconds --------------------
        all_data = self.data_manager.ae_waveform_data
        if all_data:
            times_sec = [d['elapsed_sec'] for d in all_data]
            amplitudes = [max(d['waveform']) if d['waveform'] else 0 for d in all_data]
            colors = ['red' if d['classification'] == 'defected' else 'green' for d in all_data]

            self.axs[1].scatter(times_sec, amplitudes, c=colors, alpha=0.7)
            self.axs[1].set_title("AE Max Amplitude vs Time")
            self.axs[1].set_xlabel("Time (s) since plot start")
            self.axs[1].set_ylabel("Max Amplitude")
            self.axs[1].grid(True, alpha=0.3)

        self.fig1.tight_layout()
        return self.fig1



    
    def create_3d_geometry_plot(self):
        """Create real-time 3D geometry plot from memory data"""
        coordinates = self.data_manager.get_coordinate_history()
        self.ax3d.clear()
        
        if coordinates:
            x_coords = [coord['x'] for coord in coordinates if coord['x'] is not None]
            y_coords = [coord['y'] for coord in coordinates if coord['y'] is not None]
            z_coords = [coord['z'] for coord in coordinates if coord['z'] is not None]
            
            if x_coords and y_coords and z_coords:
                self.ax3d.plot(x_coords, y_coords, z_coords, 'b-', linewidth=1, alpha=0.7)
                
                # Mark defect points
                if self.defect_points:
                    defect_x, defect_y, defect_z = zip(*self.defect_points)
                    self.ax3d.scatter(defect_x, defect_y, defect_z, c='red', s=50, 
                                    marker='o', label='Defects')
                
                self.ax3d.set_xlabel('X')
                self.ax3d.set_ylabel('Y')
                self.ax3d.set_zlabel('Z')
                self.ax3d.legend()
        
        self.fig2.tight_layout()
        return self.fig2
    
    def process_record(self, record: dict):
        """Process AE record and update data manager"""
        if 'data' not in record or not isinstance(record['data'], list):
            return

        waveform = record["data"]
        if not waveform:
            return

        timestamp = record.get("time", datetime.now().isoformat())
        job_id = getattr(st.session_state, "current_job_id", "unknown_job")
        
        # Prepare features for model
        def _f(key):
            try:
                return float(record.get(key, 0.0))
            except Exception:
                return 0.0

        sample = {
            'amplitude': _f('amplitude'),
            'duration': _f('duration'),
            'energy': _f('energy'),
            'rms': _f('rms'),
            'rise_time': _f('rise_time'),
            'counts': int(record.get('counts', 0))
        }

        # Model inference
        input_dict = {k: np.array([v], dtype=np.float32) for k, v in sample.items()}
        prediction = self._predict_scalar(input_dict)

        defect = round(prediction ) == 1
        classification = 'defected' if defect else 'non-defected'
        probability = float(prediction)



        # Update data manager
        self.data_manager.add_ae_data(waveform, classification, probability, timestamp)

                    # Insert into DB
        self.db.insert_time_series(
        job_id=job_id,
        class_name=classification,
        probability=probability,
        waveform=waveform,
        sample=sample
    )

        # Update defect tracking
        if defect:
            self.consecutive_defects += 1
            # Add to defect points if we have coordinates
            coordinates = self.data_manager.get_coordinate_history()
            if coordinates:
                last_coord = coordinates[-1]
                if last_coord['x'] is not None and last_coord['y'] is not None:
                    self.defect_points.append((last_coord['x'], last_coord['y'], last_coord['z']))
        else:
            self.consecutive_defects = 0

        # Auto-pause on multiple defects
        if self.consecutive_defects >= 5:
            try:
                logger.info("5 consecutive defects detected. Pausing printer.")
                send_octoprint_command("pause", action="pause")
                self.consecutive_defects = 0
            except Exception as e:
                logger.error(f"Pause failed: {e}")
    
    def tail_gcode_coordinates(self):
        """Monitor G-code and extract coordinates in real-time for Godot server and local use."""

        host = self.session_config["ip"]
        username = 'pi'
        password = '1234'
        remote_file = '/home/pi/.octoprint/logs/serial.log'

        # TCP server for Godot (send latest coordinates)
        TCP_IP = '127.0.0.2'
        TCP_PORT = 50003
        HEADER = "CRD"
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((TCP_IP, TCP_PORT))
        server_socket.listen(1)
        print(f"[GCODE SERVER] Listening for Godot on {TCP_IP}:{TCP_PORT}...")

        client_socket, addr = server_socket.accept()
        print(f"[GCODE SERVER] Connection from {addr}")

        backoff = 2
        last_z = 0
        last_coord = (0, 0, 0, 0, 1000)  # (X, Y, Z, E, F)
        ssh_client = None

        def extract_coordinates_and_speed(line):
            x_match = re.search(r'X(-?\d*\.\d+|-?\d+)', line)
            y_match = re.search(r'Y(-?\d*\.\d+|-?\d+)', line)
            z_match = re.search(r'Z(-?\d*\.\d+|-?\d+)', line)
            f_match = re.search(r'F(\d*\.\d+|\d+)', line)
            e_match = re.search(r'E(-?\d*\.\d+|-?\d+)', line)
            x = float(x_match.group(1)) if x_match else None
            y = float(y_match.group(1)) if y_match else None
            z = float(z_match.group(1)) if z_match else None
            f = float(f_match.group(1)) if f_match else None
            e = float(e_match.group(1)) if e_match else None
            return (x, y, z, f, e)

        def format_message(header, coord, speed, extrusion):
            x, y, z = coord
            message = bytearray(1024)
            header_bytes = header.encode('utf-8')[:3]
            message[0:3] = header_bytes.ljust(3, b'\x00')
            message[3] = 5
            coord_bytes = struct.pack('<fffff', x, y, z, speed, extrusion)
            message[4:24] = coord_bytes
            return message

        while True:
            ssh_client = paramiko.SSHClient()
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                ssh_client.connect(
                    hostname=host,
                    username=username,
                    password=password,
                    banner_timeout=60,
                    auth_timeout=30,
                    timeout=30
                )
                sftp = ssh_client.open_sftp()
                remote_fh = sftp.file(remote_file, 'r')
                remote_fh.seek(0, 2)
                print(f"[GCODE] Monitoring {remote_file} for new coordinates...")

                backoff = 2
                while True:
                    line = remote_fh.readline()
                    if not line:
                        time.sleep(0.1)
                        continue

                    if 'G1' in line:
                        coords = extract_coordinates_and_speed(line)
                        if any(v is not None for v in coords[:3]):
                            x = coords[0] or last_coord[0]
                            y = coords[1] or last_coord[1]
                            z = coords[2] or last_coord[2]
                            f = coords[3] or last_coord[4]
                            e = coords[4] or last_coord[3]

                            # Save locally
                            timestamp = datetime.now().isoformat()
                            self.data_manager.add_printer_coordinates(x, y, z, timestamp)

                            # Send to Godot
                            msg = format_message(HEADER, (x, y, z), f, e)
                            try:
                                client_socket.sendall(msg)
                            except Exception as e:
                                print(f"[GCODE SERVER] Client disconnected: {e}")
                                client_socket, addr = server_socket.accept()
                                print(f"[GCODE SERVER] Reconnected: {addr}")

                            last_coord = (x, y, z, e, f)

            except Exception as e:
                print(f"[GCODE ERROR] {e} - retrying in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            finally:
                try:
                    ssh_client.close()
                except Exception:
                    pass

    
    def stream_ae_from_pi(self):
        """Stream AE data directly from Raspberry Pi"""
        host = self.session_config["ip"]
        backoff = 2
        while True:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            channel = None
            try:
                ssh.connect(
                    host,
                    username="pi",
                    password="1234",
                    banner_timeout=60,
                    auth_timeout=30,
                    timeout=30
                )
                transport = ssh.get_transport()
                channel = transport.open_session()
                channel.get_pty()
                channel.exec_command("python3 /home/pi/spotwave_ae.py")

                buffer = ""
                backoff = 2
                while not channel.exit_status_ready():
                    if channel.recv_ready():
                        chunk = channel.recv(4096).decode("utf-8", errors="ignore")
                        if not chunk:
                            time.sleep(0.1)
                            continue
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                                self.process_record(record)
                            except json.JSONDecodeError:
                                continue
                    else:
                        time.sleep(0.2)
            except Exception as e:
                logger.error(f"AE stream error: {e} - reconnecting in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
            finally:
                try:
                    if channel is not None:
                        channel.close()
                except Exception:
                    pass
                try:
                    ssh.close()
                except Exception:
                    pass
    
    def start_monitoring(self):
        """Start AE monitoring threads"""
        threads = [
            threading.Thread(target=self.tail_gcode_coordinates, daemon=True),
            threading.Thread(target=self.stream_ae_from_pi, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        logger.info("AE monitoring started")

# =================== COMPUTER VISION MONITORING ===================
class CVMonitor:
    def __init__(self, data_manager, db):
        self.session_config = load_session_config()
        self.data_manager = data_manager
        self.model = None
        self.load_model()
        self.db = db
    
    def load_model(self):
        """Load YOLOv5 model for defect detection"""
        repo_root = Path(__file__).resolve().parents[1]
        y5_dir = repo_root / "models" / "yolov5"
        weights = repo_root / "models" / "best.pt"
        
        if not (y5_dir / "hubconf.py").exists():
            logger.error(f"YOLOv5 repo not found at: {y5_dir}")
            return
        
        if not weights.exists():
            logger.error(f"Weights not found at: {weights}")
            return
        
        # Patch torch.load for compatibility
        orig_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return orig_load(*args, **kwargs)
        
        torch.load = patched_load
        try:
            self.model = torch.hub.load(
                str(y5_dir),
                model='custom',
                path=str(weights),
                source='local',
                force_reload=True
            )
            logger.info("YOLOv5 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLOv5 model: {e}")
        finally:
            torch.load = orig_load
    
    def get_webcam_image(self):
        """Get image directly from webcam stream"""
        session_config = load_session_config()
        try:
            cap = cv2.VideoCapture(f"http://{session_config['ip']}/webcam/?action=stream")
            if not cap.isOpened():
                return None
            
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
        except Exception as e:
            logger.error(f"Webcam error: {e}")
        
        return None
    
    def process_frame(self, frame):
        """Process frame with YOLO and return detections"""
        if self.model is None or frame is None:
            return frame, []
        
        # Run inference
        results = self.model(frame)
        detections = results.pandas().xyxy[0]  # x1, y1, x2, y2, confidence, class, name
        
        # Annotate frame
        annotated_frame = frame.copy()
        defect_detected = False
        job_id = getattr(st.session_state, "current_job_id", "unknown_job")
        
        # Draw bounding boxes and labels
        for _, detection in detections.iterrows():
            x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            confidence = float(detection['confidence'])
            name = detection['name']
            
            # Draw bounding box
            color = (255, 0, 0) if 'defect' in name.lower() else (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{name}: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Check if defect
            if name in ['spaghettification', 'underextrusion', 'overextrusion', 'stringing']:
                defect_detected = True

        # Save annotated frame once per processed frame
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S%f')
        job_folder = os.path.join("database", "dynamic", job_id)
        os.makedirs(job_folder, exist_ok=True)
        image_path = os.path.join(job_folder, f"frame_{timestamp_str}.png")
        cv2.imwrite(image_path, annotated_frame)

        # Insert each detection into DB, all pointing to the same saved image
        for _, detection in detections.iterrows():
            self.db.insert_computer_vision(
                job_id=job_id,
                class_name=detection['name'],
                probability=float(detection['confidence']),
                image_path=image_path
            )
        
        return annotated_frame, detections.to_dict('records'), defect_detected
    
    def monitor_defects(self):
        """Main CV monitoring loop"""
        defect_buffer = 0
        
        while True:
            # Get frame directly from webcam
            frame = self.get_webcam_image()
            if frame is not None:
                timestamp = datetime.now().isoformat()
                
                # Process frame
                processed_frame, detections, defect_detected = self.process_frame(frame)
                
                # Update data manager
                self.data_manager.add_cv_data(processed_frame, detections, timestamp)
                
                # Defect logic
                if defect_detected:
                    defect_buffer += 1
                    logger.info(f"Defect detected (buffer: {defect_buffer})")
                else:
                    defect_buffer = 0
                
                # Auto-pause on multiple defects
                if defect_buffer >= 3:
                    try:
                        logger.info("Multiple defects detected. Pausing printer.")
                        send_octoprint_command("pause", action="pause")
                        defect_buffer = 0
                    except Exception as e:
                        logger.error(f"Pause failed: {e}")
            
            time.sleep(1)  # Process every 2 seconds
    
    def start_monitoring(self):
        """Start CV monitoring in separate thread"""
        thread = threading.Thread(target=self.monitor_defects, daemon=True)
        thread.start()
        logger.info("CV monitoring started")

# =================== TEMPERATURE MONITORING ===================
class TemperatureMonitor:
    def __init__(self, data_manager):
        self.session_config = load_session_config()
        self.data_manager = data_manager
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.start_time = time.time()
    
    def create_temperature_plot(self):
        """Create real-time temperature plot from memory data"""
        temp_history = self.data_manager.get_temperature_history()
        self.ax.clear()
        
        if temp_history:
            timestamps = range(len(temp_history))
            bed_temps = [t['bed_temp'] for t in temp_history]
            tool_temps = [t['tool_temp'] for t in temp_history]
            
            self.ax.plot(timestamps, bed_temps, 'r-', label='Bed Temperature', linewidth=2)
            self.ax.plot(timestamps, tool_temps, 'b-', label='Tool Temperature', linewidth=2)
            
            self.ax.set_xlabel('Time (updates)')
            self.ax.set_ylabel('Temperature (°C)')
            self.ax.set_title('Real-time Temperature Monitoring')
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            
            # Dynamic Y-axis scaling
            all_temps = bed_temps + tool_temps
            if all_temps:
                max_temp = max(all_temps)
                self.ax.set_ylim(0, max(250, max_temp + 20))
        
        self.fig.tight_layout()
        return self.fig
    
    def monitor_temperature(self):
        """Monitor temperature directly from OctoPrint API"""
        while True:
            try:
                temp_data = get_printer_temperature()
                timestamp = datetime.now().isoformat()
                
                self.data_manager.add_temperature_data(
                    temp_data['bed_actual'],
                    temp_data['tool_actual'],
                    timestamp
                )
                
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Temperature monitoring error: {e}")
                time.sleep(5)
    
    def start_monitoring(self):
        """Start temperature monitoring in separate thread"""
        thread = threading.Thread(target=self.monitor_temperature, daemon=True)
        thread.start()
        logger.info("Temperature monitoring started")
################################################################################################




#############################################################################################
# =================== STREAMLIT UI ===================
def main():
    st.set_page_config(layout="wide", page_title="3D Printer Dashboard")
    
    # Initialize session state
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    if 'monitoring_started' not in st.session_state:
        st.session_state.monitoring_started = False
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = RealTimeDataManager()
    if 'ae_monitor' not in st.session_state:
        st.session_state.ae_monitor = None
    if 'cv_monitor' not in st.session_state:
        st.session_state.cv_monitor = None
    if 'temp_monitor' not in st.session_state:
        st.session_state.temp_monitor = None
    

    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager(db_path=r".\data.db")
    db = st.session_state.db
    st.title("3D Printer Monitoring & Control")
    
    # IP Input Section
    session_config = load_session_config()
    ip_input = st.text_input("Printer IP Address", session_config["ip"])
    API_KEY = "0B280554DA16426CB85536D88A82B672"
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Save IP"):
            save_session_config(ip_input)
            st.success("IP address saved! Please reload to apply changes.")
    
    # Initialize monitoring systems
    if not st.session_state.monitoring_started:
        if st.button("Start Monitoring Systems"):
            st.session_state.ae_monitor = AEMonitor(st.session_state.data_manager, db=db)
            st.session_state.cv_monitor = CVMonitor(st.session_state.data_manager, db=db)
            st.session_state.temp_monitor = TemperatureMonitor(st.session_state.data_manager)
            
            # Start all monitoring systems
            st.session_state.ae_monitor.start_monitoring()
            st.session_state.cv_monitor.start_monitoring()
            st.session_state.temp_monitor.start_monitoring()
            
            st.session_state.monitoring_started = True
            st.success("All monitoring systems started!")
    
    # Status display
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    job_status = get_job_status() or {}
    status = job_status.get("state", "Unknown")
    progress_data = job_status.get("progress") or {}
    completion = progress_data.get("completion", 0)
    completion = completion if isinstance(completion, (int, float)) else 0.0
    job_file_data = job_status.get("job") or {}
    current_file = job_file_data.get("file", {}).get("name", "---")
    
    # Get current temperature
    temp_data = get_printer_temperature()
    
    status_col1.metric("Status", status)
    status_col2.metric("Progress", f"{completion:.1f}%")
    status_col3.metric("Current File", current_file)
    status_col4.metric("Tool Temp", f"{temp_data['tool_actual']:.1f}°C")
    
    # File upload
    with st.expander("Upload G-code to OctoPrint"):
        uploaded_file = st.file_uploader("Choose a G-code file", type="gcode")
        if uploaded_file and st.button("Upload File"):
            try:
                local_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(local_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                logger.info(f"Uploading: {uploaded_file.name}")
                
                client = make_octoprint_client(f"http://{session_config['ip']}", API_KEY)
                if client:
                    client.upload(local_path, location="local", select=False)
                    st.success("File uploaded successfully!")
                else:
                    st.error("OctoPrint connection not available")
            except Exception as e:
                logger.error(f"Upload error: {e}")
                st.error(f"Upload failed: {str(e)}")
    
    # Print controls
    with st.expander("Print Controls"):
        octoprint_files = get_octoprint_files()
        selected_file = st.selectbox("Select a file to print", [""] + octoprint_files)
        job_name = st.text_input("Job Name", "Unnamed")
        
        col1, col2, col3, col4 = st.columns(4)

        db = DatabaseManager("data.db")
        
        if col1.button("⚙️ Start Print"):
            if selected_file:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                job_id = f"{job_name}_{timestamp}"
                job_folder = os.path.join(r"database\dynamic", job_id)
                os.makedirs(job_folder, exist_ok=True)
                
                save_session_config(session_config["ip"], job_name, timestamp, job_folder)
                st.session_state.current_job_id = job_id
                st.success(f"Job started: {job_id}")
                
                try:
                    client = make_octoprint_client(f"http://{session_config['ip']}", API_KEY)
                    if client:
                        client.select(selected_file)
                        time.sleep(1)
                        if get_job_status().get("state") == "Operational":
                            if send_octoprint_command("start"):
                                st.success("Print started successfully!")
                            else:
                                st.error("Failed to start print")
                        else:
                            st.warning("Printer is not operational")
                except Exception as e:
                    logger.error(f"Failed to start print: {e}")
                    st.error(f"Failed to start print: {str(e)}")
            else:
                st.warning("Please select a file to print")
        
        if col2.button("⏸️ Pause"):
            st.success("Print paused" if send_octoprint_command("pause", action="pause") else "Failed to pause print")
        
        if col3.button("▶️ Resume"):
            st.success("Print resumed" if send_octoprint_command("pause", action="resume") else "Failed to resume print")
        
        if col4.button("⛔ Cancel", type="primary"):
            st.success("Print cancelled" if send_octoprint_command("cancel") else "Failed to cancel print")
    
    # Real-time Monitoring Plots
    st.header("Real-time Monitoring")
    
    if st.session_state.monitoring_started:
        # 2x2 layout
        plot_col1, plot_col2 = st.columns(2)
        plot_col3, plot_col4 = st.columns(2)

        # --- AE Waveform & Classification ---
        with plot_col1:
            st.subheader("AE Waveform & Classification")
            @st.fragment(run_every="1s")
            def ae_plot_fragment():
                if st.session_state.ae_monitor:
                    fig = st.session_state.ae_monitor.create_ae_plot()
                    st.pyplot(fig)
            ae_plot_fragment()

        # --- 3D Printer Path ---
        with plot_col2:
            st.subheader("3D Printer Path")
            @st.fragment(run_every="1s")
            def geometry_plot_fragment():
                if st.session_state.ae_monitor:
                    fig = st.session_state.ae_monitor.create_3d_geometry_plot()
                    st.pyplot(fig)
            geometry_plot_fragment()

        # --- Temperature Monitoring ---
        with plot_col3:
            st.subheader("Temperature Monitoring")
            @st.fragment(run_every="1s")
            def temp_plot_fragment():
                if st.session_state.temp_monitor:
                    fig = st.session_state.temp_monitor.create_temperature_plot()
                    st.pyplot(fig)
            temp_plot_fragment()

        # --- Live Camera Feed ---
        with plot_col4:
            st.subheader("Live Camera Feed")
            @st.fragment(run_every="1s")
            def camera_fragment():
                cv_data = st.session_state.data_manager.get_latest_cv_data()
                if cv_data and cv_data['image'] is not None:
                    pil_image = Image.fromarray(cv_data['image'])
                    st.image(pil_image, caption="Live Camera with Defect Detection", use_container_width=True)
                    if cv_data['detections']:
                        st.write("Detections:")
                        for detection in cv_data['detections']:
                            st.write(f"- {detection.get('name', 'Unknown')}: {detection.get('confidence', 0):.2f}")
                else:
                    st.info("Waiting for camera feed...")
            camera_fragment()

        # Sleep to allow roughly 1-second refresh for fragments
        time.sleep(1)

    else:
        st.info("Click 'Start Monitoring Systems' to begin real-time monitoring")

    





    ######################################################################





    # -------------------------------
    # Configuration (use st.secrets in prod!)
    # -------------------------------
    DB_PATH = r".\data.db"
    db = DatabaseManager(db_path=DB_PATH)
    IMAGE_COLUMN_NAME = "image_path"
    WAVE_COLUMN_NAME = "wave"  # Optional: for waveform plotting

    # Use st.secrets or env vars in production
    API_KEY = "0B280554DA16426CB85536D88A82B672"
    OCTOPRINT_URL = "http://150.250.209.167"

    # LLM config
    LLM_API_BASE = "http://127.0.0.1:11434"
    MODEL_ID = "ollama_chat/qwen3-coder:30b"

    # -------------------------------
    # Database Connection & Schema
    # -------------------------------
    if not os.path.isfile(DB_PATH):
        st.error(f"Database file not found at: {DB_PATH}")
        st.stop()

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    @st.cache_data(ttl=3600, show_spinner=False)
    def load_table_schema() -> Dict[str, Dict[str, Any]]:
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            table_names = [row[0] for row in cursor.fetchall()]
            if not table_names:
                st.error("No tables found in the database.")
                return {}

            schema = {}
            for table in table_names:
                try:
                    # Use [table] to handle spaces/reserved words
                    df_sample = pd.read_sql(f"SELECT * FROM [{table}] LIMIT 1;", conn)
                    schema[table] = {
                        "columns": list(df_sample.columns),
                        "dtypes": df_sample.dtypes.to_dict()
                    }
                except Exception as e:
                    st.error(f"Error loading schema for table `{table}`: {e}")
            # st.success(f"Loaded tables: {', '.join(table_names)}")
            return schema
        except Exception as e:
            st.error(f"Error loading table schema: {e}")
            return {}

    table_schema = load_table_schema()
    if not table_schema:
        st.error("No table schema loaded.")
        st.stop()

    def generate_schema_description(schema: Dict[str, Dict[str, Any]]) -> str:
        desc = ""
        for table, info in schema.items():
            desc += f"Table '{table}':\n  Columns:\n"
            for col, dtype in info['dtypes'].items():
                desc += f"    - {col}: {dtype}\n"
        return desc

    tables_description = generate_schema_description(table_schema)

    # -------------------------------
    # Tools
    # -------------------------------

    @tool
    def sql_engine(query: str) -> pd.DataFrame:
        """
        Execute a SQL query on the SQLite database and return a pandas DataFrame.

        This is the **MANDATORY FIRST TOOL** for every user request.

        Args:
            query (str): Valid SQLite query using exact table/column names.

        Returns:
            pd.DataFrame: Query result or DataFrame with 'error' column on failure.
        """

        print(f"[sql_engine] Running: {query}")
        try:
            sqlglot.parse_one(query, dialect="sqlite")
        except Exception as e:
            return pd.DataFrame({"error": [f"Invalid SQL: {e}"]})

        try:
            df = pd.read_sql_query(query, conn)
            # Save the agent output for later use
            st.session_state["latest_df"] = df  # or use a file: df.to_pickle("latest_agent_output.pkl")
            st.session_state["latest_query"] = query

        except Exception as e:
            return pd.DataFrame({"error": [f"Execution error: {e}"]})

        if df.empty:
            return pd.DataFrame({"result": ["No data returned."]})

        df.columns = [col.strip() for col in df.columns]
        df = df.infer_objects()
        return df

    @tool
    def octoprint_controller(
        command: str,
        file_name: Optional[str] = None,
        target_temp: Optional[float] = None,
        flow_percent: Optional[float] = None,
        feed_percent: Optional[float] = None
    ) -> str:
        """
    Controls an OctoPrint 3D printer server.

    Supported commands:
    - "list_files": List all uploaded G-code files. Returns a string of file names.
    - "start": Start printing a G-code file. Requires `file_name`.
    - "pause": Pause the current print.
    - "resume": Resume a paused print.
    - "cancel": Cancel the current print.
    - "set_nozzle_temp": Set nozzle temperature. Optional `target_temp` (default 200°C).
    - "set_bed_temp": Set bed temperature. Optional `target_temp` (default 60°C).
    - "set_flow": Set extrusion flow percentage. Optional `flow_percent` (default 100%).
    - "set_feed": Set movement feed rate percentage. Optional `feed_percent` (default 100%).
    - "status": Get current printer and job status as JSON string.

    Args:
        command (str): The action to perform (see list above).
        file_name (str, optional): Name of the G-code file (required for "start").
        target_temp (float, optional): Target temperature in °C for nozzle or bed.
        flow_percent (float, optional): Flow rate percentage for extrusion.
        feed_percent (float, optional): Feed rate percentage for movement.

    Returns:
        str: Human-readable message describing the result or error.
    """
        try:
            client = OctoRest(url=OCTOPRINT_URL, apikey=API_KEY)
        except Exception as e:
            return f"Connection failed: {e}"

        headers = {"Content-Type": "application/json", "X-Api-Key": API_KEY}

        try:
            if command == "list_files":
                files = client.files().get("files", [])
                names = [f["name"] for f in files if "name" in f]
                return "Uploaded files:\n" + "\n".join(names) if names else "No files."

            elif command == "start" and file_name:
                files = client.files().get("files", [])
                if not any(f["name"] == file_name for f in files):
                    return f"File '{file_name}' not found."
                client.select(file_name, print=False)
                time.sleep(1)
                r = requests.post(f"{OCTOPRINT_URL}/api/job", headers=headers, json={"command": "start"})
                return f"Started printing {file_name}" if r.status_code == 204 else f"Failed: {r.text}"

            elif command == "pause":
                r = requests.post(f"{OCTOPRINT_URL}/api/job", headers=headers, json={"command": "pause", "action": "pause"})
                return "Paused." if r.status_code == 204 else f"Failed: {r.text}"

            elif command == "resume":
                r = requests.post(f"{OCTOPRINT_URL}/api/job", headers=headers, json={"command": "pause", "action": "resume"})
                return "Resumed." if r.status_code == 204 else f"Failed: {r.text}"

            elif command == "cancel":
                r = requests.post(f"{OCTOPRINT_URL}/api/job", headers=headers, json={"command": "cancel"})
                return "Canceled." if r.status_code == 204 else f"Failed: {r.text}"

            elif command == "set_nozzle_temp":
                temp = target_temp or 200
                r = requests.post(f"{OCTOPRINT_URL}/api/printer/tool", headers=headers,
                                json={"command": "target", "targets": {"tool0": temp}})
                return f"Nozzle → {temp}°C" if r.status_code == 204 else f"Failed: {r.text}"

            elif command == "set_bed_temp":
                temp = target_temp or 60
                r = requests.post(f"{OCTOPRINT_URL}/api/printer/bed", headers=headers,
                                json={"command": "target", "target": temp})
                return f"Bed → {temp}°C" if r.status_code == 204 else f"Failed: {r.text}"

            elif command == "set_flow":
                flow = flow_percent or 100
                r = requests.post(f"{OCTOPRINT_URL}/api/printer/command", headers=headers,
                                json={"commands": [f"M221 S{flow}"]})
                return f"Flow → {flow}%" if r.status_code == 204 else f"Failed: {r.text}"

            elif command == "set_feed":
                feed = feed_percent or 100
                r = requests.post(f"{OCTOPRINT_URL}/api/printer/command", headers=headers,
                                json={"commands": [f"M220 S{feed}"]})
                return f"Feed → {feed}%" if r.status_code == 204 else f"Failed: {r.text}"

            elif command == "status":
                job = requests.get(f"{OCTOPRINT_URL}/api/job", headers=headers).json()
                printer = requests.get(f"{OCTOPRINT_URL}/api/printer", headers=headers).json()
                state = job.get("state", "Unknown")
                prog = job.get("progress", {})
                temps = printer.get("temperature", {})
                summary = (
                    f"**Status:** {state}\n"
                    f"**Progress:** {prog.get('completion', 0):.1f}% | "
                    f"{prog.get('printTimeLeft', 'N/A')}s left\n"
                    f"**Nozzle:** {temps.get('tool0', {}).get('actual', 'N/A')}°C / "
                    f"{temps.get('tool0', {}).get('target', 'N/A')}°C\n"
                    f"**Bed:** {temps.get('bed', {}).get('actual', 'N/A')}°C / "
                    f"{temps.get('bed', {}).get('target', 'N/A')}°C"
                )
                return summary

            else:
                return f"Unknown command: {command}"

        except Exception as e:
            return f"Error: {e}"

    # -------------------------------
    # Agents
    # -------------------------------
    model = LiteLLMModel(model_id=MODEL_ID, api_base=LLM_API_BASE, num_ctx=8192)

    database_agent = CodeAgent(
        tools=[sql_engine],
        model=model,
        max_steps=2,
        name="database_agent",
        description="Query print history, defects, images, and sensor data.",
        additional_authorized_imports=["pandas", "numpy", "PIL"]
    )

    printing_agent = CodeAgent(
        tools=[octoprint_controller],
        model=model,
        max_steps=3,
        name="printing_agent",
        description="Control printer: start, pause, set temps, list files, etc.",
        additional_authorized_imports=["time", "requests", "json"]
    )

    # Master agent routes based on intent
    master_agent = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[database_agent, printing_agent],
        name="master_agent",
        description="Route DB queries to database_agent, printer control to printing_agent."
    )

    # -------------------------------
    # Streamlit UI
    # -------------------------------
    st.set_page_config(page_title="3D Print Assistant", layout="wide")
    st.title("3D Printing Agent")

    # --- Schema & Sample Data Display ---
    st.subheader("Database")
    num_tables = len(table_schema)
    cols = st.columns(min(num_tables, 3))

    for idx, (table_name, info) in enumerate(table_schema.items()):
        with cols[idx % 3]:
            with st.expander(f"**{table_name}**", expanded=False):
                # Schema table
                schema_data = [
                    {"Column": col, "Type": str(dtype)}
                    for col, dtype in info['dtypes'].items()
                ]
                st.dataframe(
                    pd.DataFrame(schema_data),
                    width='stretch',
                    hide_index=True,
                    column_config={
                        "Column": st.column_config.TextColumn(width="medium"),
                        "Type": st.column_config.TextColumn(width="small"),
                    }
                )

                # Sample data
                try:
                    sample_df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 20;", conn)
                    st.markdown("**Sample Rows (Top 5)**")
                    st.dataframe(sample_df,  width='stretch', hide_index=False)
                except Exception as e:
                    st.warning(f"Could not load sample: {e}")

    # Main UI



    user_input = st.text_area(
        "Ask about prints, defects, or control the printer:",
        placeholder="e.g., 'Show failed prints last week' or 'Start printing bracket.gcode'",
        height=120
    )
    prompt=f"""You are the master agent in charge of two specialized agents:

    1. **database_agent**: Handles SQL queries to the database. Returns results as pandas DataFrames.
    2. **printing_agent**: Controls a 3D printer via OctoPrint. Can list files, start/pause/resume/cancel prints, set nozzle/bed temperature, adjust flow or feed, and get printer status.
    ---

    ### DATABASE SCHEMA
    {tables_description}

    ---
    Your job is to:
    - Analyze the user's request carefully.
    - Decide whether it requires a database query, a printer action, or both.
    - Execute the **database_agent first** if needed, then use the results to determine any printer actions.
    - Always provide a natural-language response to the user, explaining what you did or plan to do.

    **Rules & Guidelines:**
    1. For SQL queries, always use the `database_agent`. Validate the query against existing tables/columns.
    2. For printer commands, use the `printing_agent`. Only send commands if required.
    3. You can **combine both agents**: e.g., check sensor values from the database and adjust printer parameters accordingly.
    4. Always include reasoning internally: check conditions, thresholds, or calculations before issuing printer commands.
    5. Avoid making assumptions; rely only on database query results and known printer commands.
    6. Respond in a friendly, clear way, summarizing actions for the user.

    **Example Use Cases:**
    - "Check the last print’s temperature readings. If the nozzle temperature was above 220°C, pause the printer."  
    → Query the database for the last print, analyze temperatures, and decide whether to send a pause command.
    - "Get the amplitude of sensor X. If it exceeds 10, increase nozzle temperature by 5°C."  
    → Query the database, evaluate the condition, then issue a nozzle temperature adjustment if needed.
    - "List uploaded G-code files."  
    → Directly call the `printing_agent` to list files.

    **Output Requirements:**
    - First, decide which agent(s) to call.
    - Show any **calculations or conditions** in your reasoning before executing commands.
    - Then provide a **summary of actions in plain language** for the user.



    ### USER QUESTION
    {user_input}
    """
    if st.button("Run", type="primary"):
        st.session_state["latest_df"] = None
        st.session_state["latest_query"] = None
        if not user_input.strip():
            st.warning("Please enter a question or command.")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Let master agent generate the reasoning / summary
                    summary = master_agent.run(prompt)

                    # Display master agent summary
                    #st.subheader("Summary / Reasoning")
                    st.write(summary)

                    # Show the SQL query (if saved)
                    if "latest_query" in st.session_state:
                        st.subheader("Executed SQL Query")
                        st.code(st.session_state["latest_query"], language="sql")

                    # --- Display latest DataFrame from database_agent ---
                    df = st.session_state.get("latest_df")
                    if df is not None and not df.empty:
                        st.dataframe(df, use_container_width=True)

                        # Images
                        if IMAGE_COLUMN_NAME in df.columns:
                            for p in df[IMAGE_COLUMN_NAME].dropna().unique():
                                if os.path.exists(p):
                                    st.image(p, use_container_width=True)
                                else:
                                    st.warning(f"Image not found: {p}")

                        # Waves
                        if WAVE_COLUMN_NAME in df.columns:
                            for idx, wave_str in enumerate(df[WAVE_COLUMN_NAME].dropna()):
                                try:
                                    wave = ast.literal_eval(wave_str) if isinstance(wave_str, str) else wave_str
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(y=wave, mode="lines"))
                                    fig.update_layout(height=250, title=f"Waveform #{idx+1}")
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Invalid wave at row {idx}: {e}")
                    else:
                        st.info("No data returned.")

                except Exception as e:
                    st.error(f"Agent failed: {e}")
                    st.exception(e)




    




    ################################

if __name__ == "__main__":
    main()