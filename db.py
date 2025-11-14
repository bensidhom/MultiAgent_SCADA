import sqlite3
import json
from pathlib import Path
 
def create_database(db_path="data.db"):
    # Ensure the directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
 
    # Connect to SQLite database (it will be created if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
 
    # Create the computer_vision table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS computer_vision (
        date_time TEXT PRIMARY KEY,
        job_id TEXT,
        time REAL,
        class TEXT,
        probability REAL,
        image_path TEXT
    )
    """)
 
    # Create the time_series table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS time_series (
        date_time TEXT PRIMARY KEY,
        job_id TEXT,
        time REAL,
        class TEXT,
        probability REAL,
        amplitude REAL,
        duration REAL,
        energy REAL,
        rms REAL,
        rise_time REAL,
        counts INTEGER,
        wave TEXT  -- Stored as JSON string of list[float]
    )
    """)
 
    conn.commit()
    conn.close()
    print(f"✅ SQLite database created successfully at: {db_path}")
 
 
def insert_example_data(db_path="data.db"):
    """Insert a small sample row in each table to verify the schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
 
    # Example insert for computer_vision
    cursor.execute("""
    INSERT OR REPLACE INTO computer_vision
    (date_time, job_id, time, class, probability, image_path)
    VALUES (?, ?, ?, ?, ?, ?)
    """, ("2025-10-27T12:00:00", "job_001", 1.23, "cat", 0.98, "C:\SCADA\MultiAgent_SCADA\circle_plot.png"))
 
    # Example insert for time_series
    wave_data = [0.1, 0.5, 0.3, 0.7]
    cursor.execute("""
    INSERT OR REPLACE INTO time_series
    (date_time, job_id, time, class, probability, amplitude, duration, energy, rms, rise_time, counts, wave)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "2025-10-27T12:01:00", "job_002", 2.5, "impact", 0.95,
        1.2, 0.8, 5.6, 0.45, 0.2, 100, json.dumps(wave_data)
    ))
 
    conn.commit()
    conn.close()
    print("✅ Example data inserted successfully.")
 
 
if __name__ == "__main__":
    create_database("data.db")
    insert_example_data("data.db")
 