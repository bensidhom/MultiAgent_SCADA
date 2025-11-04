import streamlit as st
from smolagents import LiteLLMModel, CodeAgent, tool
from octorest import OctoRest
import requests
import time
import json

# -------------------------------
# Streamlit session state
# -------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

col_input, col_output = st.columns([2, 3])

# -------------------------------
# OctoPrint Configuration
# -------------------------------
API_KEY = '0B280554DA16426CB85536D88A82B672'
OCTOPRINT_URL = "http://150.250.210.88"

# -------------------------------
# OctoPrint Controller Tool
# -------------------------------
@tool
def octoprint_controller(
    command: str,
    file_name: str | None = None,
    target_temp: float | None = None,
    flow_percent: float | None = None,
    feed_percent: float | None = None
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
        return f"❌ Connection failed: {e}"

    headers = {"Content-Type": "application/json", "X-Api-Key": API_KEY}

    try:
        if command == "list_files":
            files = client.files()
            file_list = [f["name"] for f in files.get("files", []) if "name" in f]
            return "📄 Uploaded G-code files:\n" + "\n".join(file_list)

        elif command == "start" and file_name:
            client.select(file_name)
            time.sleep(0.5)
            r = requests.post(f"{OCTOPRINT_URL}/api/job", headers=headers, json={"command": "start"})
            return f"▶️ Started printing {file_name}" if r.status_code == 204 else f"⚠️ Failed: {r.text}"

        elif command == "pause":
            r = requests.post(f"{OCTOPRINT_URL}/api/job", headers=headers, json={"command": "pause", "action": "pause"})
            return "⏸️ Print paused." if r.status_code == 204 else f"⚠️ Failed: {r.text}"

        elif command == "resume":
            r = requests.post(f"{OCTOPRINT_URL}/api/job", headers=headers, json={"command": "pause", "action": "resume"})
            return "▶️ Print resumed." if r.status_code == 204 else f"⚠️ Failed: {r.text}"

        elif command == "cancel":
            r = requests.post(f"{OCTOPRINT_URL}/api/job", headers=headers, json={"command": "cancel"})
            return "🛑 Print canceled." if r.status_code == 204 else f"⚠️ Failed: {r.text}"

        elif command == "set_nozzle_temp":
            target_temp = target_temp or 200
            r = requests.post(f"{OCTOPRINT_URL}/api/printer/tool", headers=headers, json={"command": "target", "targets": {"tool0": target_temp}})
            return f"🔥 Nozzle temp set to {target_temp}°C" if r.status_code == 204 else f"⚠️ Failed: {r.text}"

        elif command == "set_bed_temp":
            target_temp = target_temp or 60
            r = requests.post(f"{OCTOPRINT_URL}/api/printer/bed", headers=headers, json={"command": "target", "target": target_temp})
            return f"🛏️ Bed temp set to {target_temp}°C" if r.status_code == 204 else f"⚠️ Failed: {r.text}"

        elif command == "set_flow":
            flow_percent = flow_percent or 100
            r = requests.post(f"{OCTOPRINT_URL}/api/printer/command", headers=headers, json={"commands": [f"M221 S{flow_percent}"]})
            return f"💧 Flow rate set to {flow_percent}%" if r.status_code == 204 else f"⚠️ Failed: {r.text}"

        elif command == "set_feed":
            feed_percent = feed_percent or 100
            r = requests.post(f"{OCTOPRINT_URL}/api/printer/command", headers=headers, json={"commands": [f"M220 S{feed_percent}"]})
            return f"⚙️ Feed rate set to {feed_percent}%" if r.status_code == 204 else f"⚠️ Failed: {r.text}"

        elif command == "status":
            job = requests.get(f"{OCTOPRINT_URL}/api/job", headers=headers).json()
            printer = requests.get(f"{OCTOPRINT_URL}/api/printer", headers=headers).json()
            return json.dumps({"job": job, "printer": printer}, indent=2)

        else:
            return f"⚠️ Unknown command or missing parameter: {command}"

    except Exception as e:
        return f"❌ Command failed: {e}"


# -------------------------------
# Initialize NLP Agent
# -------------------------------
model = LiteLLMModel(
    model_id="ollama_chat/qwen3-coder:30b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

agent = CodeAgent(
    tools=[octoprint_controller],
    model=model,
    max_steps=5,
    additional_authorized_imports=["time", "requests", "json"]
)

# -------------------------------
# Streamlit NLP Input
# -------------------------------
with col_input:
    user_input = st.text_input("Ask the OctoPrint Agent:", placeholder="e.g., list files or start printing test.gcode")
    if user_input:
        prompt = f"""
You are a 3D printing assistant that only calls the `octoprint_controller` function to interact with OctoPrint.
The function supports: list_files, start, pause, resume, cancel, set_nozzle_temp, set_bed_temp, set_flow, set_feed, status.
### Output Format:
Thoughts: reasoning here
<code>
octoprint_controller("command", parameter=value)
</code>
Question: {user_input}
"""
        with st.spinner("🤖 Processing..."):
            output = agent.run(prompt)
            st.session_state.conversation.append({"user": user_input, "agent": output})

# -------------------------------
# Display Conversation
# -------------------------------
with col_output:
    st.subheader("💬 Interaction History")
    for msg in st.session_state.conversation:
        st.markdown(f"**You:** {msg['user']}")
        st.markdown(f"**Agent:** {msg['agent']}")
