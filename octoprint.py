import streamlit as st
from smolagents import LiteLLMModel, CodeAgent, tool
import re
from octorest import OctoRest
import sys
import time

# -------------------------------
# Configuration
# -------------------------------
API_KEY = '0B280554DA16426CB85536D88A82B672'
OCTOPRINT_URL = "http://150.250.220.113"

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
    Controls OctoPrint operations such as starting, pausing, resuming, or setting temperatures.

    Args:
        command (str): One of the following commands —
            ['start', 'pause', 'resume', 'cancel',
             'set_nozzle_temp', 'set_bed_temp',
             'set_flow', 'set_feed', 'status', 'list_files'].
        file_name (str, optional): The file name to print when using the 'start' command.
        target_temp (float, optional): Target temperature for nozzle or bed.
        flow_percent (float, optional): Flow rate percentage for extrusion.
        feed_percent (float, optional): Feed rate percentage for movement.

    Returns:
        str: Status message indicating the result of the command.
    """
    try:
        client = OctoRest(url=OCTOPRINT_URL, apikey=API_KEY)
    except Exception as e:
        return f"❌ Connection failed: {e}"

    try:
        if command == "list_files":
            files = client.files()
            if not files or "files" not in files:
                return "📂 No files found on OctoPrint."
            file_list = [f["name"] for f in files["files"] if "name" in f]
            return "📄 Uploaded G-code files:\n" + "\n".join(file_list)

        elif command == "start":
            if not file_name:
                return "❌ file_name required for start"
            client.select_file(file_name, print=True)
            return f"▶️ Started printing {file_name}"

        elif command == "pause":
            client.pause_print()
            return "⏸️ Print paused."

        elif command == "resume":
            client.resume_print()
            return "▶️ Print resumed."

        elif command == "cancel":
            client.cancel_print()
            return "🛑 Print canceled."

        elif command == "set_nozzle_temp":
            if target_temp is None:
                target_temp = 200
            client.tool_target(tool="tool0", target=target_temp)
            return f"🔥 Nozzle temp set to {target_temp}°C"

        elif command == "set_bed_temp":
            if target_temp is None:
                target_temp = 60
            client.bed_target(target=target_temp)
            return f"🛏️ Bed temp set to {target_temp}°C"

        elif command == "set_flow":
            if flow_percent is None:
                flow_percent = 100
            client.send_command({"command": "M221", "S": flow_percent})
            return f"💧 Flow rate set to {flow_percent}%"

        elif command == "set_feed":
            if feed_percent is None:
                feed_percent = 100
            client.send_command({"command": "M220", "S": feed_percent})
            return f"⚙️ Feed rate set to {feed_percent}%"

        elif command == "status":
            job = client.job_info()
            printer = client.printer()
            return str({"Job": job, "Printer": printer})

        else:
            return f"⚠️ Unknown command: {command}"

    except Exception as e:
        return f"❌ Command '{command}' failed: {e}"

# -------------------------------
# Initialize model and agent
# -------------------------------

model = LiteLLMModel(
    model_id="ollama_chat/deepseek-coder-v2:16b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

agent = CodeAgent(
    tools=[octoprint_controller],
    model=model,
    max_steps=5,
    additional_authorized_imports=["octorest", "sys", "time"]
)

# -------------------------------
# Utility: extract code from agent output
# -------------------------------
def extract_code(agent_text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", agent_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match2 = re.search(r"<code>(.*?)</code>", agent_text, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    return ""

# -------------------------------
# Streamlit Interface
# -------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

col_input, col_code, col_output = st.columns([2, 2, 3])

with col_input:
    user_input = st.text_input(
        "Ask the agent:",
        placeholder="e.g., list files OR start printing test.gcode OR set nozzle to 210"
    )

    if st.button("Run") and user_input:
        prompt = f"""
You are a 3D printing assistant that controls an OctoPrint server using the `octoprint_controller` function.

The function supports:
- 'list_files' (returns available uploaded G-code files)
- 'start' (requires file_name)
- 'pause'
- 'resume'
- 'cancel'
- 'set_nozzle_temp' (requires target_temp)
- 'set_bed_temp' (requires target_temp)
- 'set_flow' (requires flow_percent)
- 'set_feed' (requires feed_percent)
- 'status'

Mappings:
"list files" → list_files
"start print", "begin print" → start
"pause print" → pause
"resume print" → resume
"cancel print", "stop print" → cancel
"set nozzle to X" → set_nozzle_temp, target_temp=X
"set bed to X" → set_bed_temp, target_temp=X
"set flow to X" → set_flow, flow_percent=X
"set feed to X" → set_feed, feed_percent=X
"status" → status

### Output Format:
Thoughts: reasoning here
<code>
octoprint_controller("command", parameter=value)
</code>

Question:
{user_input}
"""

        # -------------------------------
        # Validation of agent output
        # -------------------------------
        def assert_agent_logic(code_text: str, user_input: str) -> bool:
            valid_commands = [
                "list_files", "start", "pause", "resume", "cancel",
                "set_nozzle_temp", "set_bed_temp",
                "set_flow", "set_feed", "status"
            ]

            if "octoprint_controller(" not in code_text and "st.error(" not in code_text:
                st.error("❌ Agent must call octoprint_controller or return an error.")
                return False

            command_match = re.search(r'octoprint_controller\("([^"]+)"', code_text)
            if command_match:
                command = command_match.group(1)
                if command not in valid_commands:
                    st.error(f"❌ Invalid command '{command}'. Must be one of {valid_commands}.")
                    return False

                if command == "start" and "file_name=" not in code_text:
                    st.error("❌ 'start' requires file_name parameter.")
                    return False
                if command in ["set_nozzle_temp", "set_bed_temp"] and "target_temp=" not in code_text:
                    st.error(f"❌ '{command}' requires target_temp parameter.")
                    return False
                if command == "set_flow" and "flow_percent=" not in code_text:
                    st.error("❌ 'set_flow' requires flow_percent parameter.")
                    return False
                if command == "set_feed" and "feed_percent=" not in code_text:
                    st.error("❌ 'set_feed' requires feed_percent parameter.")
                    return False

            return True

        # -------------------------------
        # Run agent and execute result
        # -------------------------------
        try:
            with st.spinner("🤖 Processing your request..."):
                output = agent.run(prompt)
                code_text = extract_code(output)
                explanation_text = output.replace(code_text, "").strip()

                if not assert_agent_logic(code_text, user_input):
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": f"Agent error: Invalid logic. {explanation_text}",
                        "code": code_text
                    })
                    st.stop()

                if code_text:
                    local_vars = {"st": st, "octoprint_controller": octoprint_controller}
                    exec(code_text, local_vars)

                st.session_state.conversation.append({"role": "user", "content": user_input})
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": explanation_text,
                    "code": code_text
                })

        except Exception as e:
            st.error(f"Agent error: {e}")
            st.session_state.conversation.append({
                "role": "assistant",
                "content": f"Agent error: {str(e)}",
                "code": ""
            })

# -------------------------------
# Conversation & Outputs
# -------------------------------
st.subheader("💬 Interaction History")
for msg in st.session_state.conversation:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Agent:** {msg['content']}")

with col_code:
    st.subheader("🧩 Generated Code")
    if st.session_state.conversation:
        last_msg = st.session_state.conversation[-1]
        code_text = last_msg.get("code", "")
        if code_text:
            st.code(code_text, language="python")
        else:
            st.write("No code generated yet.")
    else:
        st.write("Run a query to generate code.")

with col_output:
    st.subheader("📡 OctoPrint Output")
    st.write("Command outputs or error messages will appear here after running a command.")
