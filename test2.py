# app.py
import streamlit as st
import matplotlib.pyplot as plt
import re
from smolagents import LiteLLMModel, CodeAgent

# -------------------------------
# Session state
# -------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# -------------------------------
# Initialize model and agent
# -------------------------------
model = LiteLLMModel(
    model_id="ollama_chat/deepseek-coder:6.7b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)




agent = CodeAgent(
    tools=[],
    model=model,
    additional_authorized_imports=["numpy", "matplotlib", "matplotlib.pyplot"],
    max_steps=3,

)

# -------------------------------
# Utility: extract code from output
# -------------------------------
def extract_code(agent_text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", agent_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match2 = re.search(r"<code>(.*?)</code>", agent_text, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    # fallback: assume lines starting with code-like syntax
    lines = agent_text.split("\n")
    code_lines = [l for l in lines if l.strip().startswith(("import", "plt", "np", "#"))]
    return "\n".join(code_lines)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("🧠 Defect Analysis Agent")

col_input, col_code, col_plot = st.columns([2, 2, 3])

# ---- Column 1: Input & Agent Output ----
with col_input:
    user_input = st.text_input("Ask the agent:", placeholder="e.g., plot a sine wave")

    if st.button("Run") and user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})

        context = "\n".join(
            [f"{m['role']}: {m['content']}" for m in st.session_state.conversation]
        )

        try:
            result = agent.run(user_input)
            agent_text = str(result)

            code_text = extract_code(agent_text)
            explanation_text = agent_text.replace(code_text, "").strip()

            st.session_state.conversation.append(
                {"role": "assistant", "content": explanation_text, "code": code_text}
            )

        except Exception as e:
            st.session_state.conversation.append(
                {"role": "assistant", "content": f"Agent error: {e}", "code": None}
            )

    st.subheader("💬 Interaction")
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Agent:** {msg['content']}")

# ---- Column 2: Code ----
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

# ---- Column 3: Plot ----
with col_plot:
    st.subheader("📊 Visualization")

    fig = None
    if st.session_state.conversation:
        last_msg = st.session_state.conversation[-1]
        code_text = last_msg.get("code", "")
        if code_text:
            try:
                local_vars = {}
                exec(code_text, {}, local_vars)
                fig = plt.gcf()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error executing code: {e}")
        else:
            st.write("No plot generated.")
    else:
        st.write("Plots will appear here after running code.")
