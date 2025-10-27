from smolagents import LiteLLMModel, CodeAgent, tool
import pandas as pd
import sqlite3
import streamlit as st

# -------------------------------
# SQLite connection
# -------------------------------
DB_PATH = r"C:\all\github_projects\MultiAgent_SCADA\output_data.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# -------------------------------
# Universal tool for all tables
# -------------------------------
@tool
def sqlite_pandas_tool(question: str) -> str:
    """
    Allows the agent to query any table in the SQLite database using pandas.
    
    The agent has access to all tables as pandas DataFrames with their original table names.

    Args:
        question (str): A natural language question about the database.

    Returns:
        str: Result of the query or an error message.
    """
    try:
        # Load all tables as DataFrames dynamically
        table_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)["name"].tolist()
        local_vars = {name: pd.read_sql(f"SELECT * FROM {name};", conn) for name in table_names}

        # Provide pandas and a result placeholder
        local_vars.update({"pd": pd, "result": None})

        # Here we assume the agent will generate code to compute 'result'
        from smolagents import agent
        generated_code = agent.run(f"""
You have access to the following pandas DataFrames: {', '.join(table_names)}
User question: {question}
Write Python code using these DataFrames to answer the question.
Store the answer in a variable called 'result'.
""")

        # Execute the agent code safely
        exec(generated_code, {}, local_vars)
        return str(local_vars.get("result", "No result returned."))

    except Exception as e:
        return f"❌ Error: {e}"

# -------------------------------
# Initialize the agent
# -------------------------------
model = LiteLLMModel(
    model_id="ollama_chat/deepseek-coder:6.7b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

agent = CodeAgent(
    tools=[sqlite_pandas_tool],  # only one tool, all tables accessible
    model=model,
    max_steps=5,
)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="SQLite → Pandas NLP Agent", layout="wide")
st.title("🧠 SQLite → Pandas NLP Agent")

# Display tables
st.subheader("📘 Loaded Tables")
table_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)["name"].tolist()
for table_name in table_names:
    df = pd.read_sql(f"SELECT * FROM {table_name};", conn)
    with st.expander(f"Table: {table_name}", expanded=False):
        st.dataframe(df)

# User question input
st.subheader("💬 Ask a question about the database")
user_question = st.text_area(
    "Enter your question",
    placeholder="e.g. What is the average amplitude in sand_features?"
)

if st.button("Run"):
    if user_question.strip():
        with st.spinner("🤖 Processing your question..."):
            output = agent.run(user_question)
        st.success("✅ Finished")
        st.write("### Answer / Output")
        st.text(output)
    else:
        st.warning("Please enter a question first.")
