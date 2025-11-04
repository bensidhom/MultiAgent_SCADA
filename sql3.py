# app.py
import os
import sqlite3
import pandas as pd
import streamlit as st
from smolagents import LiteLLMModel, CodeAgent, tool
import re
import sqlglot
from langchain_community.utilities.sql_database import SQLDatabase
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Database Setup
# -------------------------------
DB_PATH = r".\output_data.db"

if not os.path.isfile(DB_PATH):
    st.error(f"Database not found: {DB_PATH}")
    st.stop()

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# -------------------------------
# Load Schema
# -------------------------------
@st.cache_data
def load_schema():
    tables = db.get_table_names()
    if not tables:
        st.error("No tables found.")
        return {}
    schema = {}
    for t in tables:
        df = pd.read_sql(f"SELECT * FROM {t} LIMIT 1;", conn)
        schema[t] = {"cols": list(df.columns), "dtypes": df.dtypes.to_dict()}
    st.success(f"Tables: {', '.join(tables)}")
    return schema

schema = load_schema()
if not schema:
    st.stop()

def schema_desc(s):
    d = ""
    for t, info in s.items():
        d += f"Table '{t}':\n"
        for c, typ in info['dtypes'].items():
            d += f"  - {c}: {typ}\n"
    return d

tables_desc = schema_desc(schema)

# -------------------------------
# SQL Tool
# -------------------------------
@tool
def sql_engine(query: str) -> str:
    """
    Executes a SQL query on the connected SQLite database and returns a preview of the result.

    This function:
    - Validates the SQL syntax using sqlglot (SQLite dialect)
    - Executes the query via pandas.read_sql
    - Stores the **full result DataFrame** in `st.session_state.last_df` for downstream analysis/plotting
    - Returns a **string preview** (first few rows) suitable for LLM feedback

    Args:
        query (str): 
            A valid SQLite SELECT query. 
            Example: "SELECT rms FROM sand_friction_features LIMIT 100"

    Returns:
        str: 
            - A formatted string of the first few rows if results exist
            - "Empty result." if the query returns no rows
            - Error message starting with "SQL Error:" on failure
            """
    try:
        sqlglot.parse_one(query, dialect="sqlite")
        df = pd.read_sql(query, conn)
        st.session_state.last_df = df
        return df.head().to_string(index=False) if not df.empty else "Empty"
    except Exception as e:
        return f"SQL Error: {e}"

# -------------------------------
# LLM + Agent
# -------------------------------
model = LiteLLMModel(
    model_id="ollama_chat/qwen3-coder:30b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

agent = CodeAgent(
    tools=[sql_engine],
    model=model,
    max_steps=3,
    additional_authorized_imports=[
        "pandas", "numpy", "matplotlib.pyplot as plt",
        "plotly.express as px", "scipy", "statsmodels.api as sm"
    ]
)

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="SQL + Python Agent", layout="wide")
st.title("SQLite → SQL + Python Agent")

# Schema preview
st.subheader("Database Schema")
cols = st.columns(min(len(schema), 3))
for i, t in enumerate(schema):
    with cols[i % 3]:
        with st.expander(t):
            st.dataframe(pd.DataFrame([
                {"Col": c, "Type": str(d)} for c, d in schema[t]['dtypes'].items()
            ]), use_container_width=True)
            preview = pd.read_sql(f"SELECT * FROM {t} LIMIT 5;", conn)
            st.dataframe(preview, use_container_width=True)

# Input
st.subheader("Ask a Question or Request a Plot")
user_q = st.text_area(
    "Your request",
    placeholder="e.g., plot the rms in sand friction features",
    value="plot the rms in sand friction features",
    height=100
)

if st.button("Clear Cache & Reload"):
    st.cache_data.clear()

    schema = load_schema()
    tables_desc = schema_desc(schema)
    st.success("Reloaded.")

# -------------------------------
# Run Agent
# -------------------------------
# -------------------------------------------------
# Run Agent
# -------------------------------------------------
if st.button("Run Analysis"):
    if not user_q.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Agent working..."):
            prompt = f"""
You have these tables:
{tables_desc}

Rules:
1. Run SQL with `sql_engine(query)` → result saved in `st.session_state.last_df`
2. **Always** use `df = st.session_state.last_df` for analysis/plotting
3. For **plots**, prefer `plotly.express as px` → `fig.show()`
4. For **numeric answers**, assign `answer = ...` and `print("Answer:", answer)`
5. **Never** use the string returned by `sql_engine`
6. Return **only executable Python** inside <code>...</code>

Question: {user_q}
"""
            raw = agent.run(prompt)

            # ---- Extract <code> block ----
            m = re.search(r"<code>(.*?)</code>", raw, re.DOTALL)
            if not m:
                st.info("No <code> block. Raw output:")
                st.write(raw)
                st.stop()

            code = m.group(1).strip()

            # ---- SAFE EXECUTION ENVIRONMENT ----
            exec_ns = {
                "st": st,                     # <-- THIS WAS MISSING
                "sql_engine": sql_engine,
                "pd": pd,
                "px": px,
                "plt": plt,
                "np": np,
            }

            try:
                exec(code, {}, exec_ns)               # run the generated code
                st.success("Code ran successfully")

                # ---- Auto-display results ----
                if "fig" in exec_ns:                  # Plotly figure
                    st.plotly_chart(exec_ns["fig"], use_container_width=True)
                elif plt.get_fignums():               # Matplotlib figure
                    st.pyplot(plt)

                # Show a printed answer if present
                if "answer" in exec_ns:
                    st.metric("Answer", exec_ns["answer"])

                # Fallback: show the DataFrame
                elif st.session_state.get("last_df") is not None:
                    st.dataframe(st.session_state.last_df.head(10))

            except Exception as e:
                st.error(f"Execution failed: {e}")
                st.code(code, language="python")
# -------------------------------
# Cleanup
# -------------------------------
def shutdown():
    conn.close()
    st.success("DB closed.")

st.sidebar.button("Close DB", on_click=shutdown)