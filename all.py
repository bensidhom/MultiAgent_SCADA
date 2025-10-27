import os
import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from smolagents import LiteLLMModel, CodeAgent, tool
import re
import sqlglot
from langchain_community.utilities.sql_database import SQLDatabase
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import cv2

# -------------------------------
# Path to SQLite database
# -------------------------------
DB_PATH = r"C:\all\github_projects\MultiAgent_SCADA\output_data.db"

if not os.path.isfile(DB_PATH):
    st.error(f"❌ Database file not found at: {DB_PATH}")
    st.stop()

# -------------------------------
# Connect to SQLite and initialize SQLDatabase
# -------------------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# -------------------------------
# Load table schema for UI and prompt
# -----------------------
@st.cache_data
def load_table_schema():
    try:
        table_names = db.get_table_names()
        if not table_names:
            st.error("No tables found in the database.")
            return {}
        schema = {}
        for table in table_names:
            try:
                df = pd.read_sql(f"SELECT * FROM {table} LIMIT 1;", conn)
                schema[table] = {"columns": list(df.columns), "dtypes": df.dtypes.to_dict()}
            except Exception as e:
                st.error(f"Error loading schema for table {table}: {e}")
        st.success(f"Loaded tables: {', '.join(table_names)}")
        return schema
    except Exception as e:
        st.error(f"Error loading table schema: {e}")
        return {}

table_schema = load_table_schema()
if not table_schema:
    st.error("No table schema loaded. Check database path and contents.")
    st.stop()

# -------------------------------
# Generate table schema description
# -----------------------
def generate_schema_description(schema):
    description = ""
    for table, info in schema.items():
        description += f"Table '{table}':\n  Columns:\n"
        for col, dtype in info['dtypes'].items():
            description += f"    - {col}: {dtype}\n"
    return description

tables_description = generate_schema_description(table_schema)

# -------------------------------
# Define tools for the agent
# -----------------------
@tool
def sql_engine(query: str) -> pd.DataFrame:
    """
    Executes SQL queries on the SQLite database and returns a pandas DataFrame.
    
    Args:
        query: The SQL query to execute. Must be valid SQLite syntax and use exact table/column names.
    
    Returns:
        pd.DataFrame: DataFrame containing the query results. Returns:
            - A multi-column DataFrame for table results (multiple rows/columns).
            - A single-column DataFrame for vector results (one column, multiple rows).
            - A single-cell DataFrame for scalar results (one row, one column).
            - A single-row DataFrame for one-row results (one row, multiple columns).
            - An empty DataFrame with an error message in st.error if the query fails or returns no data.
            - Cell contents (e.g., strings, numbers, serialized lists) are preserved as-is.
    
    The available tables are:
    {tables_description}
    """
    print("Generated SQL Query:\n", query)
    print("Available Tables:", db.get_table_names())
    
    # Validate SQL syntax
    try:
        sqlglot.parse_one(query, dialect="sqlite")
    except sqlglot.errors.ParseError as e:
        st.error(f"❌ Invalid SQL query: {str(e)}")
        return pd.DataFrame()
    
    # Execute query and return results as DataFrame
    try:
        df = pd.read_sql(query, conn)
        if df.empty:
            st.warning("⚠️ Query executed successfully but returned no data.")
            return pd.DataFrame()
        # Provide feedback based on result shape
        if df.shape[0] == 1 and df.shape[1] == 1:
            st.info("ℹ️ Query returned a single value.")
        elif df.shape[0] == 1:
            st.info("ℹ️ Query returned a single row.")
        elif df.shape[1] == 1:
            st.info("ℹ️ Query returned a single column (vector).")
        # Preserve cell contents (e.g., strings, serialized lists) as-is
        return df
    except Exception as e:
        st.error(f"❌ Error executing SQL query: {str(e)}")
        return pd.DataFrame()
    


@tool
def plot_engine(
    data: pd.DataFrame,
    x_col: str = None,
    value_cols: list = None,
    plot_type: str = "line",
    image_col: str = None
) -> str:
    """
    Generates a plot from a DataFrame and displays images from a specified column.
    - Columns with lists in cells are plotted as waveforms (x-axis = indices of list elements).
    - Otherwise, plots normally using DataFrame columns.
    - Falls back to Matplotlib if Plotly fails.

    Args:
        data: pandas DataFrame from sql_engine.
        x_col: Column name for x-axis; defaults to DataFrame index.
        value_cols: List of columns to plot; defaults to numeric columns.
        plot_type: 'line', 'scatter', 'bar', 'area'.
        image_col: Column containing image paths to display.

    Returns:
        str: Success or error message.
    """
    try:
        if not isinstance(data, pd.DataFrame) or data.empty:
            return "❌ No valid data provided for plotting."
        
        # Default x_col
        if x_col is None or x_col not in data.columns:
            data = data.reset_index()
            x_col = data.columns[0]

        # Default value_cols
        if value_cols is None:
            value_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if not value_cols:
                value_cols = [col for col in data.columns if col != x_col]
            if not value_cols:
                return "❌ No valid columns to plot."

        valid_plot_types = ['line', 'scatter', 'bar', 'area']
        if plot_type not in valid_plot_types:
            return f"❌ Invalid plot type '{plot_type}'. Choose from {valid_plot_types}."

        # Plotly plotting
        fig = go.Figure()
        for col in value_cols:
            if data[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
                for i, lst in enumerate(data[col]):
                    if not isinstance(lst, (list, tuple, np.ndarray)):
                        lst = [lst]
                    
                    # Downsample for plotting if too long
                    max_points = 500  # maximum points to plot
                    if len(lst) > max_points:
                        step = len(lst) // max_points
                        lst_plot = lst[::step]
                        x_vals = list(range(0, len(lst), step))
                    else:
                        lst_plot = lst
                        x_vals = list(range(len(lst)))
                    
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=lst_plot,
                        mode='lines',
                        name=f"{col}_{i}"
                    ))
            else:
                if plot_type == 'line':
                    fig.add_trace(go.Scatter(x=data[x_col], y=data[col], mode='lines', name=col))
                elif plot_type == 'scatter':
                    fig.add_trace(go.Scatter(x=data[x_col], y=data[col], mode='markers', name=col))
                elif plot_type == 'bar':
                    fig.add_trace(go.Bar(x=data[x_col], y=data[col], name=col))
                elif plot_type == 'area':
                    fig.add_trace(go.Scatter(x=data[x_col], y=data[col], fill='tozeroy', name=col))

        fig.update_layout(
            title=f"{plot_type.capitalize()} Plot: {', '.join(value_cols)} vs {x_col}",
            xaxis_title="Index" if x_col is None else x_col,
            yaxis_title="Values",
            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly"
        )
        st.plotly_chart(fig, width="stretch")

        # Display images
        if image_col and image_col in data.columns:
            paths = data[image_col].dropna()
            for path in paths:
                try:
                    img = Image.open(path)
                    st.image(img, caption=f"Image: {path}", width="stretch")
                except Exception as e1:
                    try:
                        img = cv2.imread(path)
                        if img is None:
                            raise ValueError(f"OpenCV could not load image: {path}")
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, caption=f"Image: {path}", width="stretch")
                    except Exception as e2:
                        st.warning(f"⚠️ Failed to load image '{path}': PIL error ({str(e1)}), OpenCV error ({str(e2)})")

        return f"{plot_type.capitalize()} plot and images (if any) generated successfully."

    except Exception as e_plotly:
        # Fallback to Matplotlib
        try:
            plt.figure(figsize=(10, 5))
            for col in value_cols:
                if data[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
                    for i, lst in enumerate(data[col]):
                        if not isinstance(lst, (list, tuple, np.ndarray)):
                            lst = [lst]
                        plt.plot(range(len(lst)), lst, label=f"{col}_{i}")
                else:
                    if plot_type == "line":
                        plt.plot(data[x_col], data[col], label=col)
                    elif plot_type == "scatter":
                        plt.scatter(data[x_col], data[col], label=col)
                    elif plot_type == "bar":
                        plt.bar(data[x_col], data[col], label=col)
                    elif plot_type == "area":
                        plt.fill_between(data[x_col], data[col], label=col)
            plt.xlabel(x_col)
            plt.ylabel("Values")
            plt.title(f"{plot_type.capitalize()} Plot: {', '.join(value_cols)} vs {x_col}")
            plt.legend()
            st.pyplot(plt.gcf())
            return f"Matplotlib fallback {plot_type} plot generated successfully (Plotly failed: {str(e_plotly)})."
        except Exception as e_matplotlib:
            return f"❌ Error generating plot: Plotly failed ({str(e_plotly)}), Matplotlib failed ({str(e_matplotlib)})."

# Initialize single LLM model and agent
# -----------------------
model = LiteLLMModel(
    model_id="ollama_chat/deepseek-coder-v2:16b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)


# -------------------------------
# Utility: extract code from output
# -----------------------
def extract_code(agent_text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", agent_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match2 = re.search(r"<code>(.*?)</code>", agent_text, re.DOTALL)
    if match2:
        return match2.group(1).strip()
    lines = agent_text.split("\n")
    code_lines = [l for l in lines if l.strip().startswith(("import", "plt", "np", "pd", "st", "#", "data =", "result =", "plot_engine", "st.dataframe"))]
    return "\n".join(code_lines)

# -------------------------------
# Streamlit UI
# -----------------------
st.set_page_config(layout="wide")
st.title("🧠 Defect Analysis Agent")

# ---- Database Schema and Content ----
st.subheader("📘 Database Schema and Content")
num_tables = len(table_schema)
cols = st.columns(min(num_tables, 3))

for idx, table in enumerate(table_schema):
    with cols[idx % 3]:
        with st.expander(f"Table: {table}", expanded=True):
            try:
                schema_data = [
                    {"Column Name": col, "Data Type": str(dtype)}
                    for col, dtype in table_schema[table]['dtypes'].items()
                ]
                df_schema = pd.DataFrame(schema_data)
                st.dataframe(
                    df_schema,
                   width="stretch",
                    column_config={
                        "Column Name": st.column_config.TextColumn("Column Name", width="medium"),
                        "Data Type": st.column_config.TextColumn("Data Type", width="medium"),
                    },
                    height=150,
                )
                
                st.markdown("**Table Content (Top 5 Rows)**")
                try:
                    df_content = pd.read_sql(f"SELECT * FROM {table} LIMIT 5;", conn)
                    st.dataframe(
                        df_content,
                        width="stretch",
                        height=200,
                    )
                except Exception as e:
                    st.warning(f"⚠️ Error displaying content for {table}: {e}")
            except Exception as e:
                st.warning(f"⚠️ Error displaying schema for {table}: {e}")

if st.button("Clear Cache and Reload Tables"):
    st.cache_data.clear()
    table_schema = load_table_schema()
    tables_description = generate_schema_description(table_schema)
    st.success("Cache cleared and tables reloaded.")
    st.rerun()

# ---- Combined Fetch & Plot ----

if "conversation" not in st.session_state:
    st.session_state.conversation = []




# ---- Utility: Logic Assertion ----
def assert_agent_logic(code_text: str, user_input: str) -> bool:
    """
    Validate that the generated code follows the required logic:
    1. Must call sql_engine first.
    2. Only call plot_engine if plotting or images are requested.
    3. Use display_dataframe (or st.dataframe) for non-plotting cases.
    """
    if "sql_engine(" not in code_text:
        st.error("❌ Agent error: Code must include a call to sql_engine.")
        return False

    plot_keywords = [
        "plot", "graph", "chart", "visualize", "time series",
        "bar", "line", "scatter", "area", "image",
        "show image", "display image", "picture"
    ]
    needs_plot = any(keyword in user_input.lower() for keyword in plot_keywords)
    has_plot_engine = "plot_engine(" in code_text

    if needs_plot and not has_plot_engine:
        st.error("❌ Agent error: Plot or image requested but plot_engine not called.")
        return False
    if not needs_plot and has_plot_engine:
        st.error("❌ Agent error: plot_engine called without plot/image request.")
        return False
    if not needs_plot and "display_dataframe(" not in code_text and "st.dataframe(" not in code_text:
        st.error("❌ Agent error: Non-plotting request must include display_dataframe or st.dataframe.")
        return False

    return True


# ---- Optional Tool: display_dataframe ----
@tool
def display_dataframe(df: pd.DataFrame, title: str = "") -> str:
    """
    Display a pandas DataFrame in Streamlit with optional title formatting.

    This utility is designed for use within the agent to present query results or
    analysis outputs in a structured, readable way. It automatically detects empty
    DataFrames, supports dynamic titling, and uses Streamlit’s responsive layout.

    Args:
        df (pd.DataFrame): The DataFrame to display.
        title (str, optional): A title to display above the table. Defaults to "".

    Returns:
        str: A message describing the display result.
            - "✅ DataFrame displayed (...)" if successful.
            - "Empty DataFrame displayed." if the DataFrame is empty.
            - "❌ Error displaying DataFrame: <error>" if an exception occurs.

    Example:
        ```python
        data = sql_engine("SELECT * FROM sand_friction_features LIMIT 10")
        display_dataframe(data, title="Sand Friction Features Sample")
        ```
    """
    try:
        if df.empty:
            st.warning("Empty DataFrame")
            return "Empty DataFrame displayed."
        if title:
            st.markdown(f"### {title}")
        st.dataframe(df, width="stretch")
        return f"✅ DataFrame displayed ({df.shape[0]} rows, {df.shape[1]} columns)."
    except Exception as e:
        return f"❌ Error displaying DataFrame: {str(e)}"


# ---- Model Connection Check (Fixed) ----
try:
    # Create a minimal temporary agent to test model availability
    _test_agent = CodeAgent(
        tools=[],
        model=model,
        max_steps=1
    )
    _ = _test_agent.run("Say 'connected' if you are available.")
    st.success("✅ Model server connected.")
except Exception as e:
    st.error(f"❌ Model server error: {e}")
    st.stop()



# ---- Agent Definition ----
agent = CodeAgent(
    tools=[sql_engine, plot_engine, display_dataframe],
    model=model,
    max_steps=2,
    additional_authorized_imports=[
        "numpy", "matplotlib", "matplotlib.pyplot", "pandas", "streamlit",
        "os", "plotly.express", "plotly.graph_objects", "PIL", "cv2",
        "sqlglot", "langchain_community"
    ]
)






col_input, col_code, col_plot = st.columns([2, 2, 3])

with col_input:
    user_input = st.text_input("Ask the agent:", placeholder="e.g., fetch and plot waves from sand_friction_features as time series")

    if st.button("Run") and user_input:
        # Dynamically identify potential columns based on schema

        prompt = f"""
You are a data-analysis assistant that works with a SQLite database.
Available tables (with columns and dtypes) are listed below:
{tables_description}

### INSTRUCTIONS (follow **exactly**)

1. **Always start by calling `sql_engine` with a valid SQLite query** that returns the data you need.
   - Use only the table/column names shown above (case-sensitive).
   - If the user asks for a single value, a vector, or a full table, still fetch it with `sql_engine`.
   - Example:  
     ```python
     data = sql_engine("SELECT MAX(amplitude) AS max_amp FROM sand_friction_features")

2. Only call plot_engine if the user explicitly requests:
  Plotting terms: "plot", "graph", "chart", "visualize", "time series", "bar", "line", "scatter", "area".
  Image display: "image", "show image", "display image", "picture".
  Example:
    ```python
    data = sql_engine("SELECT amplitude, image_path FROM sand_friction_features")
    plot_engine(data, x_col=None, value_cols=["amplitude"], plot_type="line", image_col="image_path")

3. If NO plotting requested, display the raw DataFrame result with st.dataframe(result)
4. Use exact table/column names from the schema above (case-sensitive)

WORKFLOW:
1. sql_engine("SELECT ...") → store as 'data' or 'result'
2. IF plotting requested: plot_engine(data, x_col=..., value_cols=[...], plot_type=..., image_col=...)
3. ELSE: display_dataframe(result, "Query Results")

EXAMPLES:

Question: "What is the highest amplitude?"
Thoughts: User wants a single value. Use sql_engine to fetch MAX(amplitude) and display it.
<code>
data = sql_engine("SELECT MAX(amplitude) AS max_amplitude FROM sand_friction_features")
display_dataframe(data, "Query Results")
</code>

Question: "Show top 5 rows of sand_friction_features"
Thoughts: User wants a table. Use sql_engine to fetch top 5 rows and display with st.dataframe.
<code>
data = sql_engine("SELECT * FROM sand_friction_features LIMIT 5")
display_dataframe(data, "Query Results")
</code>

Question: "Plot amplitude vs timestamp as line"
Thoughts: User requests a line plot. Fetch timestamp and amplitude with sql_engine, then use plot_engine.
<code>
data = sql_engine("SELECT amplitude FROM sand_friction_features")
plot_engine(data, x_col=None, value_cols=["amplitude"], plot_type="line")
</code>




Question: {user_input}
"""
        try:
            # ---- Agent run ----
            output = agent.run(prompt)
            code_text = extract_code(output)
            explanation_text = output.replace(code_text, "").strip()

            # ---- Validate Logic ----
            if not assert_agent_logic(code_text, user_input):
                st.session_state.conversation.append({
                    "role": "assistant",
                    "content": f"Agent error: Invalid code logic. {explanation_text}",
                    "code": code_text
                })
                st.stop()

            # ---- Execute Safe Code ----
            local_vars = {
                "st": st, "plt": plt, "pd": pd, "np": np, "os": os,
                "sql_engine": sql_engine, "plot_engine": plot_engine,
                "px": px, "go": go, "Image": Image, "cv2": cv2,
                "sqlglot": sqlglot, "SQLDatabase": SQLDatabase,
                "display_dataframe": display_dataframe,
                "__builtins__": {}
            }
            exec(code_text, local_vars)

            # ---- Store Conversation ----
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
                "code": None
            })

    # ---- Conversation Display ----
    st.subheader("💬 Interaction")
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


with col_plot:
    st.subheader("📊 Visualization & Images")
    st.info("Plots and images will appear directly after code execution.")


# ---- Close SQLite Connection Safely ----
if 'conn' in locals() and conn:
    conn.close()