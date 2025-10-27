import os
import sqlite3
import pandas as pd
import streamlit as st
from smolagents import LiteLLMModel, CodeAgent, tool
import re
import sqlglot
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.config import RunnableConfig

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

# Validate sand_friction_features and amplitude column
if 'sand_friction_features' not in table_schema:
    st.error("Table 'sand_friction_features' not found in the database.")
    st.stop()
if 'amplitude' not in table_schema.get('sand_friction_features', {}).get('columns', []):
    st.error("Column 'amplitude' not found in 'sand_friction_features' table.")
    st.stop()

# -------------------------------
# Generate table schema description
# -------------------------------
def generate_schema_description(schema):
    description = ""
    for table, info in schema.items():
        description += f"Table '{table}':\n  Columns:\n"
        for col, dtype in info['dtypes'].items():
            description += f"    - {col}: {dtype}\n"
    return description

tables_description = generate_schema_description(table_schema)

# -------------------------------
# Define the SQL tool for the agent
# -------------------------------
@tool
def sql_engine(query: str) -> str:
    """
    Executes SQL queries on the SQLite database.
    
    Args:
        query: The SQL query to execute. Must be valid SQLite syntax and use exact table/column names.
    
    Returns:
        str: String representation of the query result or error message.
    
    The available tables are:
    {tables_description}
    """
    # Debug: Log the query
    print("Generated SQL Query:\n", query)
    print("Available Tables:", db.get_table_names())
    
    # Validate SQL query
    try:
        sqlglot.parse_one(query, dialect="sqlite")
    except sqlglot.errors.ParseError as e:
        return f"❌ Invalid SQL query: {str(e)}"
    
    # Execute query
    try:
        result = db.run(query)
        return result if result else "No results returned."
    except Exception as e:
        return f"❌ Error executing SQL query: {str(e)}"

# -------------------------------
# Initialize LLM and agent
# -------------------------------
model = LiteLLMModel(
    model_id="ollama_chat/deepseek-coder-v2:16b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)

agent = CodeAgent(
    tools=[sql_engine],
    model=model,
    max_steps=5,
    additional_authorized_imports=["pandas", "numpy"]
)

# -------------------------------
# Streamlit UI
# -------------------------------
# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="SQLite → SQL Agent", layout="wide")
st.title("🧠 SQLite → SQL NLP Agent")

# Display schema and content side by side
st.subheader("📘 Database Schema and Content")
num_tables = len(table_schema)
cols = st.columns(min(num_tables, 3))  # Up to 3 tables per row for readability

for idx, table in enumerate(table_schema):
    with cols[idx % 3]:  # Distribute tables across columns
        with st.expander(f"Table: {table}", expanded=False):
            # Display schema as a table
            try:
                schema_data = [
                    {"Column Name": col, "Data Type": str(dtype)}
                    for col, dtype in table_schema[table]['dtypes'].items()
                ]
                df_schema = pd.DataFrame(schema_data)
                st.dataframe(
                    df_schema,
                    use_container_width=True,
                    column_config={
                        "Column Name": st.column_config.TextColumn("Column Name", width="medium"),
                        "Data Type": st.column_config.TextColumn("Data Type", width="medium"),
                    },
                    height=150,  # Compact height for schema
                )
                
                # Display table content on click (expander already handles click-to-expand)
                st.markdown("**Table Content (Top 5 Rows)**")
                try:
                    df_content = pd.read_sql(f"SELECT * FROM {table} LIMIT 5;", conn)
                    st.dataframe(
                        df_content,
                        use_container_width=True,
                        height=200,  # Slightly taller for content
                    )
                except Exception as e:
                    st.warning(f"⚠️ Error displaying content for {table}: {e}")
            except Exception as e:
                st.warning(f"⚠️ Error displaying schema for {table}: {e}")

# User input
st.subheader("💬 Ask a question about the database")
user_question = st.text_area(
    "Enter your question",
    placeholder="e.g., What is the highest amplitude in sand_friction_features?",
    value="show me the max amplitude in sand_friction_features"  # Fixed typo
)

# Clear cache button
if st.button("Clear Cache and Reload Tables"):
    st.cache_data.clear()
    table_schema = load_table_schema()
    tables_description = generate_schema_description(table_schema)
    st.success("Cache cleared and tables reloaded.")

# Run agent (unchanged)
if st.button("Run"):
    if user_question.strip():
        with st.spinner("🤖 Processing your question..."):
            # Correct typo in question (handled in your original code)
            
            # Build the agent prompt
            prompt = f"""
You have access to the following SQLite tables:
{tables_description}

INSTRUCTIONS:
- Write a SQLite query to answer the question, enclosed in <code>...</code> tags.
- Use ONLY the table and column names listed above (case-sensitive).
- If the query fails, inspect the error and try a different approach (e.g., verify table/column names).
- Return a single query that directly answers the question.
- Thoughts: Describe your reasoning before the query, ensuring it addresses the specific question.
- Example for question "What is the highest amplitude in sand_friction_features?":
Thoughts: The question asks for the maximum value in the 'amplitude' column of the 'sand_friction_features' table. The schema confirms 'sand_friction_features' has an 'amplitude' column (float64). Use SELECT MAX(amplitude) to get the highest value.
<code>
SELECT MAX(amplitude) FROM sand_friction_features
</code>

Question: {user_question}
"""
            # Debug: Show the prompt
            st.text("Prompt sent to agent:\n" + prompt)
            
            # Run the agent and handle output
            try:
                output = agent.run(prompt)
                # Handle different possible outputs from CodeAgent
                if isinstance(output, list) and len(output) > 0 and 'function' in output[0]:
                    # Extract query from tool call
                    if output[0]['function']['name'] == 'sql_engine':
                        query = output[0]['function']['arguments']
                        output = sql_engine(query)
                    else:
                        output = f"❌ Unexpected tool call: {output[0]['function']['name']}"
                elif isinstance(output, str):
                    # Extract query from <code> tags
                    query_match = re.search(r"<code>(.*?)</code>", output, re.DOTALL)
                    if query_match:
                        query = query_match.group(1).strip()
                        output = sql_engine(query)
                    else:
                        # Try triple backticks as fallback
                        query_match = re.search(r"```sql\n(.*?)\n```", output, re.DOTALL)
                        if query_match:
                            query = query_match.group(1).strip()
                            output = sql_engine(query)
                        else:
                            output = f"❌ No valid SQL query found in output: {output}"
                else:
                    output = f"❌ Unexpected output format: {type(output)}"
            except Exception as e:
                output = f"❌ Error running agent: {str(e)}"
            
        st.success("✅ Finished")
        st.subheader("Answer / Output")
        st.text(output)
    else:
        st.warning("Please enter a question first.")

# Close SQLite connection
if conn:
    conn.close()