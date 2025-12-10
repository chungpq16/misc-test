import os
import json
import requests
import duckdb
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables from .env
load_dotenv()

# -----------------------------
# Confluence API & HTML parsing
# -----------------------------


def fetch_confluence_html(
    base_url: str,
    page_id: str,
    token: str,
) -> str:
    """
    Call Confluence REST API and return body.storage.value (HTML string).
    """
    url = f"{base_url}/rest/api/content/{page_id}"
    params = {
        "expand": "body.storage,version,space"
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    html = data["body"]["storage"]["value"]
    return html


def html_table_to_dataframe(html: str, wanted_columns=None, table_type="env") -> pd.DataFrame:
    """
    Parse Confluence HTML containing (possibly nested) tables into a pandas DataFrame.

    - Uses the FIRST top-level table it finds.
    - Removes nested tables inside each cell.
    - Extracts visible text only (no URLs).
    - Keeps only wanted_columns if provided.
    - table_type: "env" for environment table, "lb" for loadbalancer table
    """

    if wanted_columns is None:
        if table_type == "lb":
            wanted_columns = [
                "Cluster",
                "Project",
                "Namespace",
                "Service:Port",
                "LB-IP Address",
                "LB Port",
                "SecZone",
                "Zone Managers",
                "DCS T&R",
                "Status",
            ]
        else:
            wanted_columns = [
                "Project",
                "Customer",
                "Responsible Person",
                "Cluster",
                "SNAT-IP",
                "Security zone - Egress",
            ]

    soup = BeautifulSoup(html, "html.parser")

    # Try to find the main table with class="relative-table"
    main_table = soup.find("table", class_="relative-table")
    if main_table is None:
        # Fallback: first table in the HTML
        main_table = soup.find("table")

    if main_table is None:
        raise ValueError("No <table> found in Confluence HTML.")

    tbody = main_table.find("tbody") or main_table
    rows = tbody.find_all("tr", recursive=False)
    if not rows:
        raise ValueError("No <tr> rows found in main table.")

    # Header row
    header_cells = rows[0].find_all(["th", "td"], recursive=False)
    headers = [c.get_text(" ", strip=True) for c in header_cells]
    col_index = {name: idx for idx, name in enumerate(headers) if name}

    wanted_columns_existing = [c for c in wanted_columns if c in col_index]
    if not wanted_columns_existing:
        raise ValueError(
            f"None of the expected columns {wanted_columns} were found in table headers: {headers}"
        )

    # Data rows
    data_rows = []
    for row in rows[1:]:
        cells = row.find_all(["td", "th"], recursive=False)
        if not cells:
            continue

        row_dict = {}
        for col_name in wanted_columns_existing:
            idx = col_index[col_name]
            if idx >= len(cells):
                row_dict[col_name] = ""
                continue

            cell = cells[idx]

            # Remove nested tables in this cell
            for nested in cell.find_all("table"):
                nested.decompose()

            # Remove Confluence macros if needed
            for macro in cell.find_all("ac:structured-macro"):
                macro.decompose()

            text = cell.get_text(" ", strip=True)
            row_dict[col_name] = text

        # Skip rows that are entirely empty
        if any(str(v).strip() for v in row_dict.values()):
            data_rows.append(row_dict)

    df = pd.DataFrame(data_rows, columns=wanted_columns_existing)
    return df


# -----------------------------
# Caching layer (Phase 1) – now supports multiple pages
# -----------------------------


@st.cache_data(ttl=300, show_spinner=False)
def cached_load_table_single(base_url: str, page_id: str, token: str, table_type: str = "env") -> pd.DataFrame:
    """
    Cached loader for a single Confluence page.
    table_type: "env" or "lb"
    """
    html = fetch_confluence_html(base_url, page_id, token)
    df = html_table_to_dataframe(html, table_type=table_type)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def cached_load_tables_combined(
    base_url: str,
    token: str,
    page_ids: tuple,
    table_type: str = "env",
) -> pd.DataFrame:
    """
    Load & combine multiple Confluence pages into a single dataframe.

    - page_ids: tuple of page IDs (strings). Empty strings are ignored.
    - Adds a '__source_page_id__' column to indicate origin of each row.
    - table_type: "env" or "lb"
    """
    dfs = []
    for pid in page_ids:
        pid = str(pid).strip()
        if not pid:
            continue
        df_single = cached_load_table_single(base_url, pid, token, table_type=table_type)
        df_single = df_single.copy()
        df_single["__source_page_id__"] = pid
        dfs.append(df_single)

    if not dfs:
        raise ValueError("No valid page IDs provided.")

    combined = pd.concat(dfs, ignore_index=True)
    return combined


# -----------------------------
# Excel file loading
# -----------------------------

@st.cache_data(ttl=300, show_spinner=False)
def load_excel_file(file_path: str) -> pd.DataFrame:
    """
    Load an Excel file and return as DataFrame.
    """
    df = pd.read_excel(file_path)
    return df


# -----------------------------
# DuckDB setup (in-memory)
# -----------------------------

# Mapping from original column names to SQL-friendly ones
COL_MAP_ENV = {
    "Project": "project",
    "Customer": "customer",
    "Responsible Person": "responsible_person",
    "Cluster": "cluster",
    "SNAT-IP": "snat_ip",
    "Security zone - Egress": "security_zone_egress",
}

COL_MAP_LB = {
    "Cluster": "cluster",
    "Project": "project",
    "Namespace": "namespace",
    "Service:Port": "service_port",
    "LB-IP Address": "lb_ip_address",
    "LB Port": "lb_port",
    "SecZone": "seczone",
    "Zone Managers": "zone_managers",
    "DCS T&R": "dcs_tr",
    "Status": "status",
}

# Excel table - will use columns as-is, converting to snake_case
def excel_columns_to_sql(df: pd.DataFrame) -> dict:
    """
    Convert Excel column names to SQL-friendly names (snake_case).
    """
    col_map = {}
    for col in df.columns:
        # Convert to lowercase and replace spaces/special chars with underscore
        sql_name = col.strip().lower().replace(" ", "_").replace("-", "_").replace(":", "_")
        # Remove any duplicate underscores
        sql_name = "_".join(filter(None, sql_name.split("_")))
        col_map[col] = sql_name
    return col_map


def build_sql_dataframe(df_raw: pd.DataFrame, table_type: str = "env") -> pd.DataFrame:
    """
    Return a copy of df_raw with columns renamed to SQL-friendly names
    based on COL_MAP and restricted to those columns.
    Keeps '__source_page_id__' if present (useful for debugging).
    table_type: "env", "lb", or "excel"
    """
    if table_type == "excel":
        COL_MAP = excel_columns_to_sql(df_raw)
        df_sql = df_raw.rename(columns=COL_MAP).copy()
    else:
        COL_MAP = COL_MAP_LB if table_type == "lb" else COL_MAP_ENV
        cols_present = [c for c in COL_MAP.keys() if c in df_raw.columns]
        df_sql = df_raw[cols_present].rename(columns=COL_MAP).copy()

    # optionally keep source page information
    if "__source_page_id__" in df_raw.columns:
        df_sql["source_page_id"] = df_raw["__source_page_id__"].values

    return df_sql


def execute_sql_on_df(df_env: pd.DataFrame = None, df_lb: pd.DataFrame = None, df_excel: pd.DataFrame = None, sql: str = "") -> pd.DataFrame:
    """
    Execute a SQL query against dataframes using DuckDB in-memory.
    The tables are registered as 'env_table', 'lb_table', and 'excel_table'.
    """
    con = duckdb.connect(database=":memory:")
    try:
        if df_env is not None:
            con.register("env_table", df_env)
        if df_lb is not None:
            con.register("lb_table", df_lb)
        if df_excel is not None:
            con.register("excel_table", df_excel)
        result_df = con.execute(sql).df()
    finally:
        con.close()
    return result_df


# -----------------------------
# LLM setup (Ollama llama3.1)
# -----------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)


# -----------------------------
# Phase 2 – Text-to-SQL chain
# -----------------------------

SQL_SYSTEM_PROMPT = (
    "You are a senior data engineer that writes SQL for DuckDB.\n\n"
    "You are given schemas of tables: env_table (environment data), lb_table (loadbalancer data), and/or excel_table (Excel file data).\n"
    "Your job is to translate the user's natural language question into a single SQL query.\n\n"
    "Rules:\n"
    "- Query the appropriate table(s): env_table, lb_table, and/or excel_table.\n"
    "- Only use the columns provided in the schema.\n"
    "- Use ILIKE or LOWER() for case-insensitive matching on text when appropriate.\n"
    "- Do NOT use backticks. Use standard SQL (DuckDB).\n"
    "- Do NOT include comments or explanations.\n"
    "- Return ONLY the SQL query text, nothing else. No ``` fences.\n"
)


def generate_sql_from_question(question: str, df_env_sql: pd.DataFrame = None, df_lb_sql: pd.DataFrame = None, df_excel_sql: pd.DataFrame = None) -> str:
    """
    LLM Call 1: Given the question and schemas, generate a SQL query string.
    """
    # Build schema description
    schema_parts = []
    
    if df_env_sql is not None:
        env_schema_lines = []
        for col in df_env_sql.columns:
            dtype = str(df_env_sql[col].dtype)
            env_schema_lines.append(f"  - {col} ({dtype})")
        env_schema_str = "\n".join(env_schema_lines)
        schema_parts.append(f"Table: env_table\nColumns:\n{env_schema_str}")
    
    if df_lb_sql is not None:
        lb_schema_lines = []
        for col in df_lb_sql.columns:
            dtype = str(df_lb_sql[col].dtype)
            lb_schema_lines.append(f"  - {col} ({dtype})")
        lb_schema_str = "\n".join(lb_schema_lines)
        schema_parts.append(f"Table: lb_table\nColumns:\n{lb_schema_str}")
    
    if df_excel_sql is not None:
        excel_schema_lines = []
        for col in df_excel_sql.columns:
            dtype = str(df_excel_sql[col].dtype)
            excel_schema_lines.append(f"  - {col} ({dtype})")
        excel_schema_str = "\n".join(excel_schema_lines)
        schema_parts.append(f"Table: excel_table\nColumns:\n{excel_schema_str}")
    
    schema_str = "\n\n".join(schema_parts)

    prompt = (
        f"{schema_str}\n\n"
        f"User question: {question}\n\n"
        f"Write a single SQL query for DuckDB that answers this question."
    )

    messages = [
        SystemMessage(content=SQL_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    resp = llm.invoke(messages)
    raw_sql = resp.content.strip()

    # Strip any surrounding code fences if model adds them
    if "```" in raw_sql:
        parts = raw_sql.split("```")
        if len(parts) >= 3:
            raw_sql = parts[1]
        raw_sql = raw_sql.strip()

    if raw_sql.endswith(";"):
        raw_sql = raw_sql[:-1].strip()

    return raw_sql


ANSWER_SYSTEM_PROMPT = (
    "You are a helpful data analyst.\n\n"
    "You will be given:\n"
    "- The original user question.\n"
    "- The SQL query that was executed.\n"
    "- The query result as JSON (list of rows).\n\n"
    "Your job is to answer the question in clear, concise natural language.\n"
    "If the result is empty, say that no matching data was found.\n"
    "If there are rows, summarize them and highlight the key values.\n"
)


def generate_answer_from_result(
    question: str,
    sql: str,
    result_df: pd.DataFrame,
) -> str:
    """
    LLM Call 2: Given the question, SQL, and result dataframe, generate a natural language answer.
    """
    if result_df is None or result_df.empty:
        result_json = "[]"
    else:
        result_json = result_df.to_json(orient="records")

    prompt = (
        f"User question:\n{question}\n\n"
        f"Executed SQL:\n{sql}\n\n"
        f"Result rows (JSON):\n{result_json}\n\n"
        f"Now answer the user's question based on this result."
    )

    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    resp = llm.invoke(messages)
    return resp.content.strip()


def text_to_sql_chain(question: str, df_env_raw: pd.DataFrame = None, df_lb_raw: pd.DataFrame = None, df_excel_raw: pd.DataFrame = None):
    """
    Full flow for Phase 2 over the combined dataframes:
    1) Build SQL-friendly copies of dataframes.
    2) LLM: question -> SQL.
    3) Execute SQL on DuckDB.
    4) LLM: (question + sql + result) -> final answer.
    Returns (sql, result_df, final_answer).
    """
    if (df_env_raw is None or df_env_raw.empty) and (df_lb_raw is None or df_lb_raw.empty) and (df_excel_raw is None or df_excel_raw.empty):
        raise ValueError("No dataframes available; load data first.")

    # Build SQL-friendly dataframes
    df_env_sql = build_sql_dataframe(df_env_raw, "env") if df_env_raw is not None and not df_env_raw.empty else None
    df_lb_sql = build_sql_dataframe(df_lb_raw, "lb") if df_lb_raw is not None and not df_lb_raw.empty else None
    df_excel_sql = build_sql_dataframe(df_excel_raw, "excel") if df_excel_raw is not None and not df_excel_raw.empty else None

    # Step 1: LLM generates SQL
    sql = generate_sql_from_question(question, df_env_sql, df_lb_sql, df_excel_sql)

    # Step 2: Execute SQL
    try:
        result_df = execute_sql_on_df(df_env_sql, df_lb_sql, df_excel_sql, sql)
    except Exception as e:
        raise RuntimeError(f"SQL execution error: {e}\nSQL: {sql}") from e

    # Step 3: LLM summarizes result
    answer = generate_answer_from_result(question, sql, result_df)

    return sql, result_df, answer


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(
        page_title="Confluence Multi-Page + DuckDB Text-to-SQL",
        layout="wide",
    )

    st.title("📄 Confluence Multi-Page Viewer + 🐤 DuckDB Text-to-SQL Chat")

    st.markdown(
        """
**Phase 1 — Data Ingestion & Display**

- Loads **Environment** tables (ENVPAGE_IDS) and **LoadBalancer** tables (LB_PAGE_IDS) from Confluence.  
- Loads **Excel** file (EXCEL_FILE_PATH) from local filesystem.  
- Cleans nested tables / links and keeps only key columns.  
- Adds a `__source_page_id__` column to Confluence data.  
- Caches the result for 5 minutes.  

**Phase 2 — Text-to-SQL Chat Chain**

- LLM Call 1: question → SQL over `env_table`, `lb_table`, and/or `excel_table` (DuckDB).  
- Execute SQL against in-memory DuckDB (all tables available).  
- LLM Call 2: question + result → natural language answer.
"""
    )

    st.sidebar.header("Confluence Settings")

    default_base_url = os.getenv(
        "CONFLUENCE_BASE_URL",
        "https://inside-docupedia.example.com/confluence",
    )

    # Read ENV page IDs from .env as comma-separated list
    env_page_ids_str = os.getenv("ENVPAGE_IDS", "")
    env_page_ids = [pid.strip() for pid in env_page_ids_str.split(",") if pid.strip()]
    
    # Read LB page IDs from .env as comma-separated list
    lb_page_ids_str = os.getenv("LB_PAGE_IDS", "")
    lb_page_ids = [pid.strip() for pid in lb_page_ids_str.split(",") if pid.strip()]

    default_token = os.getenv("CONFLUENCE_TOKEN", "")

    base_url = st.sidebar.text_input("Confluence Base URL", value=default_base_url)
    
    st.sidebar.info(
        f"**Environment Table Pages (ENVPAGE_IDS):** {len(env_page_ids)} page(s)\n\n"
        f"IDs: {', '.join(env_page_ids) if env_page_ids else 'None'}\n\n"
        f"**LoadBalancer Table Pages (LB_PAGE_IDS):** {len(lb_page_ids)} page(s)\n\n"
        f"IDs: {', '.join(lb_page_ids) if lb_page_ids else 'None'}"
    )

    token = st.sidebar.text_input(
        "Bearer Token",
        value=default_token,
        type="password",
        help="Value also taken from CONFLUENCE_TOKEN and CONFLUENCE_PAGE_IDS in .env if set.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Excel File")
    
    # Excel file path from .env or text input
    default_excel_path = os.getenv("EXCEL_FILE_PATH", "")
    excel_file_path = st.sidebar.text_input(
        "Excel File Path",
        value=default_excel_path,
        help="Absolute path to .xlsx file. Can also be set via EXCEL_FILE_PATH in .env",
    )
    
    if st.sidebar.button("Load Excel File"):
        if not excel_file_path:
            st.error("Please provide an Excel file path.")
        elif not os.path.exists(excel_file_path):
            st.error(f"File not found: {excel_file_path}")
        else:
            try:
                with st.spinner("Loading Excel file..."):
                    df_excel = load_excel_file(excel_file_path)
                    st.session_state["excel_df"] = df_excel
                    st.session_state["excel_file_path"] = excel_file_path
                    st.success(f"✅ Loaded Excel file: {len(df_excel)} rows, {len(df_excel.columns)} columns")
            except Exception as e:
                st.error(f"Error loading Excel file: {e}")

    st.sidebar.markdown("---")
    if st.sidebar.button("Load / Refresh Tables from Confluence"):
        if not token:
            st.error("Please provide a Bearer token.")
        elif not env_page_ids and not lb_page_ids:
            st.error("No Page IDs found in .env (ENVPAGE_IDS or LB_PAGE_IDS). Please configure them.")
        else:
            errors = []
            
            # Load Environment tables
            if env_page_ids:
                with st.spinner(f"Fetching Environment table(s) ({len(env_page_ids)} page(s))..."):
                    try:
                        df_env = cached_load_tables_combined(
                            base_url,
                            token,
                            tuple(env_page_ids),
                            table_type="env",
                        )
                        st.session_state["confluence_env_df"] = df_env
                        st.session_state["env_page_ids"] = env_page_ids
                        st.success(f"✅ Loaded {len(env_page_ids)} ENV page(s), {len(df_env)} rows")
                    except Exception as e:
                        errors.append(f"ENV tables error: {e}")
            
            # Load LoadBalancer tables
            if lb_page_ids:
                with st.spinner(f"Fetching LoadBalancer table(s) ({len(lb_page_ids)} page(s))..."):
                    try:
                        df_lb = cached_load_tables_combined(
                            base_url,
                            token,
                            tuple(lb_page_ids),
                            table_type="lb",
                        )
                        st.session_state["confluence_lb_df"] = df_lb
                        st.session_state["lb_page_ids"] = lb_page_ids
                        st.success(f"✅ Loaded {len(lb_page_ids)} LB page(s), {len(df_lb)} rows")
                    except Exception as e:
                        errors.append(f"LB tables error: {e}")
            
            if errors:
                for err in errors:
                    st.error(err)

    df_env = st.session_state.get("confluence_env_df")
    df_lb = st.session_state.get("confluence_lb_df")
    df_excel = st.session_state.get("excel_df")
    loaded_env_page_ids = st.session_state.get("env_page_ids", [])
    loaded_lb_page_ids = st.session_state.get("lb_page_ids", [])
    excel_file_path = st.session_state.get("excel_file_path", "")

    # Phase 1 – Display tables
    if df_env is not None:
        st.subheader("Environment Table (from Confluence)")
        if loaded_env_page_ids:
            st.caption(f"Source page IDs: {', '.join(loaded_env_page_ids)}")
        st.dataframe(df_env, use_container_width=True)

        csv_data = df_env.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download ENV as CSV",
            data=csv_data,
            file_name="confluence_env_table.csv",
            mime="text/csv",
        )
    
    if df_lb is not None:
        st.subheader("LoadBalancer Table (from Confluence)")
        if loaded_lb_page_ids:
            st.caption(f"Source page IDs: {', '.join(loaded_lb_page_ids)}")
        st.dataframe(df_lb, use_container_width=True)

        csv_data = df_lb.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download LB as CSV",
            data=csv_data,
            file_name="confluence_lb_table.csv",
            mime="text/csv",
        )
    
    if df_excel is not None:
        st.subheader("Excel Table (from local file)")
        if excel_file_path:
            st.caption(f"Source file: {excel_file_path}")
        st.dataframe(df_excel, use_container_width=True)

        csv_data = df_excel.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Excel as CSV",
            data=csv_data,
            file_name="excel_table.csv",
            mime="text/csv",
        )
    
    if df_env is None and df_lb is None and df_excel is None:
        st.info("Load tables from Confluence or Excel file using the sidebar buttons.")

    # Phase 2 – Text-to-SQL Chat
    st.markdown("---")
    st.subheader("💬 Ask a question (Text-to-SQL over combined DuckDB table)")

    st.markdown(
        """
Examples:
- `What is the SNAT IP of project PS-FEP-IES-Q?` (ENV table)
- `List all projects and SNAT IPs where customer is PS` (ENV table)
- `Show all LoadBalancer entries for cluster si0dcs09` (LB table)
- `What is the LB IP address for namespace bd-cros-comp02-d-ias-shared?` (LB table)
- `Show all data from excel_table` (Excel table)
- `Join data from ENV and Excel tables where cluster matches`
"""
    )

    user_q = st.chat_input("Ask a question about the tables...")

    if user_q:
        with st.chat_message("user"):
            st.markdown(user_q)

        if df_env is None and df_lb is None and df_excel is None:
            with st.chat_message("assistant"):
                st.markdown(
                    "I don't have any data yet. "
                    "Please load tables from Confluence or Excel file first."
                )
        else:
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Running Text-to-SQL chain (2 LLM calls + DuckDB)..."):
                        sql, result_df, answer = text_to_sql_chain(user_q, df_env, df_lb, df_excel)

                    # Final answer
                    st.markdown(f"**Answer:**\n\n{answer}")

                    # Debug / transparency
                    with st.expander("🧠 SQL generated by LLM"):
                        st.code(sql, language="sql")

                    if result_df is not None and not result_df.empty:
                        with st.expander("📊 Raw SQL result (DuckDB)"):
                            st.dataframe(result_df, use_container_width=True)
                    else:
                        with st.expander("📊 Raw SQL result (DuckDB)"):
                            st.write("No rows returned.")

                except Exception as e:
                    st.error(f"Error in Text-to-SQL chain: {e}")


if __name__ == "__main__":
    main()

