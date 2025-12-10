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

# -----------------------------
# Firewall Clearance Request Processing
# -----------------------------

def detect_firewall_request(question: str) -> dict:
    """
    Detect if the question is a firewall clearance request.
    Returns dict with plant_name, use_case, environment or None.
    """
    import re
    
    question_lower = question.lower()
    
    # Pattern: firewall clearance for {plant}, use case {usecase}, environment {env}
    if "firewall" in question_lower and "clearance" in question_lower:
        result = {}
        
        # Extract plant name (after "for" and before "use case" or ",")
        plant_match = re.search(r'for\s+([\w-]+)', question_lower)
        if plant_match:
            result['plant_name'] = plant_match.group(1).upper()
        
        # Extract use case
        usecase_match = re.search(r'use\s*case\s+([\w-]+)', question_lower)
        if usecase_match:
            result['use_case'] = usecase_match.group(1).lower()
        
        # Extract environment
        env_match = re.search(r'environment\s+([pqd])', question_lower)
        if env_match:
            result['environment'] = env_match.group(1).upper()
        
        if 'plant_name' in result and 'use_case' in result and 'environment' in result:
            return result
    
    return None


def process_firewall_request(
    plant_name: str,
    use_case: str,
    environment: str,
    df_env: pd.DataFrame,
    df_lb: pd.DataFrame,
    df_excel: pd.DataFrame,
) -> dict:
    """
    Process firewall clearance request.
    Returns dict with snat_ip, lb_ip (if opbase), cluster, project_name.
    """
    result = {'snat_ip': None, 'lb_ip': None, 'cluster': None, 'project_name': None}
    
    # Step 1: Lookup plant code in Excel
    plant_code = None
    if df_excel is not None and not df_excel.empty:
        # Assume first column has format "code, name"
        first_col = df_excel.columns[0]
        for val in df_excel[first_col]:
            if pd.notna(val) and plant_name in str(val).upper():
                # Extract code (before comma)
                parts = str(val).split(',')
                if len(parts) >= 2:
                    plant_code = parts[0].strip()
                    break
    
    # Step 2: Search ENV table for project
    if df_env is not None and not df_env.empty:
        # Pattern: xxx-{plant_code|plant_name}-{use_case}-{environment}
        for idx, row in df_env.iterrows():
            project = str(row.get('Project', '')).lower()
            
            # Split project by dashes
            parts = project.split('-')
            if len(parts) >= 4:
                # Check if matches pattern: *-{plant}-{usecase}-{env}
                proj_plant = parts[-3]
                proj_usecase = parts[-2]
                proj_env = parts[-1]
                
                # Match plant (code preferred, then name)
                plant_match = False
                if plant_code and proj_plant == plant_code.lower():
                    plant_match = True
                elif proj_plant == plant_name.lower():
                    plant_match = True
                
                # Match use case and environment
                if plant_match and proj_usecase == use_case.lower() and proj_env == environment.lower():
                    result['project_name'] = row.get('Project', '')
                    result['snat_ip'] = row.get('SNAT-IP', '')
                    result['cluster'] = row.get('Cluster', '')
                    break
    
    # Step 3: If use_case is opbase, get LB IP from LB table
    # Search by exact project name match from ENV table
    if use_case.lower() == 'opbase' and result['project_name'] and df_lb is not None and not df_lb.empty:
        # Use the exact project name found in ENV table to search LB table
        project_to_find = result['project_name'].lower()
        
        for idx, row in df_lb.iterrows():
            lb_project = str(row.get('Project', '')).lower()
            
            # Exact match with the project found in ENV table
            if lb_project == project_to_find:
                result['lb_ip'] = row.get('LB-IP Address', '')
                break
    
    return result


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
    1) Check if it's a firewall clearance request
    2) Build SQL-friendly copies of dataframes.
    3) LLM: question -> SQL.
    4) Execute SQL on DuckDB.
    5) LLM: (question + sql + result) -> final answer.
    Returns (sql, result_df, final_answer).
    """
    if (df_env_raw is None or df_env_raw.empty) and (df_lb_raw is None or df_lb_raw.empty) and (df_excel_raw is None or df_excel_raw.empty):
        raise ValueError("No dataframes available; load data first.")

    # Check if this is a firewall clearance request
    fw_request = detect_firewall_request(question)
    
    if fw_request:
        # Process firewall clearance request
        fw_result = process_firewall_request(
            fw_request['plant_name'],
            fw_request['use_case'],
            fw_request['environment'],
            df_env_raw,
            df_lb_raw,
            df_excel_raw,
        )
        
        # Build firewall clearance table
        # For now, use placeholder destinations - in reality, these would come from user input or another source
        destinations = [
            {'ip': '10.11.12.13', 'protocol': 'TCP', 'port': '12345', 'remark': 'Mail server'},
            {'ip': '13.14.15.16', 'protocol': 'TCP', 'port': '80', 'remark': 'Trust Center'},
            {'ip': '1.2.3.4', 'protocol': 'TCP', 'port': '8011', 'remark': 'Oracle DB'},
        ]
        
        rows = []
        # Add SNAT-IP rows
        if fw_result['snat_ip']:
            for dest in destinations:
                rows.append({
                    'SOURCE': fw_result['snat_ip'],
                    'DESTINATION': dest['ip'],
                    'PROTOCOL': dest['protocol'],
                    'PORT': dest['port'],
                    'REMARK': dest['remark'],
                })
        
        # Add LB-IP rows (only for opbase)
        if fw_result['lb_ip']:
            for dest in destinations:
                rows.append({
                    'SOURCE': fw_result['lb_ip'],
                    'DESTINATION': dest['ip'],
                    'PROTOCOL': dest['protocol'],
                    'PORT': dest['port'],
                    'REMARK': dest['remark'],
                })
        
        result_df = pd.DataFrame(rows)
        
        # Generate answer
        if not result_df.empty:
            answer = (
                f"Firewall clearance request for **{fw_request['plant_name']}**, "
                f"use case **{fw_request['use_case']}**, environment **{fw_request['environment']}**\n\n"
                f"**Project found:** {fw_result['project_name']}\n"
                f"**SNAT-IP:** {fw_result['snat_ip']}\n"
            )
            if fw_result['lb_ip']:
                answer += f"**LB-IP:** {fw_result['lb_ip']} (opbase use case)\n"
            if fw_result['cluster']:
                answer += f"**Cluster:** {fw_result['cluster']}\n"
            
            answer += "\n**Firewall rules generated below.**"
        else:
            answer = (
                f"Could not find project matching: {fw_request['plant_name']}, "
                f"{fw_request['use_case']}, {fw_request['environment']}"
            )
        
        return "-- Firewall clearance request processed --", result_df, answer

    # Regular SQL query flow
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

    # Hide sidebar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("📄 Confluence Multi-Page Viewer + 🐤 DuckDB Text-to-SQL Chat")

    # Check DEV_MODE
    dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"

    st.markdown(
        """
**Phase 1 — Data Ingestion & Display**

- Loads **Environment** tables (ENVPAGE_IDS) and **LoadBalancer** tables (LB_PAGE_IDS) from Confluence.  
- Loads **Excel** file (BWN.xlsx) from local filesystem.  
- Cleans nested tables / links and keeps only key columns.  
- Adds a `__source_page_id__` column to Confluence data.  
- Caches the result for 5 minutes.  

**Phase 2 — Text-to-SQL Chat Chain**

- LLM Call 1: question → SQL over `env_table`, `lb_table`, and/or `excel_table` (DuckDB).  
- Execute SQL against in-memory DuckDB (all tables available).  
- LLM Call 2: question + result → natural language answer.
"""
    )

    # Auto-load data on first run
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = False

    # Read configuration from .env
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

    base_url = default_base_url
    token = default_token
    
    # Auto-load data on startup
    if not st.session_state["data_loaded"]:
        with st.spinner("Auto-loading data on startup..."):
            errors = []
            
            # Auto-load Excel file (BWN.xlsx)
            excel_file_path = "BWN.xlsx"
            if os.path.exists(excel_file_path):
                try:
                    df_excel = load_excel_file(excel_file_path)
                    st.session_state["excel_df"] = df_excel
                    st.session_state["excel_file_path"] = excel_file_path
                except Exception as e:
                    errors.append(f"Excel file error: {e}")
            else:
                errors.append(f"Excel file not found: {excel_file_path}")
            
            # Auto-load Confluence tables
            if token and (env_page_ids or lb_page_ids):
                # Load Environment tables
                if env_page_ids:
                    try:
                        df_env = cached_load_tables_combined(
                            base_url,
                            token,
                            tuple(env_page_ids),
                            table_type="env",
                        )
                        st.session_state["confluence_env_df"] = df_env
                        st.session_state["env_page_ids"] = env_page_ids
                    except Exception as e:
                        errors.append(f"ENV tables error: {e}")
                
                # Load LoadBalancer tables
                if lb_page_ids:
                    try:
                        df_lb = cached_load_tables_combined(
                            base_url,
                            token,
                            tuple(lb_page_ids),
                            table_type="lb",
                        )
                        st.session_state["confluence_lb_df"] = df_lb
                        st.session_state["lb_page_ids"] = lb_page_ids
                    except Exception as e:
                        errors.append(f"LB tables error: {e}")
            
            st.session_state["data_loaded"] = True
            
            if errors:
                for err in errors:
                    st.warning(err)
            else:
                st.success("✅ All data loaded successfully!")

    df_env = st.session_state.get("confluence_env_df")
    df_lb = st.session_state.get("confluence_lb_df")
    df_excel = st.session_state.get("excel_df")
    loaded_env_page_ids = st.session_state.get("env_page_ids", [])
    loaded_lb_page_ids = st.session_state.get("lb_page_ids", [])
    excel_file_path = st.session_state.get("excel_file_path", "")

    # Phase 1 – Display tables (only in DEV_MODE)
    if dev_mode:
        st.subheader("📊 Data Tables (DEV_MODE=true)")
        
        if df_env is not None:
            with st.expander("Environment Table (from Confluence)", expanded=False):
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
            with st.expander("LoadBalancer Table (from Confluence)", expanded=False):
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
            with st.expander("Excel Table (BWN.xlsx)", expanded=False):
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
            st.info("No data loaded yet. Check sidebar for loading status.")
    else:
        # Production mode - just show status
        if df_env is not None or df_lb is not None or df_excel is not None:
            tables_loaded = []
            if df_env is not None:
                tables_loaded.append(f"ENV ({len(df_env)} rows)")
            if df_lb is not None:
                tables_loaded.append(f"LB ({len(df_lb)} rows)")
            if df_excel is not None:
                tables_loaded.append(f"Excel ({len(df_excel)} rows)")
            st.info(f"✅ Data loaded: {', '.join(tables_loaded)}. Ask questions below!")
        else:
            st.warning("⚠️ No data loaded. Check sidebar configuration.")

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
- `I want to create firewall clearance for SzP, use case opbase, environment P` (Firewall request)
"""
    )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display chat history
    for chat in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        
        with st.chat_message("assistant"):
            st.markdown(f"**Answer:**\n\n{chat['answer']}")
            
            with st.expander("🧠 SQL generated by LLM"):
                st.code(chat["sql"], language="sql")
            
            if chat["result_df"] is not None and not chat["result_df"].empty:
                with st.expander("📊 Raw SQL result (DuckDB)"):
                    st.dataframe(chat["result_df"], use_container_width=True)
            else:
                with st.expander("📊 Raw SQL result (DuckDB)"):
                    st.write("No rows returned.")

    user_q = st.chat_input("Ask a question about the tables...")

    if user_q:
        with st.chat_message("user"):
            st.markdown(user_q)

        if df_env is None and df_lb is None and df_excel is None:
            with st.chat_message("assistant"):
                error_msg = "I don't have any data yet. Please load tables from Confluence or Excel file first."
                st.markdown(error_msg)
                # Store error in history
                st.session_state["chat_history"].append({
                    "question": user_q,
                    "answer": error_msg,
                    "sql": "",
                    "result_df": None,
                })
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

                    # Store in chat history
                    st.session_state["chat_history"].append({
                        "question": user_q,
                        "answer": answer,
                        "sql": sql,
                        "result_df": result_df.copy() if result_df is not None else None,
                    })

                except Exception as e:
                    error_msg = f"Error in Text-to-SQL chain: {e}"
                    st.error(error_msg)
                    # Store error in history
                    st.session_state["chat_history"].append({
                        "question": user_q,
                        "answer": error_msg,
                        "sql": "",
                        "result_df": None,
                    })


if __name__ == "__main__":
    main()

