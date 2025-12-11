"""
Utility functions for PM Assistant - Confluence API, HTML parsing, and data processing.

These functions are ported from the Streamlit version and maintain the exact logic
for fetching and parsing Confluence data.
"""
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import Optional


# Column mappings for SQL-friendly names
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


def fetch_confluence_html(base_url: str, page_id: str, token: str) -> str:
    """
    Call Confluence REST API and return body.storage.value (HTML string).
    
    Args:
        base_url: Confluence base URL
        page_id: Confluence page ID
        token: Bearer token for authentication
    
    Returns:
        HTML content of the page
    """
    url = f"{base_url}/rest/api/content/{page_id}"
    params = {"expand": "body.storage,version,space"}
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    html = data["body"]["storage"]["value"]
    return html


def html_table_to_dataframe(
    html: str, 
    wanted_columns: Optional[list[str]] = None, 
    table_type: str = "env"
) -> pd.DataFrame:
    """
    Parse Confluence HTML containing (possibly nested) tables into a pandas DataFrame.
    
    This function handles noisy HTML well by:
    - Using the FIRST top-level table it finds
    - Removing nested tables inside each cell
    - Extracting visible text only (no URLs)
    - Keeping only wanted_columns if provided
    
    Args:
        html: HTML string from Confluence
        wanted_columns: List of column names to keep (optional)
        table_type: "env" for environment table, "lb" for loadbalancer table
    
    Returns:
        DataFrame with parsed table data
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


def load_confluence_tables_combined(
    base_url: str,
    token: str,
    page_ids: list[str],
    table_type: str = "env",
) -> pd.DataFrame:
    """
    Load & combine multiple Confluence pages into a single dataframe.
    
    Args:
        base_url: Confluence base URL
        token: Bearer token
        page_ids: List of page IDs to load
        table_type: "env" or "lb"
    
    Returns:
        Combined DataFrame with '__source_page_id__' column
    """
    dfs = []
    for pid in page_ids:
        pid = str(pid).strip()
        if not pid:
            continue
        
        html = fetch_confluence_html(base_url, pid, token)
        df_single = html_table_to_dataframe(html, table_type=table_type)
        df_single = df_single.copy()
        df_single["__source_page_id__"] = pid
        dfs.append(df_single)

    if not dfs:
        raise ValueError("No valid page IDs provided.")

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def excel_columns_to_sql(df: pd.DataFrame) -> dict:
    """
    Convert Excel column names to SQL-friendly names (snake_case).
    
    Args:
        df: DataFrame with Excel columns
    
    Returns:
        Dictionary mapping original column names to SQL-friendly names
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
    Return a copy of df_raw with columns renamed to SQL-friendly names.
    
    Args:
        df_raw: Raw DataFrame from Confluence or Excel
        table_type: "env", "lb", or "excel"
    
    Returns:
        DataFrame with SQL-friendly column names
    """
    if table_type == "excel":
        COL_MAP = excel_columns_to_sql(df_raw)
        df_sql = df_raw.rename(columns=COL_MAP).copy()
    else:
        COL_MAP = COL_MAP_LB if table_type == "lb" else COL_MAP_ENV
        cols_present = [c for c in COL_MAP.keys() if c in df_raw.columns]
        df_sql = df_raw[cols_present].rename(columns=COL_MAP).copy()

    # Optionally keep source page information
    if "__source_page_id__" in df_raw.columns:
        df_sql["source_page_id"] = df_raw["__source_page_id__"].values

    return df_sql


def detect_firewall_request(question: str) -> Optional[dict]:
    """
    Detect if the question is a firewall clearance request.
    
    Args:
        question: User's question
    
    Returns:
        Dictionary with plant_name, use_case, environment or None
    """
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
    
    Args:
        plant_name: Plant name (e.g., "SzP", "BuE")
        use_case: Use case (e.g., "opbase")
        environment: Environment (P, Q, or D)
        df_env: Environment DataFrame
        df_lb: Load balancer DataFrame
        df_excel: Excel DataFrame with plant codes
    
    Returns:
        Dictionary with snat_ip, lb_ip (if opbase), cluster, project_name
    """
    result = {'snat_ip': None, 'lb_ip': None, 'cluster': None, 'project_name': None}
    
    # Step 1: Lookup plant code in Excel
    plant_code = None
    if df_excel is not None and not df_excel.empty:
        # Assume first column has format "code, name"
        first_col = df_excel.columns[0]
        for val in df_excel[first_col]:
            if pd.notna(val):
                val_str = str(val).upper()
                plant_name_upper = plant_name.upper()
                
                # Check if plant_name is contained in the cell value (after comma)
                # Format: "code, name" -> extract name part and check for partial match
                if ',' in val_str:
                    parts = val_str.split(',')
                    if len(parts) >= 2:
                        code_part = parts[0].strip()
                        name_part = parts[1].strip()
                        
                        # Check if plant_name matches the beginning of name_part
                        # e.g., "BUE" matches "BUEP"
                        if name_part.startswith(plant_name_upper) or plant_name_upper in name_part:
                            plant_code = code_part
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
