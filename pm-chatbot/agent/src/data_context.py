"""
Data context for PM Assistant - holds database connection and loaded data.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
import duckdb
import pandas as pd


@dataclass
class PMDataContext:
    """
    Dependencies for PM Assistant - holds DuckDB connection and DataFrames.
    
    This context is initialized once at FastAPI startup and injected into
    the PydanticAI agent for each request.
    """
    
    # DuckDB connection (persistent during agent lifecycle)
    db_connection: duckdb.DuckDBPyConnection
    
    # DataFrames (loaded once at startup)
    df_env: Optional[pd.DataFrame] = None
    df_lb: Optional[pd.DataFrame] = None
    df_excel: Optional[pd.DataFrame] = None
    
    # Configuration from environment variables
    confluence_base_url: str = ""
    confluence_token: str = ""
    env_page_ids: list[str] = field(default_factory=list)
    lb_page_ids: list[str] = field(default_factory=list)
    excel_file_path: str = "BWN.xlsx"
    
    # Status flags
    env_data_loaded: bool = False
    lb_data_loaded: bool = False
    excel_data_loaded: bool = False
