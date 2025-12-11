import os
from contextlib import asynccontextmanager
from typing import Optional
import duckdb
import pandas as pd
from fastapi import FastAPI

from agent import AgentState, StateDeps, agent
from data_context import PMDataContext
from utils import load_confluence_tables_combined, build_sql_dataframe

# Global data context (initialized on startup)
data_context: Optional[PMDataContext] = None


def create_data_context() -> PMDataContext:
    """
    Initialize PM data context - loads all data sources on startup.
    
    This function:
    1. Creates a DuckDB in-memory connection
    2. Loads Confluence ENV and LB tables
    3. Loads Excel file with plant codes
    4. Registers all tables in DuckDB
    5. Returns initialized PMDataContext
    """
    print("🚀 Initializing PM Assistant data context...")
    
    # Read configuration from environment variables
    base_url = os.getenv("CONFLUENCE_BASE_URL", "https://inside-docupedia.example.com/confluence")
    token = os.getenv("CONFLUENCE_TOKEN", "")
    
    env_page_ids_str = os.getenv("ENVPAGE_IDS", "")
    env_page_ids = [pid.strip() for pid in env_page_ids_str.split(",") if pid.strip()]
    
    lb_page_ids_str = os.getenv("LB_PAGE_IDS", "")
    lb_page_ids = [pid.strip() for pid in lb_page_ids_str.split(",") if pid.strip()]
    
    excel_file_path = os.getenv("EXCEL_FILE_PATH", "BWN.xlsx")
    
    # Create context with DuckDB connection
    ctx = PMDataContext(
        db_connection=duckdb.connect(database=":memory:"),
        confluence_base_url=base_url,
        confluence_token=token,
        env_page_ids=env_page_ids,
        lb_page_ids=lb_page_ids,
        excel_file_path=excel_file_path,
    )
    
    # Load Confluence ENV tables
    if token and env_page_ids:
        try:
            print(f"  📥 Loading ENV tables from {len(env_page_ids)} Confluence page(s)...")
            ctx.df_env = load_confluence_tables_combined(
                base_url, token, env_page_ids, table_type="env"
            )
            ctx.env_data_loaded = True
            print(f"  ✅ ENV table loaded: {len(ctx.df_env)} rows")
        except Exception as e:
            print(f"  ⚠️  ENV table loading failed: {e}")
    
    # Load Confluence LB tables
    if token and lb_page_ids:
        try:
            print(f"  📥 Loading LB tables from {len(lb_page_ids)} Confluence page(s)...")
            ctx.df_lb = load_confluence_tables_combined(
                base_url, token, lb_page_ids, table_type="lb"
            )
            ctx.lb_data_loaded = True
            print(f"  ✅ LB table loaded: {len(ctx.df_lb)} rows")
        except Exception as e:
            print(f"  ⚠️  LB table loading failed: {e}")
    
    # Load Excel file
    if os.path.exists(excel_file_path):
        try:
            print(f"  📥 Loading Excel file: {excel_file_path}...")
            ctx.df_excel = pd.read_excel(excel_file_path)
            ctx.excel_data_loaded = True
            print(f"  ✅ Excel loaded: {len(ctx.df_excel)} rows")
        except Exception as e:
            print(f"  ⚠️  Excel loading failed: {e}")
    else:
        print(f"  ⚠️  Excel file not found: {excel_file_path}")
    
    # Register tables in DuckDB
    print("  🗄️  Registering tables in DuckDB...")
    if ctx.df_env is not None:
        df_env_sql = build_sql_dataframe(ctx.df_env, "env")
        ctx.db_connection.register("env_table", df_env_sql)
        print(f"     ✓ env_table registered")
    
    if ctx.df_lb is not None:
        df_lb_sql = build_sql_dataframe(ctx.df_lb, "lb")
        ctx.db_connection.register("lb_table", df_lb_sql)
        print(f"     ✓ lb_table registered")
    
    if ctx.df_excel is not None:
        df_excel_sql = build_sql_dataframe(ctx.df_excel, "excel")
        ctx.db_connection.register("excel_table", df_excel_sql)
        print(f"     ✓ excel_table registered")
    
    print("✨ PM Assistant ready!")
    return ctx


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load data on startup, cleanup on shutdown."""
    global data_context
    
    # Startup: Initialize data
    data_context = create_data_context()
    
    yield
    
    # Shutdown: Cleanup
    if data_context and data_context.db_connection:
        data_context.db_connection.close()
        print("🔌 DuckDB connection closed")


# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Mount CopilotKit agent endpoint
# to_ag_ui returns an AGUIApp, we need to mount it properly
ag_ui_app = agent.to_ag_ui(deps=lambda: data_context)
app.mount("/api/copilotkit", ag_ui_app)


if __name__ == "__main__":
    # run the app
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
