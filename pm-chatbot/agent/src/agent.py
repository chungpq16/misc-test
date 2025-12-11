import os
from textwrap import dedent
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps
from ag_ui.core import EventType, StateSnapshotEvent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import data context and utilities
from data_context import PMDataContext
from utils import process_firewall_request

# Configure LLM model based on environment variable
def get_model():
    """Get the configured LLM model (OpenAI or Ollama)."""
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    
    if use_ollama:
        # Ollama configuration
        # For OllamaProvider with OpenAI-compatible endpoint, base_url must include /v1
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # Ensure base_url ends with /v1 for OpenAI compatibility
        if not base_url.endswith('/v1'):
            base_url = f"{base_url}/v1"
        
        model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        api_key = os.getenv("OLLAMA_API_KEY", "not-required")  # Ollama local doesn't need API key
        
        print(f"🦙 Using Ollama model: {model_name} at {base_url}")
        return OpenAIChatModel(
            model_name=model_name,
            provider=OllamaProvider(base_url=base_url, api_key=api_key)
        )
    else:
        # OpenAI configuration (default)
        print("🤖 Using OpenAI model: gpt-4.1-mini")
        return OpenAIResponsesModel('gpt-4.1-mini')

# =====
# State
# =====
class AgentState(BaseModel):
  """Agent state for PM Assistant - tracks conversation and context."""
  conversation_history: list[str] = Field(
    default_factory=list,
    description='The conversation history for context',
  )
  
  # Data loading status (for informing user)
  env_data_loaded: bool = Field(default=False, description='Whether ENV data is loaded')
  lb_data_loaded: bool = Field(default=False, description='Whether LB data is loaded')
  excel_data_loaded: bool = Field(default=False, description='Whether Excel data is loaded')
  
  # Last query context
  last_sql_query: Optional[str] = Field(default=None, description='Last SQL query executed')
  last_result_count: int = Field(default=0, description='Number of rows in last result')
  
  # Firewall clearance context
  current_firewall_request: Optional[dict] = Field(
    default=None, 
    description='Current firewall request details (plant_name, use_case, environment)'
  )

# =====
# Agent  
# =====
agent = Agent(
  model = get_model(),
  deps_type=PMDataContext,
  system_prompt=dedent("""
    You are a helpful PM (Project Management) assistant with access to project databases.
    
    You have access to three databases:
    - env_table: Project environments (SNAT-IP, clusters, security zones)
    - lb_table: Load balancer configurations (LB-IP addresses, ports, namespaces)
    - excel_table: Plant codes and names mapping
    
    Use the available tools to help users with:
    1. Project information queries (SNAT IPs, clusters, customers, etc.)
    2. Load balancer information
    3. Firewall clearance generation
    
    For regular queries, use the query_database tool to execute SQL.
    For firewall clearance requests, use the generate_firewall_rules tool.
    
    Be concise and helpful. Format answers clearly with key information highlighted.
  """).strip()
)

# =====
# Tools
# =====

@agent.tool
async def query_database(
    ctx: RunContext[PMDataContext],
    sql_query: str
) -> dict:
    """
    Execute a SQL query against the DuckDB tables (env_table, lb_table, excel_table).
    
    Use this tool to query project information, load balancer configs, or plant codes.
    
    Args:
        sql_query: The SQL query to execute (use DuckDB syntax)
    
    Returns:
        Dictionary with 'rows' (list of dicts), 'count', 'columns', or 'error'
    """
    try:
        result_df = ctx.deps.db_connection.execute(sql_query).df()
        
        return {
            "rows": result_df.to_dict(orient="records"),
            "count": len(result_df),
            "columns": result_df.columns.tolist(),
            "success": True
        }
    except Exception as e:
        return {
            "error": str(e),
            "rows": [],
            "count": 0,
            "success": False
        }


@agent.tool
async def generate_firewall_rules(
    ctx: RunContext[PMDataContext],
    plant_name: str,
    use_case: str,
    environment: str,
    destination_ips: Optional[list[dict]] = None
) -> dict:
    """
    Generate firewall clearance rules for a specific plant, use case, and environment.
    
    Use this tool when user asks for firewall clearance/rules generation.
    
    Args:
        plant_name: Plant name (e.g., "SzP", "BuE")
        use_case: Use case (e.g., "opbase")
        environment: Environment - P (production), Q (quality), or D (development)
        destination_ips: Optional list of destinations with 'ip', 'protocol', 'port', 'remark'
    
    Returns:
        Dictionary with firewall rules and project information
    """
    # Get data from context
    df_env = ctx.deps.df_env
    df_lb = ctx.deps.df_lb
    df_excel = ctx.deps.df_excel
    
    # Process firewall request using utility function
    fw_result = process_firewall_request(
        plant_name,
        use_case,
        environment,
        df_env,
        df_lb,
        df_excel,
    )
    
    # Default destinations if not provided
    if destination_ips is None:
        destination_ips = [
            {'ip': '10.11.12.13', 'protocol': 'TCP', 'port': '12345', 'remark': 'Mail server'},
            {'ip': '13.14.15.16', 'protocol': 'TCP', 'port': '80', 'remark': 'Trust Center'},
            {'ip': '1.2.3.4', 'protocol': 'TCP', 'port': '8011', 'remark': 'Oracle DB'},
        ]
    
    # Build firewall rules
    rules = []
    if fw_result['snat_ip']:
        for dest in destination_ips:
            rules.append({
                'SOURCE': fw_result['snat_ip'],
                'DESTINATION': dest['ip'],
                'PROTOCOL': dest['protocol'],
                'PORT': dest['port'],
                'REMARK': dest['remark'],
            })
    
    if fw_result['lb_ip']:  # Only for opbase use case
        for dest in destination_ips:
            rules.append({
                'SOURCE': fw_result['lb_ip'],
                'DESTINATION': dest['ip'],
                'PROTOCOL': dest['protocol'],
                'PORT': dest['port'],
                'REMARK': dest['remark'],
            })
    
    return {
        "project_name": fw_result['project_name'],
        "snat_ip": fw_result['snat_ip'],
        "lb_ip": fw_result['lb_ip'],
        "cluster": fw_result['cluster'],
        "rules": rules,
        "rule_count": len(rules),
        "success": fw_result['project_name'] is not None
    }


@agent.tool
async def search_project_info(
    ctx: RunContext[PMDataContext],
    project_name: Optional[str] = None,
    customer: Optional[str] = None,
    cluster: Optional[str] = None
) -> dict:
    """
    Search for project information by project name, customer, or cluster.
    
    Use this tool for quick project lookups without writing SQL.
    
    Args:
        project_name: Project name (partial match supported)
        customer: Customer name (partial match supported)
        cluster: Cluster name (partial match supported)
    
    Returns:
        Dictionary with matching projects
    """
    conditions = []
    if project_name:
        conditions.append(f"LOWER(project) LIKE LOWER('%{project_name}%')")
    if customer:
        conditions.append(f"LOWER(customer) LIKE LOWER('%{customer}%')")
    if cluster:
        conditions.append(f"LOWER(cluster) LIKE LOWER('%{cluster}%')")
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    sql = f"SELECT * FROM env_table WHERE {where_clause} LIMIT 50"
    
    try:
        result_df = ctx.deps.db_connection.execute(sql).df()
        
        return {
            "projects": result_df.to_dict(orient="records"),
            "count": len(result_df),
            "success": True
        }
    except Exception as e:
        return {
            "error": str(e),
            "projects": [],
            "count": 0,
            "success": False
        }
