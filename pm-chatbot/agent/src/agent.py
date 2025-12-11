from textwrap import dedent
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.ag_ui import StateDeps
from ag_ui.core import EventType, StateSnapshotEvent
from pydantic_ai.models.openai import OpenAIResponsesModel

# load environment variables
from dotenv import load_dotenv
load_dotenv()

# =====
# State
# =====
class AgentState(BaseModel):
  """Agent state for memory and context."""
  conversation_history: list[str] = Field(
    default_factory=list,
    description='The conversation history for context',
  )

# =====
# Agent
# =====
agent = Agent(
  model = OpenAIResponsesModel('gpt-4.1-mini'),
  deps_type=StateDeps[AgentState],
  system_prompt=dedent("""
    You are a helpful PM (Project Management) assistant.
    
    You help with project management tasks, provide information, and assist with various queries.
    You maintain conversation context to provide better assistance.
  """).strip()
)

# =====
# Tools
# =====
# Add your custom tools here as needed
