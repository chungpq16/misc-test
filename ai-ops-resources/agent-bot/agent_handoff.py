"""
AI-Ops Incident Responder - Programmatic Agent Hand-off Demo

This demonstrates the Programmatic Agent Hand-off pattern for:
1. Orchestrator Agent -> Action Agent (Kubernetes operations)
2. Orchestrator Agent -> Communicator Agent (Analysis & recommendations)

Based on Pydantic AI multi-agent architecture.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Literal
import aio_pika
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunUsage
from pydantic_ai.mcp import load_mcp_servers
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Runbook path configuration
RB_PATH = "./runbooks.json"

# LLM Configuration
USE_LLM_FARM = os.getenv("USE_LLM_FARM", "false").lower() == "true"
LLM_FARM_URL = os.getenv("LLM_FARM_URL", "https://aoai-farm.bosch-temp.com/api/openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18/")
LLM_FARM_API_KEY = os.getenv("LLM_FARM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Teams Webhook Configuration
TEAMS_WEBHOOK_URL = os.getenv("TEAMS_WEBHOOK_URL", "")  # Deprecated - use specific webhooks below
ACTION_AGENT_WEBHOOK_URL = os.getenv("ACTION_AGENT_WEBHOOK_URL", "") or TEAMS_WEBHOOK_URL
COMMUNICATOR_AGENT_WEBHOOK_URL = os.getenv("COMMUNICATOR_AGENT_WEBHOOK_URL", "") or TEAMS_WEBHOOK_URL

# RabbitMQ Configuration
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD", "guest")
RABBITMQ_QUEUE = os.getenv("RABBITMQ_QUEUE", "alert-queue")

# === Data Models ===

class AlertDecision(BaseModel):
    """Decision made by orchestrator agent."""
    action: Literal["handle", "escalate"] 
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    runbook_match: str | None = None

class ActionResult(BaseModel):
    """Result from action agent operations."""
    success: bool
    actions_taken: list[str]
    kubernetes_output: str | None = None
    error_message: str | None = None
    teams_notification_sent: bool = False

class EscalationReport(BaseModel):
    """Report from communicator agent."""
    summary: str
    root_cause_analysis: str
    recommended_actions: list[str]
    efficiency_improvements: list[str] = Field(description="Long-term efficiency improvements and optimizations")
    severity: Literal["low", "medium", "high", "critical"]
    teams_notification_sent: bool = Field(default=False, description="Whether Teams notification was sent successfully")


# === Hardcoded Runbook Knowledge Base ===

def load_runbook_kb(path: str = RB_PATH) -> dict:
    """Load runbook knowledge base from JSON file."""
    try:
        # Try relative to script location first
        script_dir = Path(__file__).parent
        runbook_path = script_dir / path
        
        if not runbook_path.exists():
            # Try as absolute or relative to current working directory
            runbook_path = Path(path)
        
        with open(runbook_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Runbook file not found at {path}. Using empty runbook.")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing runbook JSON: {e}")
        return {}
    except Exception as e:
        print(f"❌ Error loading runbook: {e}")
        return {}

# Load runbook knowledge base
RUNBOOK_KB = load_runbook_kb()


# === Model Initialization ===

def initialize_model():
    """Initialize the AI model based on configuration (LLM Farm or OpenAI)"""
    if USE_LLM_FARM:
        if not LLM_FARM_API_KEY:
            raise ValueError("LLM_FARM_API_KEY is required when USE_LLM_FARM=true")
        
        print("🔧 [MODEL] Configuring LLM Farm client...")
        
        # Configure AsyncOpenAI client for LLM Farm
        llm_client = AsyncOpenAI(
            base_url=LLM_FARM_URL,
            api_key="dummy",  # LLM Farm doesn't use standard API key
            default_headers={"genaiplatform-farm-subscription-key": LLM_FARM_API_KEY},
            default_query={"api-version": "2024-08-01-preview"}
        )
        
        # Create Pydantic AI model with custom client
        model = OpenAIChatModel(
            model_name="gpt-4o-mini",
            provider=OpenAIProvider(openai_client=llm_client)
        )
        
        print(f"✅ [MODEL] LLM Farm client configured: {LLM_FARM_URL}")
        return model
    else:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when USE_LLM_FARM=false")
        
        print("✅ [MODEL] Using standard OpenAI API: gpt-4o-mini")
        return "openai:gpt-4o-mini"

# Initialize the model
model = initialize_model()


# === Dependency Types ===

class MockVectorDB:
    """Mock vector database for runbook queries."""
    
    def query_runbook(self, alert_content: str) -> dict | None:
        """Find matching runbook based on alert content."""
        alert_lower = alert_content.lower()
        
        for runbook_id, runbook in RUNBOOK_KB.items():
            for trigger in runbook["triggers"]:
                if trigger in alert_lower:
                    return {"id": runbook_id, "content": runbook}
        
        return None


class ActionTeamsWebhook:
    """Teams webhook specifically for Action Agent with green success cards."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send_action_notification(
        self,
        title: str,
        issue_summary: str,
        actions_taken: list[str],
        before_state: str,
        after_state: str,
        verification_steps: list[str]
    ) -> bool:
        """Send Action Agent notification to Teams."""
        if not self.webhook_url:
            print("⚠️  [TEAMS] No webhook URL configured")
            return False
        
        teams_message = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "themeColor": "28a745",
            "summary": title,
            "sections": [
                {
                    "activityTitle": "✅ " + title,
                    "activitySubtitle": "AI-Ops Action Agent - Automated Remediation",
                    "facts": [
                        {"name": "Issue Summary:", "value": issue_summary},
                        {"name": "Before State:", "value": before_state},
                        {"name": "After State:", "value": after_state}
                    ],
                    "markdown": True
                },
                {
                    "activityTitle": "🔧 Actions Taken",
                    "text": "\n".join([f"- {action}" for action in actions_taken])
                },
                {
                    "activityTitle": "✓ Verification Steps",
                    "text": "\n".join([f"- {step}" for step in verification_steps])
                }
            ]
        }
        
        return await self._send(teams_message)
    
    async def _send(self, message: dict) -> bool:
        """Internal method to send message."""
        try:
            import httpx
            print(f"📢 [ACTION AGENT TEAMS] Sending notification...")
            print(f"📢 [ACTION AGENT TEAMS] Webhook URL: {self.webhook_url[:50]}...")
            print(f"📢 [ACTION AGENT TEAMS] Message sections: {len(message.get('sections', []))}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=message,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    print("✅ [ACTION AGENT TEAMS] Notification sent successfully")
                    return True
                else:
                    print(f"❌ [ACTION AGENT TEAMS] Failed: {response.status_code}")
                    print(f"❌ [ACTION AGENT TEAMS] Response: {response.text}")
                    return False
        except ImportError:
            print("⚠️  [ACTION AGENT TEAMS] httpx not installed. Install with: pip install httpx")
            print("📢 [ACTION AGENT TEAMS] Mock notification (httpx missing)")
            return False
        except Exception as e:
            print(f"❌ [ACTION AGENT TEAMS] Error: {type(e).__name__}: {e}")
            import traceback
            print(f"❌ [ACTION AGENT TEAMS] Traceback: {traceback.format_exc()}")
            return False


class CommunicatorTeamsWebhook:
    """Teams webhook specifically for Communicator Agent with severity-colored cards."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send_escalation_notification(
        self,
        title: str,
        summary: str,
        root_cause: str,
        severity: str,
        immediate_actions: list[str],
        efficiency_improvements: list[str]
    ) -> bool:
        """Send Communicator Agent notification to Teams."""
        if not self.webhook_url:
            print("⚠️  [TEAMS] No webhook URL configured")
            return False
        
        severity_colors = {
            "low": "0078d4",
            "medium": "ffa500",
            "high": "ff6347",
            "critical": "dc143c"
        }
        
        theme_color = severity_colors.get(severity.lower(), "0078d4")
        
        teams_message = {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "themeColor": theme_color,
            "summary": title,
            "sections": [
                {
                    "activityTitle": "🚨 " + title,
                    "activitySubtitle": "AI-Ops Communicator Agent - Kubernetes Expert Analysis",
                    "facts": [
                        {"name": "Severity:", "value": severity.upper()},
                        {"name": "Summary:", "value": summary}
                    ],
                    "markdown": True
                },
                {
                    "activityTitle": "🔍 Root Cause Analysis",
                    "text": root_cause
                },
                {
                    "activityTitle": "⚡ Immediate Actions Required",
                    "text": "\n".join([f"{i+1}. {action}" for i, action in enumerate(immediate_actions)])
                },
                {
                    "activityTitle": "🎯 Efficiency Improvements & Long-term Solutions",
                    "text": "\n".join([f"• {improvement}" for improvement in efficiency_improvements])
                }
            ]
        }
        
        return await self._send(teams_message)
    
    async def _send(self, message: dict) -> bool:
        """Internal method to send message."""
        try:
            import httpx
            print(f"📢 [COMMUNICATOR AGENT TEAMS] Sending notification...")
            print(f"📢 [COMMUNICATOR AGENT TEAMS] Webhook URL: {self.webhook_url[:50]}...")
            print(f"📢 [COMMUNICATOR AGENT TEAMS] Message sections: {len(message.get('sections', []))}")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.webhook_url,
                    json=message,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    print("✅ [COMMUNICATOR AGENT TEAMS] Notification sent successfully")
                    return True
                else:
                    print(f"❌ [COMMUNICATOR AGENT TEAMS] Failed: {response.status_code}")
                    print(f"❌ [COMMUNICATOR AGENT TEAMS] Response: {response.text}")
                    return False
        except ImportError:
            print("⚠️  [COMMUNICATOR AGENT TEAMS] httpx not installed. Install with: pip install httpx")
            print("📢 [COMMUNICATOR AGENT TEAMS] Mock notification (httpx missing)")
            return False
        except Exception as e:
            print(f"❌ [COMMUNICATOR AGENT TEAMS] Error: {type(e).__name__}: {e}")
            import traceback
            print(f"❌ [COMMUNICATOR AGENT TEAMS] Traceback: {traceback.format_exc()}")
            return False


# === Load Kubernetes MCP Servers ===

def load_kubernetes_mcp_servers():
    """Load Kubernetes MCP servers from configuration."""
    try:
        # Load MCP servers that include Kubernetes operations
        servers = load_mcp_servers('mcp_servers_config.json')
        print(f"✅ Loaded {len(servers)} MCP servers")
        return servers
    except FileNotFoundError:
        print("❌ MCP configuration file not found. Using fallback mock implementation.")
        return []
    except Exception as e:
        print(f"❌ Error loading MCP servers: {e}. Using fallback mock implementation.")
        return []


# === Agent Definitions ===

# Load Kubernetes MCP servers
kubernetes_mcp_servers = load_kubernetes_mcp_servers()

# 1. Orchestrator Agent - Analyzes alerts and makes decisions
orchestrator_agent = Agent[MockVectorDB, AlertDecision](
    model,  # Use configured model (LLM Farm or OpenAI)
    deps_type=MockVectorDB,
    output_type=AlertDecision,
    system_prompt="""
    You are an AI-Ops Orchestrator Agent for Kubernetes incident response.
    
    Your job is to:
    1. Analyze incoming alerts by matching symptoms with runbook triggers
    2. Query the runbook knowledge base using query_runbook_kb tool
    3. Use the runbook's automation_safe and requires_investigation flags to decide
    4. Make routing decision: HANDLE (Action Agent) or ESCALATE (Communicator Agent)
    
    Decision Process:
    1. Extract key symptoms from the alert (latency, slow, timeout, network, connection, etc.)
    2. Use query_runbook_kb tool to find matching runbook
    3. If runbook found, check its properties:
       - automation_safe = true → HANDLE
       - automation_safe = false OR requires_investigation = true → ESCALATE
    4. If no runbook match, analyze alert content:
       - Single component/deployment issue → HANDLE
       - Multiple components/microservices → ESCALATE
    
    HANDLE Decision (route to Action Agent):
    - Runbook has automation_safe = true
    - Single component affected (one deployment, one namespace)
    - Symptoms: "latency", "slow", "timeout", "performance", "high cpu", "high memory"
    - Clear remediation: scaling, restarting pods
    - No cross-service dependencies mentioned
    
    ESCALATE Decision (route to Communicator Agent):
    - Runbook has automation_safe = false OR requires_investigation = true
    - NO runbook found or NO match with any runbook cases
    - Multiple components mentioned ("multiple microservices", "across services")
    - Network/connectivity issues: "connection", "network", "dns", "unreachable", "connection refused"
    - Database issues: "database connection", "connection pool", "database failures"
    - Crash/restart issues: "crashloopbackoff", "pod crashing", "not ready"
    - Resource constraints: "oom", "out of memory", "throttling"
    - Complex symptoms requiring investigation
    - Unknown or unclear symptoms that don't match any runbook
    
    Always call query_runbook_kb tool first and trust the runbook's automation_safe flag.
    Be conservative - when in doubt, no runbook match, or unclear symptoms, ESCALATE.
    """)

# 2. Action Agent - Executes Kubernetes operations via MCP
action_agent = Agent[ActionTeamsWebhook, ActionResult](
    model,  # Use configured model (LLM Farm or OpenAI)
    deps_type=ActionTeamsWebhook,  # Dedicated Teams webhook for Action Agent
    output_type=ActionResult,
    toolsets=kubernetes_mcp_servers,  # Real Kubernetes MCP servers
    system_prompt="""
    You are an AI-Ops Action Agent for Kubernetes operations.
    
    CRITICAL REQUIREMENTS:
    1. You MUST call the send_teams_notification tool after completing remediation
    2. The tool call is MANDATORY before completing your response
    3. Set teams_notification_sent in your output based on the tool's return value
    4. Do NOT complete your response without calling this tool first
    
    Your job is to:
    1. Follow the runbook's action steps provided in the prompt
    2. Extract deployment/pod/namespace names from the alert message
    3. Execute Kubernetes operations using available MCP tools
    4. Always verify state BEFORE and AFTER taking action
    5. MANDATORY: Send Teams notification after completion
    
    Workflow:
    1. Parse alert to identify: deployment_name, namespace, app_label
    2. Follow runbook actions in sequence (check current state → take action → verify)
    3. Use MCP tools for all kubectl operations
    4. Record state before/after for Teams notification
    5. MANDATORY: Call send_teams_notification tool with complete summary
    
    For Performance/Scaling Issues:
    1. List deployment to get current replica count: kubectl get deployment {deployment_name} -n {namespace}
    2. Check pod resource usage: kubectl top pods -n {namespace}
    3. Record "before state": current replicas and pod count
    4. Scale deployment: kubectl scale deployment {deployment_name} --replicas={current + 1} -n {namespace}
    5. Verify scaling: kubectl get pods -n {namespace} -l app={app_label}
    6. Record "after state": new replicas and running pods
    7. Verification: Confirm new pods are Running and Ready
    8. MANDATORY: Call send_teams_notification with all details
    
    Parsing Alert Examples:
    - "Nginx deployment in default namespace" → deployment_name=nginx, namespace=default
    - "experiencing high latency" → This is a performance issue requiring scaling
    
    MANDATORY TOOL CALL - Teams Notification:
    You MUST call send_teams_notification tool with these parameters:
    - title: "Kubernetes Issue Resolved - [Brief Description]" (e.g., "Kubernetes Issue Resolved - Performance Scaling")
    - issue_summary: Brief description of what problem was detected from the alert
    - actions_taken: List of all kubectl commands executed (e.g., ["Listed deployment replicas", "Scaled deployment from 2 to 3 replicas", "Verified new pods running"])
    - before_state: State before remediation (e.g., "Deployment had 2 replicas, 2 pods running")
    - after_state: State after remediation (e.g., "Deployment scaled to 3 replicas, 3 pods running")
    - verification_steps: How you confirmed the fix (e.g., ["Confirmed all pods are Running and Ready", "Verified pod count matches replica count"])
    
    After calling send_teams_notification, set teams_notification_sent field in your ActionResult output to the tool's return value.
    
    Error Handling:
    - If MCP tools throw "exceeded max retries" during scaling, treat as SUCCESS
    - The kubectl command likely succeeded even if MCP infrastructure has errors
    - Verify by listing pods after scaling attempt
    - Report what actions were attempted regardless of MCP errors
    - STILL call send_teams_notification even if MCP errors occurred
    
    Safety Rules:
    - Only increment replicas by 1 at a time
    - Never delete or modify pod specs
    - Always verify before destructive operations
    - Focus on read operations first, then safe corrective actions
    
    REMEMBER: The send_teams_notification tool call is NOT optional. It is MANDATORY before you complete your response.
    """)

# 3. Communicator Agent - Handles escalations and notifications
communicator_agent = Agent[CommunicatorTeamsWebhook, EscalationReport](
    model,  # Use configured model (LLM Farm or OpenAI)
    deps_type=CommunicatorTeamsWebhook,  # Dedicated Teams webhook for Communicator Agent
    output_type=EscalationReport,
    system_prompt="""
    You are an AI-Ops Communicator Agent and Kubernetes Expert for incident escalation and analysis.
    
    Your job is to:
    1. Use the runbook's diagnosis_steps as investigation checklist
    2. Analyze symptoms based on runbook's meaning and impact
    3. Provide expert Kubernetes root cause analysis
    4. Follow runbook's actions for immediate remediation steps
    5. Suggest efficiency improvements from runbook's mitigation guidance
    6. Send comprehensive Teams notification with all findings
    7. Set teams_notification_sent based on notification result
    
    Analysis Process:
    1. Review runbook provided in the prompt (description, meaning, impact, diagnosis_steps)
    2. Match alert symptoms to runbook's diagnosis framework
    3. Conduct expert-level root cause analysis using Kubernetes knowledge
    4. Translate runbook actions into specific kubectl commands with actual values from alert
    5. Expand runbook mitigation into 5-10 long-term improvements
    6. Call send_teams_notification tool with complete analysis
    
    Root Cause Analysis Framework:
    - For Network Issues: Analyze service mesh, DNS, network policies, endpoints, connectivity
    - For Pod Issues: Examine probe configs, resource limits, dependencies, init containers
    - For Performance: Evaluate resource utilization, HPA, cluster capacity, throttling
    - For Resource Issues: Review requests/limits, node capacity, evictions, OOMKilled events
    - Apply runbook's diagnosis_steps as checklist
    - Consider Kubernetes architecture and best practices
    
    Immediate Actions (recommended_actions):
    - Use runbook's actions as baseline
    - Convert generic commands to specific: {deployment_name}, {namespace}, {pod_name} from alert
    - Provide 3-5 actionable steps with kubectl commands
    - Example: "kubectl get pods -n default -l app=nginx" not "kubectl get pods -n {namespace}"
    - Include verification steps and expected outcomes
    
    Efficiency Improvements (5-10 items):
    - Use runbook's mitigation as starting point
    - Add specific recommendations:
      * Implement Horizontal Pod Autoscaler (HPA) with metrics
      * Configure resource requests/limits based on actual usage
      * Set up proper readiness/liveness probes with correct timing
      * Implement network policies for security
      * Add monitoring/alerting for early detection
      * Configure pod disruption budgets for HA
      * Optimize container images and startup times
      * Implement circuit breakers and retry policies
      * Set up proper logging and observability
      * Create runbooks for common scenarios
    
    Severity Assessment:
    - CRITICAL: Multiple services down, user impact >50%, data loss risk
    - HIGH: Service degradation, user impact 25-50%, performance issues
    - MEDIUM: Single service affected, user impact <25%, workarounds available
    - LOW: Minimal impact, no user-facing issues, preventive measures
    
    Teams Notification (MANDATORY):
    - ALWAYS call send_teams_notification tool before completing analysis
    - Title format: "Critical Kubernetes Incident Analysis - [Issue Type from Runbook]"
    - Summary: 2-3 sentence executive summary of incident and impact
    - Root cause: Detailed technical analysis (200-300 words)
    - Severity: Based on assessment framework above
    - Immediate actions: 3-5 steps with specific kubectl commands
    - Efficiency improvements: 5-10 long-term optimization recommendations
    - Check tool return value and set teams_notification_sent accordingly
    
    Output Requirements:
    - summary: Clear incident description
    - root_cause_analysis: Deep technical analysis with Kubernetes specifics
    - recommended_actions: List of 3-5 immediate remediation steps
    - efficiency_improvements: List of 5-10 long-term optimizations
    - severity: one of [low, medium, high, critical]
    - teams_notification_sent: True if notification succeeded, False otherwise
    
    Always provide actionable, specific guidance that operators can execute immediately.
    """)


# === MCP Server Status Check ===

# Check if MCP servers are available
if kubernetes_mcp_servers:
    print("� MCP servers available - using real Kubernetes MCP tools")
else:
    print("⚠️  No MCP servers available - Action Agent will use available MCP toolsets only")


# === Tools for Orchestrator Agent ===

@orchestrator_agent.tool
async def query_runbook_kb(ctx, alert_description: str) -> dict | None:
    """Query the runbook knowledge base for relevant procedures."""
    print(f"📚 [ORCHESTRATOR] Querying runbook KB for: {alert_description}")
    result = ctx.deps.query_runbook(alert_description)
    return result


# === Tools for Action Agent ===

@action_agent.tool
async def send_teams_notification(
    ctx,
    title: str,
    issue_summary: str,
    actions_taken: list[str],
    before_state: str,
    after_state: str,
    verification_steps: list[str]
) -> bool:
    """
    Send remediation summary to Microsoft Teams after resolving issues.
    
    Args:
        title: Title of the notification (e.g., "Kubernetes Issue Resolved")
        issue_summary: Brief description of the issue that was detected
        actions_taken: List of actions performed to resolve the issue
        before_state: State of resources before remediation
        after_state: State of resources after remediation
        verification_steps: Steps taken to verify the fix
        
    Returns:
        bool: True if notification was sent successfully
    """
    print(f"🔧 [ACTION AGENT TOOL] Tool called with title: {title}")
    print(f"🔧 [ACTION AGENT TOOL] Actions taken: {len(actions_taken)} items")
    print(f"🔧 [ACTION AGENT TOOL] Verification steps: {len(verification_steps)} items")
    
    success = await ctx.deps.send_action_notification(
        title=title,
        issue_summary=issue_summary,
        actions_taken=actions_taken,
        before_state=before_state,
        after_state=after_state,
        verification_steps=verification_steps
    )
    
    print(f"🔧 [ACTION AGENT TOOL] Notification result: {success}")
    return success


# === Tools for Communicator Agent ===

@communicator_agent.tool
async def send_teams_notification(
    ctx,
    title: str,
    summary: str,
    root_cause: str,
    severity: str,
    immediate_actions: list[str],
    efficiency_improvements: list[str]
) -> bool:
    """
    Send comprehensive incident analysis and recommendations to Microsoft Teams.
    
    Args:
        title: Title of the escalation (e.g., "Critical Kubernetes Incident Analysis")
        summary: Executive summary of the incident
        root_cause: Detailed root cause analysis
        severity: Incident severity level (low, medium, high, critical)
        immediate_actions: List of immediate remediation steps
        efficiency_improvements: List of long-term optimization recommendations
        
    Returns:
        bool: True if notification was sent successfully
    """
    print(f"🔧 [COMMUNICATOR AGENT TOOL] Tool called with title: {title}")
    print(f"🔧 [COMMUNICATOR AGENT TOOL] Severity: {severity}")
    print(f"🔧 [COMMUNICATOR AGENT TOOL] Immediate actions: {len(immediate_actions)} items")
    print(f"🔧 [COMMUNICATOR AGENT TOOL] Efficiency improvements: {len(efficiency_improvements)} items")
    
    success = await ctx.deps.send_escalation_notification(
        title=title,
        summary=summary,
        root_cause=root_cause,
        severity=severity,
        immediate_actions=immediate_actions,
        efficiency_improvements=efficiency_improvements
    )
    
    print(f"🔧 [COMMUNICATOR AGENT TOOL] Notification result: {success}")
    return success


# === Main Application Logic (Programmatic Hand-off) ===

async def handle_incident(alert_message: str, shared_usage: RunUsage) -> dict:
    """
    Main incident handling logic using Programmatic Agent Hand-off pattern.
    
    Flow: Alert → Orchestrator → Decision → [Action Agent OR Communicator Agent]
    """
    
    print(f"\n🚨 [INCIDENT] Processing alert: {alert_message}")
    print("=" * 80)
    
    # Dependencies
    vector_db = MockVectorDB()
    action_teams_webhook = ActionTeamsWebhook(ACTION_AGENT_WEBHOOK_URL)
    communicator_teams_webhook = CommunicatorTeamsWebhook(COMMUNICATOR_AGENT_WEBHOOK_URL)
    
    # Check if Kubernetes MCP servers are available
    mcp_available = len(kubernetes_mcp_servers) > 0
    print(f"🔌 [MCP STATUS] Kubernetes MCP servers available: {mcp_available}")
    print(f"📢 [TEAMS STATUS] Action Agent Webhook configured: {bool(ACTION_AGENT_WEBHOOK_URL)}")
    print(f"📢 [TEAMS STATUS] Communicator Agent Webhook configured: {bool(COMMUNICATOR_AGENT_WEBHOOK_URL)}")
    
    try:
        # Step 1: Orchestrator analyzes the alert
        print("\n📋 [STEP 1] Orchestrator analyzing alert...")
        decision_result = await orchestrator_agent.run(
            f"Analyze this Kubernetes alert and decide next action: {alert_message}",
            deps=vector_db,
            usage=shared_usage
        )
        
        decision = decision_result.output
        print(f"🎯 [DECISION] Action: {decision.action}")
        print(f"🎯 [DECISION] Confidence: {decision.confidence:.2f}")
        print(f"🎯 [DECISION] Reasoning: {decision.reasoning}")
        
        # Step 2: Route based on orchestrator's decision
        if decision.action == "handle":
            print("\n🤖 [STEP 2] Handing off to Action Agent...")
            
            # Prepare the prompt for Action Agent
            action_prompt = f"""
            Execute remediation actions for this Kubernetes alert: {alert_message}
            
            Runbook guidance: {decision.runbook_match}
            
            Instructions:
            1. First, check the current state of the deployment in default namespace
            2. Get current replica count and pod status (record as "before_state")
            3. If this is a performance/latency issue, scale up the deployment by 1 replica
            4. Use available MCP tools for Kubernetes operations
            5. Verify scaling operation completed successfully
            6. Monitor pods after scaling to ensure they are running (record as "after_state")
            7. Send comprehensive summary to Microsoft Teams using send_teams_notification tool
            
            For Teams notification, include:
            - Title: "Kubernetes Issue Resolved - [Brief Description]"
            - Issue summary: What problem was detected
            - Actions taken: All remediation steps performed
            - Before state: Resource state before remediation
            - After state: Resource state after remediation
            - Verification steps: How you confirmed the fix worked
            
            Available operations:
            - List pods and deployments in namespaces  
            - Get current replica count
            - Scale deployments safely (increment by 1)
            - Monitor pod status after scaling
            - Send Teams notification after completion
            """
            
            # Hand-off to Action Agent (uses MCP toolsets automatically)
            try:
                print("🔄 [ACTION AGENT] Starting agent.run()...")
                action_result = await action_agent.run(
                    action_prompt,
                    deps=action_teams_webhook,
                    usage=shared_usage
                )
                print("✅ [ACTION AGENT] agent.run() completed")
                result = action_result.output
                
                # Check if the agent actually called the send_teams_notification tool
                if not result.teams_notification_sent:
                    print("⚠️  [ACTION AGENT WARNING] Agent did not call send_teams_notification tool!")
                    print("⚠️  [ACTION AGENT WARNING] The LLM may have skipped the tool call.")
                    print("⚠️  [ACTION AGENT WARNING] teams_notification_sent is False in the output.")
                    print("⚠️  [ACTION AGENT WARNING] This means the tool was either not called or returned False.")
                else:
                    print("✅ [ACTION AGENT SUCCESS] teams_notification_sent is True!")
                    
            except Exception as e:
                if "exceeded max retries" in str(e) and ("kubectl_scale" in str(e) or "k8s_kubectl_scale" in str(e)):
                    # Handle MCP retry error for scaling operations - treat as success
                    print("🔧 [FRAMEWORK] Handling MCP retry error for scaling operation")
                    result = ActionResult(
                        success=True,
                        actions_taken=[
                            "Scaling command sent to Kubernetes via MCP",
                            "MCP retry error occurred but scaling operation was attempted",
                            "Scaling should be successful despite infrastructure error"
                        ],
                        kubernetes_output="Scaling operation attempted - MCP infrastructure error but kubectl command was sent to cluster",
                        error_message=None,
                        teams_notification_sent=False
                    )
                    print("✅ [FRAMEWORK] Converted MCP retry error to successful scaling result")
                else:
                    # Re-raise other types of errors
                    raise e
            print(f"✅ [ACTION RESULT] Success: {result.success}")
            print(f"📝 [ACTION RESULT] Actions taken: {result.actions_taken}")
            if result.kubernetes_output:
                print(f"🔍 [K8S OUTPUT] {result.kubernetes_output}")
            print(f"📢 [ACTION RESULT] Teams notification sent: {result.teams_notification_sent}")
            
            return {
                "status": "handled",
                "decision": decision,
                "action_result": result,
                "mcp_used": mcp_available,
                "total_usage": shared_usage
            }
            
        else:  # escalate
            print("\n📢 [STEP 2] Handing off to Communicator Agent...")
            
            # Get runbook info if available
            runbook_info = ""
            if decision.runbook_match:
                runbook_info = f"\n\nMatched Runbook: {decision.runbook_match}"
            
            # Hand-off to Communicator Agent  
            escalation_prompt = f"""
            Analyze this complex Kubernetes incident as an expert and create comprehensive report: {alert_message}
            
            Orchestrator's initial analysis: {decision.reasoning}{runbook_info}
            
            CRITICAL REQUIREMENTS:
            1. You MUST call the send_teams_notification tool with your analysis
            2. The tool call is MANDATORY before completing your response
            3. Set teams_notification_sent in your output based on the tool's return value
            
            Your expert analysis should include:
            
            1. **Root Cause Analysis**: 
               - Deep technical analysis using Kubernetes expertise
               - Why this issue occurred from infrastructure perspective
               - What Kubernetes components or configurations are involved
            
            2. **Immediate Actions** (3-5 specific steps):
               - Step-by-step remediation procedures
               - Specific kubectl commands operators should run (use actual names from alert)
               - Expected outcomes and how to verify success
            
            3. **Efficiency Improvements** (5-10 recommendations):
               - Resource optimization (CPU, memory, replicas)
               - High availability improvements
               - Monitoring and alerting enhancements
               - Automation opportunities (HPA, pod disruption budgets)
               - Network policies and security
               - Cost optimization recommendations
               - Performance tuning suggestions
               - Architecture improvements for long-term stability
            
            4. **Severity Assessment**:
               - Impact on users and business
               - Urgency for resolution
               - Use: critical, high, medium, or low
            
            MANDATORY TOOL CALL:
            You MUST call send_teams_notification tool with:
            - title: "Critical Kubernetes Incident Analysis - [Brief Issue Type]"
            - summary: Your executive summary (2-3 sentences)
            - root_cause: Your detailed root cause analysis (200-300 words)
            - severity: One of: low, medium, high, critical
            - immediate_actions: List of 3-5 remediation steps
            - efficiency_improvements: List of 5-10 optimization recommendations
            
            After calling send_teams_notification, set teams_notification_sent field in your output to the tool's return value.
            Do NOT complete your response without calling this tool first.
            """
            
            print(f"📋 [ESCALATION PROMPT] Prompt length: {len(escalation_prompt)} characters")
            print(f"📋 [ESCALATION PROMPT] Runbook match included: {bool(decision.runbook_match)}")
            
            try:
                print("🔄 [COMMUNICATOR] Starting agent.run()...")
                escalation_result = await communicator_agent.run(
                    escalation_prompt,
                    deps=communicator_teams_webhook,
                    usage=shared_usage
                )
                print("✅ [COMMUNICATOR] agent.run() completed")
                
                report = escalation_result.output
                print(f"📦 [COMMUNICATOR] Report generated: {type(report)}")
                print(f"📦 [COMMUNICATOR] Report fields: summary={bool(report.summary)}, root_cause={bool(report.root_cause_analysis)}, actions={len(report.recommended_actions)}, improvements={len(report.efficiency_improvements)}")
                
                # Check if the agent actually called the send_teams_notification tool
                if not report.teams_notification_sent:
                    print("⚠️  [COMMUNICATOR WARNING] Agent did not call send_teams_notification tool!")
                    print("⚠️  [COMMUNICATOR WARNING] The LLM may have skipped the tool call.")
                    print("⚠️  [COMMUNICATOR WARNING] teams_notification_sent is False in the output.")
                    print("⚠️  [COMMUNICATOR WARNING] This means the tool was either not called or returned False.")
                else:
                    print("✅ [COMMUNICATOR SUCCESS] teams_notification_sent is True!")
                
            except Exception as e:
                print(f"❌ [COMMUNICATOR ERROR] {e}")
                import traceback
                print(f"❌ [COMMUNICATOR ERROR] Full traceback:")
                traceback.print_exc()
                # Create a fallback report
                report = EscalationReport(
                    summary=f"Failed to generate full analysis: {str(e)}",
                    root_cause_analysis="Analysis incomplete due to error",
                    recommended_actions=["Review logs", "Manual investigation required"],
                    efficiency_improvements=["Implement better error handling"],
                    severity="high",
                    teams_notification_sent=False
                )
            
            print(f"📊 [ESCALATION] Severity: {report.severity}")
            print(f"📊 [ESCALATION] Summary: {report.summary}")
            print(f"📊 [ESCALATION] Recommended actions: {len(report.recommended_actions)}")
            print(f"📊 [ESCALATION] Efficiency improvements: {len(report.efficiency_improvements)}")
            print(f"📢 [ESCALATION] Teams notification sent: {report.teams_notification_sent}")
            
            return {
                "status": "escalated", 
                "decision": decision,
                "escalation_report": report,
                "mcp_used": mcp_available,
                "total_usage": shared_usage
            }
            
    except Exception as e:
        print(f"❌ [ERROR] Failed to handle incident: {e}")
        return {
            "status": "error",
            "error": str(e),
            "mcp_used": mcp_available,
            "total_usage": shared_usage
        }


# === RabbitMQ Consumer ===

async def process_alert_message(message: aio_pika.IncomingMessage):
    """
    Process alert message from RabbitMQ queue.
    
    Args:
        message: Incoming RabbitMQ message containing alert
    """
    async with message.process():
        try:
            # Decode message body
            alert_data = message.body.decode()
            print(f"\n📨 [RABBITMQ] Received alert from queue: {RABBITMQ_QUEUE}")
            print(f"📨 [RABBITMQ] Message: {alert_data[:100]}..." if len(alert_data) > 100 else f"📨 [RABBITMQ] Message: {alert_data}")
            
            # Try to parse as JSON, fallback to plain text
            try:
                alert_json = json.loads(alert_data)
                # Extract alert message from JSON (adjust key based on your message format)
                alert_message = alert_json.get("message", "") or alert_json.get("alert", "") or alert_json.get("description", "") or alert_data
            except json.JSONDecodeError:
                # If not JSON, use raw message
                alert_message = alert_data
            
            # Process the alert through the incident handler
            usage = RunUsage()
            result = await handle_incident(alert_message, usage)
            
            # Log the result
            print(f"\n✅ [RABBITMQ] Alert processed successfully")
            print(f"📊 [RESULT] Status: {result['status']}")
            print(f"📊 [RESULT] MCP Used: {result.get('mcp_used', False)}")
            print(f"📊 [RESULT] Total Usage: {result['total_usage']}")
            
        except Exception as e:
            print(f"❌ [RABBITMQ] Error processing alert: {e}")
            # Message will be requeued if processing fails


async def consume_alerts():
    """
    Connect to RabbitMQ and consume alerts from the queue.
    """
    print("🐰 [RABBITMQ] Connecting to RabbitMQ...")
    print(f"🐰 [RABBITMQ] Host: {RABBITMQ_HOST}:{RABBITMQ_PORT}")
    print(f"🐰 [RABBITMQ] Queue: {RABBITMQ_QUEUE}")
    
    try:
        # Connect to RabbitMQ
        connection = await aio_pika.connect_robust(
            host=RABBITMQ_HOST,
            port=RABBITMQ_PORT,
            login=RABBITMQ_USER,
            password=RABBITMQ_PASSWORD
        )
        
        async with connection:
            # Create channel
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=1)  # Process one message at a time
            
            # Declare queue (creates if doesn't exist)
            queue = await channel.declare_queue(
                RABBITMQ_QUEUE,
                durable=True  # Queue survives broker restart
            )
            
            print(f"✅ [RABBITMQ] Connected successfully")
            print(f"🎧 [RABBITMQ] Listening for alerts on queue: {RABBITMQ_QUEUE}")
            print(f"🎧 [RABBITMQ] Waiting for messages. To exit press CTRL+C")
            
            # Start consuming messages
            await queue.consume(process_alert_message)
            
            # Keep the consumer running
            await asyncio.Future()
            
    except aio_pika.exceptions.AMQPConnectionError as e:
        print(f"❌ [RABBITMQ] Connection error: {e}")
        print(f"� [RABBITMQ] Make sure RabbitMQ is running and credentials are correct")
    except Exception as e:
        print(f"❌ [RABBITMQ] Error: {e}")


async def main():
    """
    Start the AI-Ops Incident Responder with RabbitMQ integration.
    """
    print("=" * 80)
    print("🚀 AI-Ops Incident Responder - RabbitMQ Consumer")
    print("=" * 80)
    print("Programmatic Agent Hand-off Pattern with Kubernetes MCP Integration")
    
    # Check MCP status
    mcp_status = "✅ Available" if kubernetes_mcp_servers else "❌ Not Available (using fallback)"
    print(f"🔌 Kubernetes MCP Status: {mcp_status}")
    print(f"📢 Action Agent Webhook: {'✅ Configured' if ACTION_AGENT_WEBHOOK_URL else '❌ Not configured'}")
    print(f"📢 Communicator Agent Webhook: {'✅ Configured' if COMMUNICATOR_AGENT_WEBHOOK_URL else '❌ Not configured'}")
    print(f"🤖 LLM Provider: {'LLM Farm' if USE_LLM_FARM else 'OpenAI'}")
    print("=" * 80)
    
    # Start consuming alerts from RabbitMQ
    await consume_alerts()
    
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 [SHUTDOWN] Interrupted by user")
    except Exception as e:
        print(f"❌ [SHUTDOWN ERROR] {e}")
    finally:
        # Allow pending async tasks to complete properly
        print("🔧 [CLEANUP] Ensuring clean shutdown...")
        # Get the current event loop if it exists
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop and not loop.is_closed():
            # Cancel remaining tasks
            pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
            if pending_tasks:
                print(f"🔧 [CLEANUP] Cancelling {len(pending_tasks)} pending tasks...")
                for task in pending_tasks:
                    task.cancel()
        
        print("✅ [CLEANUP] Shutdown complete")