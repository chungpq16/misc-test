// State of the agent, make sure this aligns with your agent's state.
export type AgentState = {
  conversation_history: string[];
  env_data_loaded: boolean;
  lb_data_loaded: boolean;
  excel_data_loaded: boolean;
  last_sql_query?: string;
  last_result_count: number;
  current_firewall_request?: {
    plant_name: string;
    use_case: string;
    environment: string;
  };
}