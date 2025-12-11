"use client";

import { AgentState } from "@/lib/types";
import {
  useCoAgent,
  useFrontendTool,
} from "@copilotkit/react-core";
import { CopilotKitCSSProperties, CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";
import { useState } from "react";

export default function CopilotKitPage() {
  const [themeColor, setThemeColor] = useState("#6366f1");

  // 🪁 Frontend Actions: https://docs.copilotkit.ai/pydantic-ai/frontend-actions
  useFrontendTool({
    name: "setThemeColor",
    parameters: [
      {
        name: "themeColor",
        description: "The theme color to set. Make sure to pick nice colors.",
        required: true,
      },
    ],
    handler({ themeColor }) {
      setThemeColor(themeColor);
    },
  });

  return (
    <main
      style={
        { "--copilot-kit-primary-color": themeColor } as CopilotKitCSSProperties
      }
    >
      <YourMainContent themeColor={themeColor} />
    </main>
  );
}

function YourMainContent({ themeColor }: { themeColor: string }) {
  // 🪁 Shared State: https://docs.copilotkit.ai/pydantic-ai/shared-state
  const { state, setState } = useCoAgent<AgentState>({
    name: "my_agent",
    initialState: {
      conversation_history: [],
    },
  });

  return (
    <div className="h-screen w-full bg-white flex flex-col overflow-hidden">
      {/* Header Section - Fixed at top */}
      <div className="w-full py-6 px-8 bg-white border-b-2 border-gray-100 flex-shrink-0">
        <div className="max-w-6xl mx-auto">
          <h1 
            style={{ color: themeColor }}
            className="text-4xl font-bold mb-2 transition-colors duration-300"
          >
            🤖 PM Assistant
          </h1>
          <p className="text-gray-600 text-base">
            Your intelligent companion for project management
          </p>
        </div>
      </div>

      {/* Divider Line */}
      <div className="w-full border-b border-gray-200 flex-shrink-0" />

      {/* Chat Container - Takes remaining space, scrollable */}
      <div className="flex-1 w-full overflow-hidden">
        <div className="h-full max-w-6xl mx-auto">
          <CopilotChat
            labels={{
              title: "Chat",
              initial: "👋 Hi! I'm your PM Assistant. How can I help you today?",
            }}
            instructions="You are a helpful PM assistant that helps with project management tasks and various queries."
            className="h-full"
          />
        </div>
      </div>
    </div>
  );
}
