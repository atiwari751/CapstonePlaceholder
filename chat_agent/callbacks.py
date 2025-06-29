import uuid
from typing import Any, Dict, List, Union
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
import json
import re

# Import the scheme service singleton
from scheme_service import scheme_service

class SessionCallbackHandler(BaseCallbackHandler):
    """Callback handler to capture agent's intermediate steps for a session."""

    def __init__(self, session_data: Dict[str, Any]):
        self.session_data = session_data
        self.session_data["results"] = {}
        self.session_data.setdefault("schemes", [])  # Ensure schemes list exists
        self._current_tool_step = 0

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action (when a tool is about to be called)."""
        tool_name = action.tool
        tool_input = action.tool_input
        
        # Create a placeholder for the tool with "Running" status
        step_key = f"tool_{self._current_tool_step}"
        self.session_data["results"][step_key] = {
            "tool": tool_name,
            "input": tool_input,
            "result": "Executing...",
            "status": "Running"
        }

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when a tool ends successfully."""
        step_key = f"tool_{self._current_tool_step}"
        
        if step_key in self.session_data["results"]:
            self.session_data["results"][step_key]["result"] = output
            self.session_data["results"][step_key]["status"] = "Finished"

        # --- Scheme Data Extraction ---
        scheme_data_match = re.search(r'SCHEME_DATA:\s*(\{.*\})', output, re.DOTALL | re.IGNORECASE)
        if scheme_data_match:
            try:
                parsed_data = json.loads(scheme_data_match.group(1))
                if "schemes" in parsed_data and isinstance(parsed_data["schemes"], list):
                    for scheme_entry in parsed_data["schemes"]:
                        # Create scheme object but store its dict representation in the session
                        new_scheme = scheme_service.create_scheme_from_agent_data(scheme_entry)
                        self.session_data["schemes"].append(new_scheme.dict())
                    print(f"Callback extracted {len(parsed_data['schemes'])} schemes.")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Callback handler failed to parse SCHEME_DATA: {e}")

        self._current_tool_step += 1

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        self.session_data["final_answer"] = finish.return_values.get("output")
        self.session_data["status"] = "completed"

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run on tool error."""
        step_key = f"tool_{self._current_tool_step}"
        if step_key in self.session_data["results"]:
            self.session_data["results"][step_key]["result"] = f"Error: {str(error)}"
            self.session_data["results"][step_key]["status"] = "Error"
        self.session_data["status"] = "error"
        self.session_data["error"] = f"An error occurred in a tool: {str(error)}"
        self._current_tool_step += 1