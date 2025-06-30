import uuid
from typing import Any, Dict, List, Union, TYPE_CHECKING
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage, HumanMessage
import json
import re

# Import the scheme service singleton
from scheme_service import scheme_service

if TYPE_CHECKING:
    import shelve

class SessionCallbackHandler(BaseCallbackHandler):
    """Callback handler to capture agent's intermediate steps for a session."""

    def __init__(self, sessions: 'shelve.Shelf', session_id: str):
        self.sessions = sessions
        self.session_id = session_id
        # IMPORTANT: We get a copy from shelve, not a direct reference
        self.session_data = self.sessions[self.session_id]
        # Ensure 'results' exists but don't overwrite it if it's already there.
        self.session_data.setdefault("results", {})
        self.session_data.setdefault("schemes", [])  # Ensure schemes list exists
        self.session_data.setdefault("chat_history", []) # Ensure chat history exists
        self._current_tool_step = len(self.session_data.get("results", {}))
        self.query = ""

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
        # Persist the "Running" status immediately
        self.sessions[self.session_id] = self.session_data

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Capture the input query when the chain starts."""
        # This is needed to correctly update chat_history on finish.
        # Check if inputs is a dictionary and contains the 'input' key.
        if isinstance(inputs, dict) and "input" in inputs:
            self.query = inputs.get("input", "")

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when a tool ends successfully."""
        step_key = f"tool_{self._current_tool_step}"
        
        if step_key in self.session_data["results"]:
            self.session_data["results"][step_key]["result"] = output
            self.session_data["results"][step_key]["status"] = "Finished"
            # Persist the tool result
            self.sessions[self.session_id] = self.session_data

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
                        # Persist the extracted schemes
                        self.sessions[self.session_id] = self.session_data
                    print(f"Callback extracted {len(parsed_data['schemes'])} schemes.")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Callback handler failed to parse SCHEME_DATA: {e}")

        self._current_tool_step += 1


    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        final_answer = finish.return_values.get("output")
        self.session_data["final_answer"] = final_answer
        self.session_data["status"] = "completed"
 
        # Append the last exchange to the history as serializable dicts
        if self.query:
            # self.session_data['chat_history'] is initialized as a list
            self.session_data["chat_history"].append({"type": "human", "content": self.query})
            self.session_data["chat_history"].append({"type": "ai", "content": final_answer})

        self.sessions[self.session_id] = self.session_data

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run on tool error."""
        step_key = f"tool_{self._current_tool_step}"
        if step_key in self.session_data["results"]:
            self.session_data["results"][step_key]["result"] = f"Error: {str(error)}"
            self.session_data["results"][step_key]["status"] = "Error"
        self.session_data["status"] = "error"
        self.session_data["error"] = f"An error occurred in a tool: {str(error)}"
        # Persist the error state
        self.sessions[self.session_id] = self.session_data
        self._current_tool_step += 1