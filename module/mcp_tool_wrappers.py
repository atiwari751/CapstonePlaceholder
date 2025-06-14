import logging
from typing import Dict, Any, Optional, List
import subprocess # For interacting with mcp-server subprocess
import json
import uuid
import asyncio # For async subprocess communication

from module.memory import Memory # Import Memory

logger = logging.getLogger(__name__)


class BaseMCPToolWrapper:
    """
    Base class for MCP tool wrappers that communicate with an mcp-server.py
    subprocess started by the agent.
    """
    def __init__(self, name: str, description: str, input_schema_properties: Optional[Dict[str, Any]] = None, required_params: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self._input_schema_properties = input_schema_properties if input_schema_properties is not None else {}
        self._required_params = required_params if required_params is not None else []
        self.memory: Optional[Memory] = None
        self.mcp_process: Optional[subprocess.Popen] = None
        logger.info(f"MCP Tool Wrapper '{self.name}' initialized.")

    def set_dependencies(self, mcp_process: subprocess.Popen, memory: Memory):
        """Injects the MCP subprocess and Memory."""
        self.mcp_process = mcp_process
        self.memory = memory
        logger.info(f"Tool '{self.name}' received mcp_process and memory dependencies.")

    def get_schema(self) -> Dict[str, Any]:
        """Returns the tool's input schema for the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self._input_schema_properties,
                "required": self._required_params
            }
        }

    async def execute(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the tool by calling the MCP server via subprocess pipes using asyncio."""
        if not self.mcp_process or self.mcp_process.stdin is None or self.mcp_process.stdout is None :
            logger.error(f"MCP server process not available or its pipes are not configured for tool '{self.name}'. Cannot execute.")
            return {"status": "error", "message": "MCP server process not available."}

        logger.info(f"Tool '{self.name}' calling MCP server with input: {tool_input}")
        try:
            request_id = str(uuid.uuid4())
            json_rpc_request = {
                "jsonrpc": "2.0",
                "method": self.name,
                "params": tool_input,
                "id": request_id
            }
            request_str = json.dumps(json_rpc_request) + "\n"
            logger.info(f"Tool '{self.name}' attempting to send to MCP stdin: {request_str.strip()}") # Changed to INFO for visibility

            loop = asyncio.get_event_loop()

            def write_to_stdin_sync():
                logger.debug(f"Tool '{self.name}': In executor - writing to stdin...")
                if self.mcp_process and self.mcp_process.stdin:
                    self.mcp_process.stdin.write(request_str)
                    self.mcp_process.stdin.flush()
                    logger.debug(f"Tool '{self.name}': In executor - flushed stdin.")
                else:
                    logger.error(f"Tool '{self.name}': In executor - mcp_process.stdin is None!")
            await loop.run_in_executor(None, write_to_stdin_sync)
            logger.info(f"Tool '{self.name}' finished attempting to write to MCP stdin.")

            def read_from_stdout_sync():
                logger.debug(f"Tool '{self.name}': In executor - reading from stdout...")
                if self.mcp_process and self.mcp_process.stdout:
                    line = self.mcp_process.stdout.readline()
                    logger.debug(f"Tool '{self.name}': In executor - readline returned: '{line.strip() if line else None}'")
                    return line
                logger.error(f"Tool '{self.name}': In executor - mcp_process.stdout is None!")
                return None
            response_str = await loop.run_in_executor(None, read_from_stdout_sync)

            if response_str is None:
                logger.error(f"Tool '{self.name}' failed to read from MCP stdout.")
                return {"status": "error", "message": "Failed to read response from MCP server."}

            logger.debug(f"Tool '{self.name}' received from MCP stdout: {response_str.strip()}")
            if not response_str.strip():
                return {"status": "error", "message": "Empty response from MCP server."}

            mcp_response_data = json.loads(response_str)
            
            if "result" in mcp_response_data:
                return {"status": "success", "tool_name": self.name, "output": mcp_response_data["result"]}
            elif "error" in mcp_response_data:
                return {"status": "error", "tool_name": self.name, "message": mcp_response_data["error"]}
            else:
                return {"status": "error", "message": "Unknown response format from MCP server."}

        except Exception as e:
            logger.error(f"Error executing MCP tool '{self.name}': {e}")
            return {"status": "error", "tool_name": self.name, "message": str(e)}


class AddMCPToolWrapper(BaseMCPToolWrapper):
    def __init__(self):
        super().__init__(
            name="add",
            description="Adds two numbers using the MCP server.",
            input_schema_properties={"a": {"type": "integer"}, "b": {"type": "integer"}},
            required_params=["a", "b"]
        )

class AiFormSchemerMCPToolWrapper(BaseMCPToolWrapper):
    def __init__(self):
        super().__init__(
            name="ai_form_schemer",
            description="Uses AI to generate and evaluate a building form scheme via MCP server.",
            input_schema_properties={
                "grid_spacing_x": {"type": "number"}, "grid_spacing_y": {"type": "number"},
                "extents_x": {"type": "number"}, "extents_y": {"type": "number"},
                "no_of_floors": {"type": "integer"}
            },
            required_params=["grid_spacing_x", "grid_spacing_y", "extents_x", "extents_y", "no_of_floors"]
        )

class MaterialPropertyFetcherMCPToolWrapper(BaseMCPToolWrapper):
    def __init__(self):
        super().__init__(
            name="material_property_fetcher", 
            description="Fetches a specific property for a given material from the MCP server. Stores fetched property in agent memory.",
            input_schema_properties={
                "material_name": {"type": "string", "description": "The name of the material"},
                "property_name": {"type": "string", "description": "The property to fetch"}
            },
            required_params=["material_name", "property_name"]
        )

    async def execute(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        mcp_result = await super().execute(tool_input)
        if mcp_result.get("status") == "success" and self.memory and mcp_result.get("output"):
            output_data = mcp_result.get("output", {})
            material = tool_input.get("material_name", "unknown_material")
            prop = tool_input.get("property_name", "unknown_property")
            value_fetched = output_data.get("value", output_data) 
            if value_fetched is not None:
                fact_key = f"material_property_{material}_{prop}"
                self.memory.store_fact(fact_key, value_fetched)
                logger.info(f"Stored in agent memory via {self.name}: {fact_key} = {value_fetched}")
        return mcp_result