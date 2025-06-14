import logging
from typing import Optional, List, Any
import subprocess # For managing mcp-server subprocess
import sys
import os
import asyncio # For awaiting async methods
import json # For json.dumps in reevaluation_query

from module.memory import Memory
from module.perception import Perception
from module.decision import Decision

logger = logging.getLogger(__name__)


class ConversationalAgent:
    def __init__(self, tools_list: Optional[List[Any]] = None):
        """
        Initializes the Conversational Agent.

        Args:
            tools_list (Optional[List[Any]]): A list of tool objects.
                                              Will be used in later phases.
        """
        self.memory = Memory()  # Each agent instance gets its own memory
        self.perception = Perception()
        self.decision = Decision(tools_list=tools_list or []) # Decision module already takes tools_list

        self.mcp_server_process: Optional[subprocess.Popen] = None
        self.available_tools_list = tools_list or []
        self.tools_execution_map = {tool.name: tool for tool in self.available_tools_list}

        self.start_mcp_server() # Start MCP server when agent is initialized
        self._inject_tool_dependencies()
        self.is_session_active = False
        logger.info(f"ConversationalAgent initialized with session ID: {self.memory.session_id}")

    def _inject_tool_dependencies(self):
        """Injects necessary dependencies (like memory, other tools) into each tool."""
        for tool_name, tool_instance in self.tools_execution_map.items():
            if hasattr(tool_instance, 'set_dependencies'):
                # Prepare dependencies that the tool might need
                dependencies = {}
                if hasattr(tool_instance.set_dependencies, '__code__'):
                    expected_args = tool_instance.set_dependencies.__code__.co_varnames
                    if 'memory' in expected_args:
                        dependencies['memory'] = self.memory
                    if 'mcp_process' in expected_args and self.mcp_server_process:
                        dependencies['mcp_process'] = self.mcp_server_process

                if dependencies: # Only call if we have something to inject based on signature
                    try:
                        tool_instance.set_dependencies(**dependencies)
                        logger.info(f"Injected dependencies into tool: {tool_name}")
                    except TypeError as e:
                        logger.warning(f"Could not inject dependencies into tool {tool_name} due to TypeError: {e}. Check set_dependencies signature.")
                else:
                    logger.debug(f"Tool {tool_name} has set_dependencies but no matching arguments found for injection.")
            else:
                logger.debug(f"Tool {tool_name} does not have a set_dependencies method.")

    def start_mcp_server(self):
        """Starts the mcp-server.py as a subprocess."""
        if self.mcp_server_process and self.mcp_server_process.poll() is None:
            logger.info("MCP server process is already running.")
            return

        # Determine path to mcp-server.py.
        # Assumes mcp-server.py is in the parent directory of where agent.py is located (e.g., FinalProj/mcp-server.py)
        # or in the same directory if agent.py is top-level.
        agent_script_dir = os.path.dirname(os.path.abspath(__file__))
        #print(f"[AGENT_DEBUG] Agent script dir: {agent_script_dir}") # ADD PRINT

        # Try path assuming agent.py is in FinalProj/ and mcp-server.py is in FinalProj/)
        mcp_server_script_path = os.path.join(agent_script_dir, "mcp-server.py")    # If agent.py in FinalProj/
        #print(f"[AGENT_DEBUG] Attempting to start Popen with: {mcp_server_script_path}") # ADD PRINT

        try:
            self.mcp_server_process = subprocess.Popen(
                [sys.executable, mcp_server_script_path],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
                text=True, cwd=os.path.dirname(mcp_server_script_path)
            )
            logger.info(f"MCP server process started (PID: {self.mcp_server_process.pid}) with script: {mcp_server_script_path}")
        except Exception as e:
            logger.error(f"Failed to start MCP server process: {e}")
            self.mcp_server_process = None

    def start_session(self) -> str:
        """Starts a new conversation session."""
        if self.is_session_active:
            # If a session is somehow already active, perhaps just continue
            logger.warning(f"Session {self.memory.session_id} start_session called while already active.")
            # return "A session is already active. What do you want to do?"
        
        self.is_session_active = True
        # Clear memory from any previous session if this agent instance is reused.
        # If session_id is unique per agent instance, this might be redundant
        # but good for explicit state reset.
        self.memory.clear() 
        logger.info(f"Session {self.memory.session_id} started.")
        
        initial_greeting = "What do you want to do today?"
        # Add agent's greeting to memory as the first turn of the agent.
        self.memory.add_conversation_turn(role="agent", content=initial_greeting, final_answer=initial_greeting)
        return initial_greeting

    def end_session(self) -> str:
        """Ends the current conversation session."""
        if not self.is_session_active:
            logger.info("end_session called but no active session.")
            return "No active session to end."
        
        farewell_message = "That is all, thank you."
        logger.info(f"Session {self.memory.session_id} ended by user request or command.")
        # Add a final agent turn indicating session end, if appropriate.
        # The user's "bye" is already added in process_user_query.
        # This could be an internal marker or a final confirmation.
        self.memory.add_conversation_turn(role="agent", content="Session ended.", final_answer=farewell_message)
        self.is_session_active = False
        return farewell_message

    async def process_user_query(self, user_query: str) -> str:
        """
        Processes the user's query.
        """
        if not self.is_session_active:
            # This case should ideally be handled by the calling loop in main.py
            # (i.e., start_session should be called first).
            # However, as a fallback:
            logger.warning("process_user_query called without an active session. Starting one implicitly.")
            self.start_session() # This will add the agent's greeting to memory.

        # --- Phase 3: Perception Processing ---
        perception_output = self.perception.process_user_input(user_query)
        logger.info(f"Session {self.memory.session_id} - Perception output: {perception_output}")

        logger.info(f"Session {self.memory.session_id} - User query: {user_query}")
        self.memory.add_conversation_turn(role="user", content=user_query)

        if user_query.strip().lower() in ["that is all, thank you.", "bye", "exit", "quit"]:
            return self.end_session()

        repeated_answer = self.memory.check_for_repeated_question(user_query)
        if repeated_answer:
            logger.info(f"Session {self.memory.session_id} - Responding with answer from memory (repeated question).")
            self.memory.add_conversation_turn(role="agent", content="Retrieved from memory.", final_answer=repeated_answer)
            # The 'repeated_answer' from memory.check_for_repeated_question is the 
            # full 'final_answer' of the previous agent turn.
            return repeated_answer
        
        # --- Phase 2: Decision Making & Basic Tool Execution ---
        relevant_context_from_memory = self.memory.get_relevant_context_for_decision()
        
        # Initial decision pass
        llm_thought, llm_tool_name, llm_tool_input, llm_speak_output, llm_memory_actions = \
            await self.decision.execute_plan_with_memory( # Assuming Decision's method is async
                user_query=user_query,
                perception_data=perception_output, # Use actual perception_output from Phase 3 integration
                memory_context=relevant_context_from_memory,
                full_conversation_history=self.memory.get_conversation_history(),
                # previous_tool_output_summary=None # For re-planning due to errors, not used in this re-evaluation step
            )
        
        logger.info(f"Session {self.memory.session_id} - Initial Decision LLM Thought: {llm_thought}")
        logger.info(f"Session {self.memory.session_id} - Initial Chosen tool: {llm_tool_name}, Input: {llm_tool_input}")
        logger.info(f"Session {self.memory.session_id} - Initial LLM Speak Output: {llm_speak_output}")
        logger.info(f"Session {self.memory.session_id} - Initial Memory Actions: {llm_memory_actions}")

        # Process memory actions indicated by the Decision module
        if llm_memory_actions:
            for mem_action in llm_memory_actions:
                action_type = mem_action.get("action")
                if action_type == "store_fact":
                    self.memory.store_fact(mem_action.get("key"), mem_action.get("value"))
                elif action_type == "set_current_scheme":
                    scheme_id_to_set = mem_action.get("scheme_id")
                    if scheme_id_to_set:
                        self.memory.set_current_scheme(scheme_id_to_set)
                    else:
                        logger.warning(f"Session {self.memory.session_id} - 'set_current_scheme' action missing 'scheme_id'.")
                elif action_type == "update_current_scheme_data":
                    data_to_update = mem_action.get("data")
                    if data_to_update:
                        self.memory.update_current_scheme_data(data_to_update)
                    else:
                        logger.warning(f"Session {self.memory.session_id} - 'update_current_scheme_data' action missing 'data'.")
                else:
                    logger.warning(f"Session {self.memory.session_id} - Unknown memory action type: {action_type}")

        final_agent_answer = llm_speak_output # Default to initial speak output
        tool_output_content = None

        if llm_tool_name and llm_tool_name != "final_answer":
            if llm_tool_name in self.tools_execution_map:
                tool_to_call = self.tools_execution_map[llm_tool_name]
                try:
                    logger.info(f"Session {self.memory.session_id} - Executing tool: {llm_tool_name} with input: {llm_tool_input}")
                    tool_execution_result = await tool_to_call.execute(llm_tool_input) # Await async tool execution
                    
                    tool_output_content = str(tool_execution_result) # For logging the full raw result
                    logger.info(f"Session {self.memory.session_id} - Tool {llm_tool_name} output: {tool_output_content}")

                    # --- Phase 4.2: Re-evaluation Step ---
                    # Feed the tool's output back to the LLM to formulate a final response.
                    if tool_execution_result.get("status") == "success":
                        actual_tool_output = tool_execution_result.get("output", "Tool executed successfully but provided no specific output.")
                        
                        # Construct a new query/context for the LLM that includes the tool's output
                        # This prompt tells the LLM to generate a user-facing response based on the tool's action.
                        reevaluation_query = (
                            f"The user originally asked: '{user_query}'.\n"
                            f"I decided to use the tool '{llm_tool_name}' with input {json.dumps(llm_tool_input)}.\n"
                            f"The tool '{llm_tool_name}' executed and returned: {json.dumps(actual_tool_output)}.\n"
                            f"Based on this, please formulate a concise and natural final answer for the user. "
                            f"Do not mention the tool name unless it's natural to do so. Just provide the answer."
                        )
                        logger.info(f"Session {self.memory.session_id} - Re-evaluating with LLM. Query for re-evaluation: {reevaluation_query}")

                        # Call decision module again, but this time it should aim for a final_answer
                        # We might need a flag or modified prompt for the Decision module to know it's in a re-evaluation phase.
                        # For simplicity, we'll rely on the prompt structure.
                        # The Decision module's system prompt should guide it to provide a direct answer
                        # when given context about a tool's successful execution.
                        
                        # We pass the reevaluation_query as the new "user_query" for this LLM pass.
                        # Perception data might be less relevant here, or could be re-run on reevaluation_query if needed.
                        # Memory context and history are still relevant.
                        final_thought, final_tool_name, _, final_speak_output, final_memory_actions = \
                            await self.decision.execute_plan_with_memory(
                                user_query=reevaluation_query, # This is the key for re-evaluation
                                perception_data=self.perception.process_user_input(reevaluation_query), # Re-perceive the new context
                                memory_context=relevant_context_from_memory, # Could be updated with new facts from llm_memory_actions
                                full_conversation_history=self.memory.get_conversation_history() # History now includes the first LLM thought
                            )
                        
                        logger.info(f"Session {self.memory.session_id} - Re-evaluation LLM Thought: {final_thought}")
                        logger.info(f"Session {self.memory.session_id} - Re-evaluation LLM Speak Output: {final_speak_output}")

                        # Process any memory actions from the re-evaluation step
                        if final_memory_actions:
                            for mem_action in final_memory_actions:
                                # ... (process memory actions as before) ...
                                action_type = mem_action.get("action")
                                if action_type == "store_fact":
                                    self.memory.store_fact(mem_action.get("key"), mem_action.get("value"))
                                # ... (other memory actions) ...

                        if final_tool_name == "final_answer" and final_speak_output:
                            final_agent_answer = final_speak_output
                        else:
                            # If LLM still wants to use a tool or doesn't give a final answer, fallback
                            logger.warning(f"Session {self.memory.session_id} - LLM did not provide a final_answer after tool re-evaluation. Falling back.")
                            final_agent_answer = f"The tool '{llm_tool_name}' was used. Result: {actual_tool_output}"

                    elif tool_execution_result.get("status") == "error":
                        actual_output = tool_execution_result.get("message", "An error occurred with the tool.")
                        final_agent_answer = f"{llm_speak_output or f'I tried to use the {llm_tool_name} tool, but there was an error.'} Error: {actual_output}"
                    else: # Fallback for unexpected tool status
                        final_agent_answer = f"{llm_speak_output or f'I used the {llm_tool_name} tool.'} Details: {tool_output_content}"

                except Exception as e:
                    logger.error(f"Session {self.memory.session_id} - Error executing tool {llm_tool_name}: {e}")
                    final_agent_answer = f"Sorry, I encountered an error trying to use the tool: {llm_tool_name}. Error: {str(e)}"
                    tool_output_content = f"Error: {str(e)}" # For logging
            else:
                logger.warning(f"Session {self.memory.session_id} - LLM chose tool '{llm_tool_name}' but it's not in tools_execution_map.")
                final_agent_answer = f"I thought about using a tool called '{llm_tool_name}', but I don't seem to have it available right now."
        else: # LLM decided on a final_answer in the first pass
            final_agent_answer = llm_speak_output

        # Log the final state before returning
        # The 'thought' logged here should be the one that led to this final_agent_answer
        # If re-evaluation happened, final_thought is more relevant. If not, llm_thought.
        current_thought_for_log = final_thought if 'final_thought' in locals() else llm_thought
        self.memory.add_conversation_turn(
            role="agent", 
            content=current_thought_for_log, 
            tool_name=llm_tool_name if llm_tool_name != "final_answer" and not ('final_thought' in locals() and final_tool_name == "final_answer") else None, 
            tool_input=llm_tool_input if llm_tool_name != "final_answer" and not ('final_thought' in locals() and final_tool_name == "final_answer") else None, 
            tool_output=tool_output_content, 
            final_answer=final_agent_answer
        )
        logger.info(f"Session {self.memory.session_id} - Agent response: {final_agent_answer}")

        return final_agent_answer
    
    def stop_mcp_server(self):
        """Stops the MCP server subprocess."""
        if self.mcp_server_process and self.mcp_server_process.poll() is None: # Check if process is running
            logger.info(f"Stopping MCP server process (PID: {self.mcp_server_process.pid})...")
            self.mcp_server_process.terminate() # Send SIGTERM
            try:
                self.mcp_server_process.wait(timeout=5) # Wait for a few seconds
                logger.info("MCP server process terminated.")
            except subprocess.TimeoutExpired:
                logger.warning("MCP server process did not terminate in time, killing...")
                self.mcp_server_process.kill() # Force kill
                logger.info("MCP server process killed.")
            self.mcp_server_process = None

    def __del__(self):
        """Ensure MCP server is stopped when the agent object is deleted."""
        self.stop_mcp_server()