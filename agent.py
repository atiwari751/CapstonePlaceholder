import logging
from typing import Optional, List, Any

from module.memory import Memory
from module.perception import Perception # Now integrating Perception
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
        self.tools_execution_map = {tool.name: tool for tool in (tools_list or [])}

        self.is_session_active = False
        logger.info(f"ConversationalAgent initialized with session ID: {self.memory.session_id}")

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

    def process_user_query(self, user_query: str) -> str:
        """
        Processes the user's query.
        In Phase 1, this involves:
        - Processing input with Perception.
        - Storing user query in memory.
        - Checking for repeated questions.
        - Calling Decision module with perception data.
        - Executing tools (basic) and handling memory actions.
        - Storing agent response in memory.
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
        
        thought, tool_name, tool_input, speak_output, memory_actions = \
            self.decision.execute_plan_with_memory(
                user_query=user_query,
                perception_data=perception_output, # Use actual perception_output from Phase 3 integration
                memory_context=relevant_context_from_memory,
                full_conversation_history=self.memory.get_conversation_history()
            )
        
        logger.info(f"Session {self.memory.session_id} - Decision LLM Thought: {thought}")
        logger.info(f"Session {self.memory.session_id} - Chosen tool: {tool_name}, Input: {tool_input}")
        logger.info(f"Session {self.memory.session_id} - LLM Speak Output: {speak_output}")
        logger.info(f"Session {self.memory.session_id} - Memory Actions: {memory_actions}")

        # Process memory actions indicated by the Decision module
        if memory_actions:
            for mem_action in memory_actions:
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

        final_agent_answer = speak_output # Start with what the LLM wants to say
        tool_output_content = None

        if tool_name and tool_name != "final_answer":
            if tool_name in self.tools_execution_map:
                tool_to_call = self.tools_execution_map[tool_name]
                try:
                    logger.info(f"Session {self.memory.session_id} - Executing tool: {tool_name} with input: {tool_input}")
                    tool_execution_result = tool_to_call.execute(tool_input)
                    tool_output_content = str(tool_execution_result) # Convert tool output to string
                    logger.info(f"Session {self.memory.session_id} - Tool {tool_name} output: {tool_output_content}")
                    # In Phase 2, we directly use/append tool output. Phase 4 will re-evaluate with LLM.
                    final_agent_answer = f"{speak_output}\nTool '{tool_name}' executed. Result: {tool_output_content}"
                except Exception as e:
                    logger.error(f"Session {self.memory.session_id} - Error executing tool {tool_name}: {e}")
                    final_agent_answer = f"Sorry, I encountered an error trying to use the tool: {tool_name}. Error: {str(e)}"
                    tool_output_content = f"Error: {str(e)}"
            else:
                logger.warning(f"Session {self.memory.session_id} - LLM chose tool '{tool_name}' but it's not in tools_execution_map.")
                final_agent_answer = f"I thought about using a tool called '{tool_name}', but I don't seem to have it available right now."

        self.memory.add_conversation_turn(role="agent", content=thought, tool_name=tool_name if tool_name != "final_answer" else None, tool_input=tool_input, tool_output=tool_output_content, final_answer=final_agent_answer)
        logger.info(f"Session {self.memory.session_id} - Agent response: {final_agent_answer}")

        return final_agent_answer