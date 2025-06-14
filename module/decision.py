import logging
import json
from typing import List, Dict, Any, Tuple, Optional
import asyncio
import google.generativeai as genai # Standard import for google-generativeai
import os # For accessing environment variables

logger = logging.getLogger(__name__)

class Decision:
    def __init__(self, tools_list: List[Any], model_name: str = "gemini-2.0-flash"):
        """
        Initializes the Decision module.

        Args:
            tools_list (List[Any]): A list of tool objects available to the agent.
                                    Each tool object should have name, description, and get_schema() method.
            model_name (str): The name of the Generative AI model to use.
        """
        self.model_name = model_name
        try:
            # Explicitly load the API key from environment variable
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.critical("CRITICAL: GEMINI_API_KEY environment variable not set or empty.")
                # You might want to raise an exception here to halt initialization
                # raise ValueError("GOOGLE_API_KEY not found in environment variables.")
                self.llm = None # Ensure llm is None if API key is missing
            else:
                logger.info("GEMINI_API_KEY found. Configuring genai...")
                genai.configure(api_key=api_key)
                self.llm = genai.GenerativeModel(model_name=self.model_name)
                logger.info(f"Decision module initialized with model: {self.model_name}")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize GenerativeModel or configure API key: {e}", exc_info=True)
            self.llm = None # Ensure llm is None if initialization fails

        self.tools_list_for_prompt = self._format_tools_for_prompt(tools_list)
        # self.tools_dict = {tool.name: tool for tool in tools_list} # Not strictly needed in Decision if execution is in Agent

    def _format_tools_for_prompt(self, tools_list: List[Any]) -> str:
        """Formats the list of tools into a string for the LLM prompt."""
        if not tools_list:
            return "No tools available."
        
        formatted_tools = ["Available Tools:"]
        for i, tool_obj in enumerate(tools_list):
            try:
                # Assuming tools have 'name', 'description', and 'get_schema()'
                schema = tool_obj.get_schema() if hasattr(tool_obj, 'get_schema') else {}
                desc = getattr(tool_obj, 'description', "No description.")
                name = getattr(tool_obj, 'name', f"UnnamedTool_{i+1}")
                
                tool_info = f"{i+1}. Name: {name}\n" \
                            f"   Description: {desc}\n" \
                            f"   Input Schema: {json.dumps(schema.get('parameters', {}), indent=2)}"
                formatted_tools.append(tool_info)
            except Exception as e:
                logger.error(f"Error formatting tool {getattr(tool_obj, 'name', 'UnknownTool')}: {e}")
                formatted_tools.append(f"{i+1}. Name: {getattr(tool_obj, 'name', 'UnknownTool')} (Error in schema formatting)")
        return "\n".join(formatted_tools)

    def _get_system_prompt(self) -> str:
        """Constructs the system prompt for the LLM."""
        # This prompt is inspired by the one in ERA-V3-Capstone-Milestone1
        # and adapted for the new memory_actions and structured output.
        return f"""
You are an AI assistant. Your primary goal is to understand the user's query and available context, then decide the best course of action.
This might involve using one of the available tools or providing a direct answer.

Tool Information: {self.tools_list_for_prompt}

You also have access to a memory module that contains:
1.  Conversation History: The ongoing dialogue.
2.  Facts: Key pieces of information stored previously.
3.  Scheme Data: Detailed information about design schemes, including a 'current_scheme_id' if one is active.

Your Decision Process:
1.  Analyze the user's query, the provided 'Perception Data' (if any), 'Memory Context', and 'Conversation History'.
   The 'Perception Data' might contain pre-extracted intents or entities. Use this to inform your decision but prioritize the raw user query if there's a conflict.
2.  If the query is a direct repeat and the answer is in memory, the main agent loop will handle it. Your primary role is to decide the next step if it's not a simple repeat.
3.  Determine if any tool can help achieve the user's goal.
4.  If a tool is needed, specify its name and the exact JSON input arguments based on its schema.
5.  If no tool is needed, or after a tool has been used (in a subsequent call), provide a comprehensive final answer.
6.  Indicate if any information from the conversation or tool output should be explicitly stored or updated in memory.

Output Format:
Provide your response STRICTLY as a JSON object with the following keys:
- "thought": "Your step-by-step reasoning on how you reached the decision, including how you used memory and why you chose a specific tool or decided on a final answer."
- "tool_name": "Name of the tool to use (e.g., 'search_2050_products') or 'final_answer' if no tool is immediately needed or if you are providing the concluding response."
- "tool_input": {{ "arg_name1": "value1", ... }} (JSON object for tool arguments, empty {{}} if no tool or 'final_answer').
- "memory_actions": [ {{ "action": "store_fact", "key": "some_key", "value": "some_value" }}, {{ "action": "set_current_scheme", "scheme_id": "scheme_X" }} ] (Optional: A list of actions to perform on the memory. Valid actions: 'store_fact', 'set_current_scheme', 'update_current_scheme_data').
- "speak": "The final, user-facing response or a message indicating tool usage. If retrieving from memory (though main loop might handle repeats), you can state that."

Example of tool use:
User: Find low emission paints for the current scheme.
Your JSON Output:
{{
  "thought": "The user wants low emission paints. The current scheme details are in memory. I should use the 'search_2050_products' tool with 'paint' as category and specify 'low emissions'.",
  "tool_name": "search_2050_products",
  "tool_input": {{ "product_name": "low emission paint" }},
  "memory_actions": [],
  "speak": "I will search for low emission paints for you."
}}

Example of memory update:
User: I want to finalise scheme 3.
Your JSON Output:
{{
  "thought": "The user wants to finalize scheme 3. I need to update the memory to set scheme 3 as the current active scheme.",
  "tool_name": "final_answer",
  "tool_input": {{}},
  "memory_actions": [{{ "action": "set_current_scheme", "scheme_id": "scheme_3" }}],
  "speak": "Okay, I've marked Scheme 3 as the finalized scheme. What would you like to do next with it?"
}}
"""

    async def execute_plan_with_memory(
        self,
        user_query: str,
        perception_data: Dict[str, Any], 
        memory_context: str,
        full_conversation_history: List[Dict[str, Any]],
        previous_tool_output: Optional[str] = None # For Phase 4
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict], Optional[str], List[Dict[str, Any]]]:
        """
        Makes a decision based on user query, memory, and conversation history.

        Returns:
            Tuple: thought, tool_name, tool_input, speak, memory_actions
        """
        if not self.llm:
            logger.error("LLM not initialized in Decision module. Cannot execute plan. Check API key and initialization logs.")
            return "LLM not initialized.", "final_answer", {}, "Sorry, I'm having trouble thinking right now.", []

        prompt_parts = [self._get_system_prompt()]

        prompt_parts.append("\n--- Pre-processed Information from Perception Module ---")
        prompt_parts.append(f"Perceived Intent: {perception_data.get('intent', 'N/A')}")
        if perception_data.get('entities'):
            prompt_parts.append(f"Extracted Entities by Perception: {json.dumps(perception_data.get('entities'))}")
        if perception_data.get('command'):
            prompt_parts.append(f"Parsed Command by Perception: {json.dumps(perception_data.get('command'))}")
        prompt_parts.append("Note: Use this perception data as a hint, but always prioritize the user's raw query for final understanding.")

        prompt_parts.append("\n--- Current Memory Context ---")
        prompt_parts.append(memory_context if memory_context.strip() else "No specific facts or scheme data currently in active memory.")

        prompt_parts.append("\n--- Recent Conversation History (for context) ---")
        history_for_prompt = []
        for turn in full_conversation_history[-6:]: # Last 3 user/agent pairs
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("final_answer") if turn.get("role") == "agent" and turn.get("final_answer") else turn.get("content", "")
            history_for_prompt.append(f"{role}: {content}")
        prompt_parts.append("\n".join(history_for_prompt) if history_for_prompt else "No prior conversation in this session.")

        if previous_tool_output: # For Phase 4
            prompt_parts.append(f"\n--- Result from Last Tool Execution ---")
            prompt_parts.append(f"Tool output: {previous_tool_output}")
            prompt_parts.append("Based on this tool output and the original query, formulate your final response or decide the next step.")

        prompt_parts.append(f"\n--- User's Current Query ---\nUser: {user_query}")
        prompt_parts.append("\n--- Your JSON Response ---")

        full_prompt = "\n".join(prompt_parts)
        logger.debug(f"Decision LLM Full Prompt (first 500 chars):\n{full_prompt[:500]}...")

        try:
            # Assuming your genai library has an async version or you run sync in executor
            # For google-generativeai, you might use:
            # response = await self.llm.generate_content_async(full_prompt)
            # For now, if it's synchronous, we'd run it in an executor if called from async agent
            # But it's better if the LLM call itself is awaitable.
            # Let's assume self.llm.generate_content can be awaited or is run in executor if sync.
            
            # If self.llm.generate_content is synchronous:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.llm.generate_content, full_prompt) # This line makes the call
            
            llm_response_text = response.candidates[0].content.parts[0].text
            logger.debug(f"LLM Raw Response Text:\n{llm_response_text}")

            # --- Preprocess the LLM response to remove markdown formatting ---
            llm_response_text = llm_response_text.strip()
            
            # Handle markdown code blocks first
            if llm_response_text.startswith("```json"):
                llm_response_text = llm_response_text[6:].strip() # Remove ```json
                if llm_response_text.endswith("```"):
                    llm_response_text = llm_response_text[:-3].strip() # Remove trailing ```
            
            # If not a markdown block, or after stripping markdown,
            # try to find the actual JSON object within the string
            if not llm_response_text.startswith("{"):
                first_brace = llm_response_text.find("{")
                if first_brace != -1:
                    llm_response_text = llm_response_text[first_brace:]
            
            if not llm_response_text.endswith("}"):
                last_brace = llm_response_text.rfind("}")
                if last_brace != -1:
                    llm_response_text = llm_response_text[:last_brace+1]

            parsed_response = json.loads(llm_response_text)
            thought = parsed_response.get("thought", "No thought process provided.")
            tool_name = parsed_response.get("tool_name", "final_answer")
            tool_input = parsed_response.get("tool_input", {})
            speak = parsed_response.get("speak", "I'm not sure how to respond to that.")
            memory_actions = parsed_response.get("memory_actions", [])

            return thought, tool_name, tool_input, speak, memory_actions

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}\nRaw response: {llm_response_text}")
            return f"Error: Could not parse LLM response. Raw: {llm_response_text}", "final_answer", {}, "I'm sorry, I had trouble understanding the response format.", []
        except Exception as e: # Catching the genai API errors more specifically might be useful
            logger.error(f"Error in Decision LLM call: {e}")
            # Log specific Gemini API errors if available
            if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback'):
                 logger.error(f"Prompt Feedback: {e.response.prompt_feedback}")
            return f"Error: An unexpected error occurred: {e}", "final_answer", {}, "I'm sorry, an unexpected error occurred.", []