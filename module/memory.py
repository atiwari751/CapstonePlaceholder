# module/memory.py

import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class Memory:
    """
    Manages the agent's memory, including conversation history,
    factual data, and scheme-specific information for a session.
    """
    def __init__(self, session_id: Optional[str] = None):
        """
        Initializes the memory for a new session.

        Args:
            session_id (Optional[str]): An optional existing session ID.
                                        If None, a new one is generated.
        """
        self.session_id: str = session_id if session_id else str(uuid.uuid4())
        self.conversation_history: List[Dict[str, Any]] = []
        self.facts: Dict[str, Any] = {}  # For general key-value storage
        self.schemes_data: Dict[str, Dict[str, Any]] = {}  # To store detailed scheme info, keyed by scheme_id
        self.current_scheme_id: Optional[str] = None
        logger.info(f"Memory initialized for session: {self.session_id}")

    def add_conversation_turn(self,
                              role: str,
                              content: str,
                              tool_name: Optional[str] = None,
                              tool_input: Optional[Dict] = None,
                              tool_output: Optional[Any] = None,
                              final_answer: Optional[str] = None) -> None:
        """
        Adds a turn to the conversation history.

        Args:
            role (str): The role of the speaker (e.g., "user", "agent").
            content (str): The textual content of the turn.
            tool_name (Optional[str]): Name of the tool called, if any.
            tool_input (Optional[Dict]): Input provided to the tool.
            tool_output (Optional[Any]): Output received from the tool.
            final_answer (Optional[str]): The agent's final answer for this turn, if applicable.
        """
        turn: Dict[str, Any] = {"role": role, "content": content}
        if tool_name:
            turn["tool_name"] = tool_name
            turn["tool_input"] = tool_input if tool_input is not None else {}
            turn["tool_output"] = tool_output
        if final_answer:
            turn["final_answer"] = final_answer
        
        self.conversation_history.append(turn)
        logger.debug(f"Session {self.session_id}: Added conversation turn: {turn}")

    def get_conversation_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the conversation history.

        Args:
            last_n (Optional[int]): If provided, returns only the last N turns.

        Returns:
            List[Dict[str, Any]]: The conversation history.
        """
        if last_n:
            return self.conversation_history[-last_n:]
        return self.conversation_history

    def store_fact(self, key: str, value: Any) -> None:
        """
        Stores a general key-value fact.

        Args:
            key (str): The key for the fact.
            value (Any): The value of the fact.
        """
        self.facts[key] = value
        logger.info(f"Session {self.session_id}: Stored fact: '{key}' = '{value}'")

    def retrieve_fact(self, key: str) -> Optional[Any]:
        """
        Retrieves a fact by key.

        Args:
            key (str): The key of the fact to retrieve.

        Returns:
            Optional[Any]: The value of the fact, or None if not found.
        """
        value = self.facts.get(key)
        if value is not None:
            logger.info(f"Session {self.session_id}: Retrieved fact: '{key}' = '{value}'")
        else:
            logger.info(f"Session {self.session_id}: Fact not found: '{key}'")
        return value

    def get_all_facts(self) -> Dict[str, Any]:
        """
        Retrieves all stored general facts.

        Returns:
            Dict[str, Any]: A copy of all stored facts.
        """
        return self.facts.copy()

    def store_scheme_data(self, scheme_id: str, data: Dict[str, Any], overwrite: bool = False) -> None:
        """
        Stores or updates data for a specific scheme.

        Args:
            scheme_id (str): The unique identifier for the scheme.
            data (Dict[str, Any]): The data associated with the scheme.
            overwrite (bool): If True, replaces the entire scheme data.
                              If False (default), updates existing data.
        """
        if scheme_id not in self.schemes_data or overwrite:
            self.schemes_data[scheme_id] = {}
        self.schemes_data[scheme_id].update(data)
        logger.info(f"Session {self.session_id}: Stored/Updated data for scheme '{scheme_id}': {data}")

    def retrieve_scheme_data(self, scheme_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves data for a specific scheme.

        Args:
            scheme_id (str): The ID of the scheme to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The scheme's data, or None if not found.
        """
        data = self.schemes_data.get(scheme_id)
        if data:
            logger.info(f"Session {self.session_id}: Retrieved data for scheme '{scheme_id}'")
        else:
            logger.info(f"Session {self.session_id}: No data found for scheme '{scheme_id}'")
        return data
    
    def get_all_schemes_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieves data for all schemes.

        Returns:
            Dict[str, Dict[str, Any]]: A copy of all schemes' data.
        """
        return self.schemes_data.copy()

    def set_current_scheme(self, scheme_id: str) -> bool:
        """
        Sets the currently active/chosen scheme.
        This also stores the current_scheme_id as a fact.

        Args:
            scheme_id (str): The ID of the scheme to set as current.

        Returns:
            bool: True if the scheme exists and was set, False otherwise.
        """
        # A scheme is considered to exist if it's in schemes_data or has details stored as a fact
        # (e.g., if schemes were generated and stored as facts initially, like scheme_X_summary)
        if scheme_id in self.schemes_data or \
           self.retrieve_fact(f"{scheme_id}_details") is not None or \
           self.retrieve_fact(f"{scheme_id}_summary") is not None:
            self.current_scheme_id = scheme_id
            self.store_fact("current_scheme_id", scheme_id) # Also store as a general fact
            logger.info(f"Session {self.session_id}: Current scheme set to: '{scheme_id}'")
            return True
        logger.warning(f"Session {self.session_id}: Attempted to set current scheme to non-existent scheme: '{scheme_id}'")
        return False

    def get_current_scheme_id(self) -> Optional[str]:
        """
        Gets the ID of the currently active/chosen scheme.
        It prioritizes the instance attribute, then checks facts.

        Returns:
            Optional[str]: The ID of the current scheme, or None.
        """
        if self.current_scheme_id:
            return self.current_scheme_id
        return self.retrieve_fact("current_scheme_id")


    def get_current_scheme_data(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves data for the currently active/chosen scheme.

        Returns:
            Optional[Dict[str, Any]]: Data for the current scheme, or None.
        """
        current_id = self.get_current_scheme_id()
        if current_id:
            return self.retrieve_scheme_data(current_id)
        return None

    def update_current_scheme_data(self, data_to_update: Dict[str, Any]) -> bool:
        """
        Updates data for the currently active/chosen scheme.

        Args:
            data_to_update (Dict[str, Any]): The data to add or update for the current scheme.

        Returns:
            bool: True if updated, False if no current scheme is set.
        """
        current_id = self.get_current_scheme_id()
        if current_id:
            self.store_scheme_data(current_id, data_to_update)
            return True
        logger.warning(f"Session {self.session_id}: No current scheme set. Cannot update scheme data.")
        return False

    def check_for_repeated_question(self, user_query: str, history_lookback: int = 5) -> Optional[str]:
        """
        Checks if an identical question was asked recently by the user and returns its final answer.
        This is a basic exact match implementation.

        Args:
            user_query (str): The user's current query.
            history_lookback (int): How many past *user* turns to check.

        Returns:
            Optional[str]: The previously given final answer if a repeat is found, else None.
        """
        normalized_query = user_query.strip().lower()
        
        user_turns_checked = 0
        # Iterate backwards through the conversation history
        for i in range(len(self.conversation_history) - 1, -1, -1):
            turn = self.conversation_history[i]
            if turn.get("role") == "user":
                user_turns_checked += 1
                if user_turns_checked > history_lookback:
                    break # Stop if we've checked enough user turns

                if turn.get("content", "").strip().lower() == normalized_query:
                    # Found a matching user query. Look for the next agent's final answer.
                    for j in range(i + 1, len(self.conversation_history)):
                        next_turn = self.conversation_history[j]
                        if next_turn.get("role") == "agent" and "final_answer" in next_turn:
                            logger.info(f"Session {self.session_id}: Repeated question found. Retrieving answer from history: '{next_turn['final_answer']}'")
                            return next_turn["final_answer"]
                        # If another user turn comes before an agent's final answer for *this* specific repeated question,
                        # then it's not a direct answer to the question we matched.
                        if next_turn.get("role") == "user" and j > i:
                            break 
        return None


    def get_relevant_context_for_decision(self, max_context_str_length: int = 4000) -> str:
        """
        Constructs a string of relevant context from memory to pass to the Decision LLM.
        This aims to be concise and prioritize important information.

        Args:
            max_context_str_length (int): Approximate maximum length of the context string.

        Returns:
            str: A string containing relevant context.
        """
        context_parts: List[str] = []
        current_length = 0

        def _add_to_context(text: str) -> bool:
            nonlocal current_length
            if not text:
                return False
            if current_length + len(text) < max_context_str_length:
                context_parts.append(text)
                current_length += len(text)
                return True
            logger.warning(f"Session {self.session_id}: Context length limit ({max_context_str_length}) reached while trying to add: {text[:50]}...")
            return False

        # 1. Current Scheme Information (if any)
        current_scheme_id = self.get_current_scheme_id()
        if current_scheme_id:
            scheme_data = self.get_current_scheme_data()
            if scheme_data:
                scheme_str = f"Current Active Scheme (ID: {current_scheme_id}):\n"
                for k, v_item in scheme_data.items():
                    scheme_str += f"  - {k}: {str(v_item)}\n" # Ensure value is string
                _add_to_context(scheme_str + "\n")
            else: 
                 _add_to_context(f"Current Active Scheme ID: {current_scheme_id} (Details may be in facts or forthcoming).\n\n")


        # 2. Key Facts (selective)
        if self.facts:
            facts_header = "Key Information from Memory (Facts):\n"
            
            # Prioritize facts related to the current scheme or general project parameters
            priority_keys = [k for k in self.facts if "scheme" in k.lower() or "building" in k.lower() or "project" in k.lower() or "material" in k.lower() or "current" in k.lower()]
            other_keys = [k for k in self.facts if k not in priority_keys]
            
            sorted_fact_keys = priority_keys + other_keys

            temp_facts_str = ""
            for key in sorted_fact_keys:
                value = self.facts[key]
                fact_line = f"- {key}: {str(value)}\n" # Ensure value is string
                # Check if adding this fact (plus header if not yet added) exceeds 75% of remaining space for facts
                projected_len = len(facts_header) if not temp_facts_str else 0 # Add header len only once
                projected_len += len(temp_facts_str) + len(fact_line)
                
                if current_length + projected_len < max_context_str_length * 0.75: # Reserve some space for history
                    temp_facts_str += fact_line
                else:
                    logger.debug(f"Session {self.session_id}: Fact context limit reached while adding fact: {key}")
                    break 
            
            if temp_facts_str:
                 _add_to_context(facts_header + temp_facts_str + "\n")


        # 3. Recent Conversation History (summary of last few turns)
        history_header = "Recent Conversation Snippets (User/Agent):\n"
        turns_to_include = 0
        max_history_turns = 5 # Max number of user/agent pairs (i.e., 10 individual turns)

        temp_history_str = ""
        # Iterate from most recent backwards
        for turn in reversed(self.conversation_history):
            if turns_to_include >= max_history_turns * 2: 
                break
            
            turn_prefix = "User: " if turn.get("role") == "user" else "Agent: "
            content = turn.get("content","")
            if turn.get("role") == "agent" and turn.get("final_answer"):
                content = turn.get("final_answer") 
            elif turn.get("role") == "agent" and turn.get("tool_name"):
                content += f" (Used tool: {turn['tool_name']})"

            content_str = str(content) # Ensure content is string
            snippet = f"{turn_prefix}{content_str}\n"
            
            projected_len = len(history_header) if not temp_history_str else 0
            projected_len += len(snippet) + len(temp_history_str)

            if current_length + projected_len < max_context_str_length:
                temp_history_str = snippet + temp_history_str # Prepend
                turns_to_include +=1
            else:
                logger.debug(f"Session {self.session_id}: History context limit reached while adding turn: {turn_prefix}{content_str[:30]}...")
                break
        
        if temp_history_str:
            _add_to_context(history_header + temp_history_str)

        full_context = "".join(context_parts)
        logger.info(f"Session {self.session_id}: Generated context for decision (length {len(full_context)}): {full_context[:500]}...")
        return full_context


    def clear(self) -> None:
        """Clears all memory for the current session."""
        self.conversation_history = []
        self.facts = {}
        self.schemes_data = {}
        self.current_scheme_id = None
        logger.info(f"Memory cleared for session: {self.session_id}")

    def store_generated_schemes(self, schemes: List[Dict[str, Any]]) -> List[str]:
        """
        Stores a list of generated schemes. Each scheme is stored individually
        in schemes_data and a summary/reference in facts.

        Args:
            schemes (List[Dict[str, Any]]): A list of scheme dictionaries.

        Returns:
            List[str]: A list of the generated scheme IDs (e.g., "scheme_1", "scheme_2").
        """
        scheme_ids = []
        for i, scheme_detail in enumerate(schemes):
            if not isinstance(scheme_detail, dict):
                logger.error(f"Session {self.session_id}: Scheme detail at index {i} is not a dict: {scheme_detail}. Skipping.")
                continue

            scheme_id = f"scheme_{i+1}" 
            self.store_scheme_data(scheme_id, scheme_detail, overwrite=True)
            
            summary = scheme_detail.get("name", f"Scheme {i+1} - {scheme_detail.get('description', 'No description')[:50]}")
            self.store_fact(f"{scheme_id}_summary", summary)
            scheme_ids.append(scheme_id)

        self.store_fact("generated_scheme_ids", scheme_ids)
        logger.info(f"Session {self.session_id}: Stored {len(scheme_ids)} generated schemes with IDs: {scheme_ids}")
        return scheme_ids

    def retrieve_generated_schemes_summary(self) -> Optional[List[str]]:
        """Retrieves the list of generated scheme IDs stored as a fact."""
        return self.retrieve_fact("generated_scheme_ids")

    def retrieve_all_generated_schemes_details(self) -> Optional[List[Dict[str, Any]]]:
        """
        Retrieves the full details of all generated schemes by looking up each
        scheme_id from the 'generated_scheme_ids' fact.
        """
        scheme_ids = self.retrieve_fact("generated_scheme_ids")
        if not scheme_ids or not isinstance(scheme_ids, list):
            logger.info(f"Session {self.session_id}: No 'generated_scheme_ids' fact found or it's not a list.")
            return None
        
        all_schemes_details = []
        for scheme_id in scheme_ids:
            scheme_data = self.retrieve_scheme_data(scheme_id)
            if scheme_data:
                all_schemes_details.append(scheme_data)
            else:
                logger.warning(f"Session {self.session_id}: Could not retrieve details for scheme_id '{scheme_id}' which was listed in 'generated_scheme_ids'.")
        
        return all_schemes_details if all_schemes_details else None


