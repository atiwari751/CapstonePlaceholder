import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Perception:
    def __init__(self):
        logger.info("Perception module initialized.")
        # Regex to find "scheme" followed by a number, e.g., "scheme 3", "scheme3"
        self.scheme_pattern = re.compile(r"scheme\s*(\d+)", re.IGNORECASE)
        # Regex for finalizing a scheme, e.g., "finalize scheme 3", "choose scheme 2"
        self.finalize_pattern = re.compile(r"(finali[s|z]e|choose|select|pick)\s+scheme\s*(\d+)", re.IGNORECASE)
        # Keywords that might indicate the user wants to end the session
        self.end_session_keywords = ["that is all, thank you", "exit", "quit", "bye", "goodbye"]

    def process_user_input(self, user_query: str) -> Dict[str, Any]:
        """
        Processes the raw user input, performing basic intent and entity extraction.

        Args:
            user_query (str): The raw text input from the user.

        Returns:
            Dict[str, Any]: A dictionary containing:
                            - "original_query": The user_query.
                            - "normalized_query": Lowercased, stripped query.
                            - "intent": A detected simple intent (e.g., "end_session", 
                                        "finalize_scheme", "general_query").
                            - "entities": A dict of extracted entities (e.g., {"scheme_id": "scheme_3"}).
                            - "command": A structured command if a specific actionable intent is found.
        """
        logger.debug(f"Perception processing query: {user_query}")
        
        normalized_query = user_query.strip().lower()
        detected_intent = "general_query" # Default intent
        extracted_entities: Dict[str, Any] = {}
        detected_command: Optional[Dict[str, Any]] = None

        # 1. Check for end session intent (already handled in agent, but good for perception to be aware)
        for keyword in self.end_session_keywords:
            if keyword in normalized_query:
                detected_intent = "end_session"
                break
        
        if detected_intent != "end_session":
            # 2. Check for "finalize scheme X" command/intent
            finalize_match = self.finalize_pattern.search(user_query)
            if finalize_match:
                detected_intent = "finalize_scheme"
                scheme_number = finalize_match.group(2)
                extracted_entities["scheme_id_number"] = int(scheme_number) # Store as number
                extracted_entities["scheme_id_string"] = f"scheme_{scheme_number}" # Store as formatted string
                detected_command = {
                    "type": "finalize_scheme",
                    "scheme_id": f"scheme_{scheme_number}" 
                }
            # Add more intent/entity rules here as needed (e.g., extracting material names, locations)

        return {
            "original_query": user_query,
            "normalized_query": normalized_query,
            "intent": detected_intent,
            "entities": extracted_entities,
            "command": detected_command
        }
