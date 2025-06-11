import logging
from agent import ConversationalAgent # Assuming agent.py is in the same directory or accessible
from module.utils import setup_logging # Import the setup function
from typing import Dict, Any, Optional # Added for type hinting
from module.mcp_tools import AddMCPTool, SearchDocumentsMCPTool, AiFormSchemerMCPTool, Search2050ProductsMCPTool, MultiplyMCPTool # Import your tool wrappers
# from mcp_tools import SubtractMCPTool, DivideMCPTool # Import other tools as needed

if __name__ == "__main__":
    # Call setup_logging() once at the beginning of your application
    # You can customize the level, e.g., logging.DEBUG for more verbose output
    setup_logging(level=logging.INFO) 
    logger_maintpy = logging.getLogger(__name__)
    # Use the renamed logger to avoid conflicts if tool modules also define 'logger'
    logger_maintpy.info("Starting Conversational Agent Application...")

    # 1. Instantiate your tools (replace placeholders with your actual tool classes)
    # These are now wrappers for tools defined in mcp-server.py
    tool_list = [
        AddMCPTool(),
        SearchDocumentsMCPTool(),
        AiFormSchemerMCPTool(),
        Search2050ProductsMCPTool(), # Corresponds to search_2050_products
        MultiplyMCPTool()
        # Add other wrappers: SubtractMCPTool, DivideMCPTool, etc.
    ]

    agent = ConversationalAgent(tools_list=tool_list)
    print(f"Agent: {agent.start_session()}")
    # Start the session and print the initial agent message
    initial_message = agent.start_session()
    print(f"Agent: {initial_message}")

    try:
        while agent.is_session_active:
            user_input = input("You: ")
            if not user_input.strip():
                continue
            response = agent.process_user_query(user_input)
            print(f"Agent: {response}")
    except KeyboardInterrupt:
        logger_maintpy.info("User interrupted the session.")
        if agent.is_session_active:
            print(f"\nAgent: {agent.end_session()}")
    finally:
        logger_maintpy.info("Conversational Agent Application ended.")