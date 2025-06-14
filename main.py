import logging
from agent import ConversationalAgent # Assuming agent.py is in the same directory or accessible
from module.utils import setup_logging # Import the setup function
from typing import Dict, Any, Optional # Added for type hinting
from dotenv import load_dotenv # For loading environment variables
# Import your MCP tool wrappers
from module.mcp_tool_wrappers import AddMCPToolWrapper, AiFormSchemerMCPToolWrapper, MaterialPropertyFetcherMCPToolWrapper
import asyncio # For running async agent
import json # For the reevaluation query in agent.py
load_dotenv()
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
        AddMCPToolWrapper(),
        AiFormSchemerMCPToolWrapper(),
        MaterialPropertyFetcherMCPToolWrapper()
        # Add other MCPToolWrappers for tools defined in your mcp-server.py
        # e.g., SearchDocumentsMCPToolWrapper(), Search2050ProductsMCPToolWrapper(), MultiplyMCPToolWrapper()
    ]

    agent = ConversationalAgent(tools_list=tool_list)
    initial_message = agent.start_session() # This is synchronous
    print(f"Agent: {initial_message}")

    try:
        while agent.is_session_active:
            user_input = input("You: ")
            if not user_input.strip():
                continue
            # Since process_user_query is now async, run it using asyncio.run()
            response = asyncio.run(agent.process_user_query(user_input))
            print(f"Agent: {response}")
    except KeyboardInterrupt:
        logger_maintpy.info("User interrupted the session.")
        if agent.is_session_active:
            print(f"\nAgent: {agent.end_session()}")
    finally:
        logger_maintpy.info("Conversational Agent Application ended.")