import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables from .env file
load_dotenv()

# Import tools from our custom tools module
from .custom_tools import all_tools

def create_agent_executor():
    """
    Creates and returns the LangChain agent executor.
    """
    # 1. Set up the LLM
    # Ensure GEMINI_API_KEY is set in your .env file
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )

    # 2. Define the prompt
    # This prompt guides the agent on how to use the tools correctly based on the user's query.
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant for sustainable building design. Your memory of the conversation is provided in the chat history.
You have access to a set of tools to answer user questions.
Your primary functions are:
1.  **Answering General Questions**: For general questions about sustainable building options or techniques, use the `search_sustainable_building_options` tool to consult internal documents.
2.  **Evaluating Building Schemes**: When asked to 'evaluate', 'compare', or 'analyze' building schemes, you must use the `evaluate_building_schemes` tool. This tool performs a complete analysis and is the only way to get tonnage and emissions data.
3.  **Finding Specific Products**: For questions about specific products like 'paint' or 'windows', use the `find_low_emission_product` tool.
4.  **Handling Follow-up Questions**: After you have evaluated building schemes (e.g., "Scheme A"), the user might ask a follow-up question like "Find paint for Scheme A". You should use your memory of the conversation to understand the context, and then use the `find_low_emission_product` tool with the product type 'paint'. The scheme name ('Scheme A') provides context but is not a direct input for your tools.
5.  **Calculations**: For simple math, use the calculator tools (add, subtract, multiply, divide).

Always provide the final answer to the user in a clear, well-formatted way.
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # 3. Create the agent
    # Note: `create_openai_tools_agent` is a generic function that works with any LLM
    # that supports the OpenAI tools-calling interface, including Gemini.
    agent = create_openai_tools_agent(llm, all_tools, prompt_template)

    # 4. Create the Agent Executor
    agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)

    return agent_executor

def run_chat():
    """
    Runs an interactive command-line chat session with the agent.
    """
    print("Starting Sustainable Building Design Assistant...")
    print("Type 'exit' to end the conversation.")
    
    required_vars = [
        'DEVELOPER_TOKEN', 'API_URL', 'API_ENDPOINT_NAME',
        'AZURE_CLIENT_ID', 'AZURE_CLIENT_SECRET', 'AZURE_SCOPE'
    ]
    if any(not os.getenv(v) for v in required_vars):
        print("\nWARNING: Not all mcp_server tool environment variables are set.")
        print("The 'evaluate_building_schemes' and product search tools may fail.")
        print(f"Please ensure {', '.join(required_vars)} are in your .env file.\n")

    agent_executor = create_agent_executor()
    chat_history = []

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            print("Assistant: Goodbye!")
            break

        response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
        print(f"\nAssistant:\n{response['output']}")

        chat_history.extend([HumanMessage(content=user_input), AIMessage(content=response["output"])])

if __name__ == "__main__":
    run_chat()