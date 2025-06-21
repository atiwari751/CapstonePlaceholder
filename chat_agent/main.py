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
                """You are a helpful assistant for sustainable building design.
Your memory of the conversation is provided in the `chat_history`.

**Your Core Task is to use tools to answer questions. Follow these rules strictly:**

**Rule 1: Answering Building Questions**
- **IF** the user asks about building options, case studies, or techniques (e.g., "low waste construction").
- **THEN** you MUST use the `search_building_case_studies` tool. Do not use your general knowledge.

**Rule 2: Evaluating Schemes**
- **IF** the user asks to 'evaluate', 'compare', or 'analyze' building schemes.
- **THEN** you MUST use the `evaluate_building_schemes` tool.

**Rule 3: Finding Products**
- **IF** the user asks for specific products (e.g., 'paint', 'windows'), use the `find_low_emission_product` tool.
- **IF** the user asks for 'more' or 'alternatives' for a product, you MUST find the previous `find_low_emission_product` call in the `chat_history`, and call it again with the same `product_type` and an incremented `page` number.

**Rule 4: Answering from Memory (CRITICAL RULE)**
- **IF** the user asks a follow-up question about a scheme, product, or result that is already in the `chat_history`.
- **THEN** your task is to find the answer directly within the `chat_history` and provide it. Do not use any tools for this unless a new calculation is required.
- **Example:** If the user asks for "details of Square Bay", you must find the "## Square Bay" section in the `chat_history` and extract the "Scheme Inputs" from it to form your answer.
- **MANDATORY:** It is a critical failure to use a tool or claim information is unavailable when the answer is already in the `chat_history`.

**Rule 5: Calculations**
- **IF** the user asks a simple math question, use the calculator tools.

**Rule 6: Output Presentation**
- When presenting the results from the `evaluate_building_schemes` tool, you MUST show the full, detailed output from the tool. This includes the scheme inputs, tonnage, products used, and all calculated emissions for each scheme. Do not summarize or omit any details from the tool's output.

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