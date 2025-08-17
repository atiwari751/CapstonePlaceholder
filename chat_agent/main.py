import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables from .env file
load_dotenv()

# Import tools from our custom tools module
from custom_tools import all_tools

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

**Rule 1: Handling Product Alternatives (CRITICAL First Check)**
- **BEFORE using any tool**, you MUST check if the user is asking for 'more', 'other', or 'alternative' products.
- **IF** they are, you MUST NOT use any tool.
- **INSTEAD**, you must search the `chat_history` for a `PRODUCT_DATA` block from a previous turn.
- **THEN**, parse the JSON in that block and present the 2nd and 3rd options from the `product_options` list.
- It is a critical failure to use a tool when the user asks for alternatives and the `PRODUCT_DATA` is in the history.

**Rule 2: Finding Products (Initial Search)**
- **IF** the user asks for a specific product for the first time (e.g., 'paint', 'windows'), and it's not a request for alternatives.
- **THEN** you MUST use the `find_low_emission_product` tool. This tool returns the single best option and stores others in memory.
- **MEMORY CRITICAL:** After the tool runs, your final output that gets saved to `chat_history` MUST contain the full `PRODUCT_DATA` block from the tool's observation.
- You should present a human-friendly summary to the user, but the `PRODUCT_DATA` block must be included in your response to be saved in memory.
- **Example of a good final response:** "The best option is X. Other options are available.
PRODUCT_DATA: {{ ...json... }}"

**Rule 3: Evaluating Schemes**
- **IF** the user asks to 'evaluate', 'compare', or 'analyze' building schemes.
- **THEN** you MUST use the `evaluate_building_schemes` tool.

**Rule 4: Answering Building Questions**
- **IF** the user asks about building options, case studies, or techniques (e.g., "low waste construction").
- **THEN** you MUST use the `search_building_case_studies` tool. Do not use your general knowledge.

**Rule 5: Answering from Memory (General Follow-ups)**
- **IF** the user asks a follow-up question about a scheme that is already in the `chat_history` (and it's not for product alternatives).
- **THEN** your task is to find the answer directly within the `chat_history` and provide it. Do not use any tools for this.
- **Example (Scheme Details):** If the user asks for "details of Square Bay", you must find the "## Square Bay" section in the `chat_history` and extract the "Scheme Inputs" from it to form your answer.

**Rule 6: Calculations**
- **IF** the user asks a simple math question, use the calculator tools.

**Rule 7: Output Presentation**
- When presenting the results from the `evaluate_building_schemes` tool, you MUST show the full, detailed output from the tool. This includes the scheme inputs, tonnage, products used, and all calculated emissions for each scheme. Do not summarize or omit any details from the tool's output.
- Use Markdown for formatting your final answers (e.g., `##` for headers, `-` for lists, `**bold**` for emphasis).
- **CRITICAL FOR MEMORY:** The tool's observation will contain a human-readable summary AND one or more `PRODUCT_DATA` JSON blocks. Your final answer that gets saved to `chat_history` **MUST** include both parts. First, present the human-readable summary. Then, append the complete, unmodified `PRODUCT_DATA` blocks. This is essential for answering follow-up questions about alternatives.
- **Example of a good final response after evaluation:**
"
## Scheme Evaluation
... (human-readable summary of the scheme) ...
**Total Manufacturing Emissions: 19.41 kgCO2e**

PRODUCT_DATA: {{ ...json for steel... }}
PRODUCT_DATA: {{ ...json for concrete... }}
"
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
    # The agent is now stateless. Memory is managed per-session in the API server.
    agent_executor = AgentExecutor(
        agent=agent, tools=all_tools, verbose=True
    )

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
    # For command-line chat, we manage history manually in a list
    chat_history_for_agent = []

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            print("Assistant: Goodbye!")
            break
        
        # Pass the history to the invoke method
        response = agent_executor.invoke({"input": user_input, "chat_history": chat_history_for_agent})
        print(f"\nAssistant:\n{response['output']}")

        # Manually update the history list
        chat_history_for_agent.extend([HumanMessage(content=user_input), AIMessage(content=response["output"])])

if __name__ == "__main__":
    run_chat()