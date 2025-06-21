import sys
from pathlib import Path

# This client simulates interaction with the mcp_server.
# For simplicity, it directly imports and calls the tool functions.
# In a real-world scenario with separate processes, this would use
# a transport mechanism like HTTP requests to communicate with the server.

# Add the parent directory to the path to allow importing mcp_server
# This is necessary because we are running from the chat_agent subdirectory.
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from mcp_server import (
    add,
    subtract,
    multiply,
    divide,
    search_2050_products,
    search_documents,
    ai_form_schemer,
)
from models import (
    AddInput,
    Search2050ProductsInput,
    AiFormSchemerInput
)

# The ai_form_schemer tool in mcp_server.py has a dependency on a model
# that is created using environment variables. We need to ensure they are loaded.
from dotenv import load_dotenv
load_dotenv()

# The original ai_form_schemer has a dependency on create_structural_surrogate_model
# which is called internally. This means the environment variables for the structural model
# must be set in the environment where this agent code is running.

class MCPClient:
    """
    A client to interact with the tools defined in mcp_server.py.
    """
    def add(self, a: int, b: int) -> int:
        return add(AddInput(a=a, b=b)).result

    def subtract(self, a: int, b: int) -> int:
        return subtract(a=a, b=b)

    def multiply(self, a: float, b: float) -> float:
        return multiply(a=a, b=b)

    def divide(self, a: float, b: float) -> float:
        return divide(a=a, b=b)

    def search_documents(self, query: str) -> list[str]:
        return search_documents(query=query)

    def search_2050_products(self, product_name: str):
        return search_2050_products(Search2050ProductsInput(product_name=product_name))

    def ai_form_schemer(self, grid_spacing_x: int, grid_spacing_y: int, extents_x: int, extents_y: int, no_of_floors: int):
        input_data = AiFormSchemerInput(
            grid_spacing_x=grid_spacing_x,
            grid_spacing_y=grid_spacing_y,
            extents_x=extents_x,
            extents_y=extents_y,
            no_of_floors=no_of_floors
        )
        return ai_form_schemer(input=input_data)

# Instantiate a single client for the application to use
mcp_client = MCPClient()