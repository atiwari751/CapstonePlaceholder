from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os

from client import mcp_client


# Tool 1: Search documents
@tool
def search_building_case_studies(query: str) -> str:
    """
    Searches internal documents for case studies and information on building options,
    materials, or construction techniques, including topics like waste management.
    You MUST use this tool to answer any questions about building examples, case studies,
    or specific construction methods. Do not answer from your own knowledge.
    """
    results = mcp_client.search_documents(query)
    if not results or "ERROR" in results[0]:
        return f"Could not find information for '{query}'."
    return "\n".join(results)

# Tool 2: Evaluate building schemes (the complex workflow)
class BuildingSchemeInput(BaseModel):
    description: str = Field(..., description="A brief description of the building to be evaluated, e.g., 'a 10-story office building with a regular grid'.")
    number_of_schemes: int = Field(2, description="The number of building schemes to generate and compare. Defaults to 2.")

@tool(args_schema=BuildingSchemeInput)
def evaluate_building_schemes(description: str, number_of_schemes: int = 2) -> str:
    """
    Performs a full structural and environmental evaluation of multiple building schemes.
    It generates a specified number of plausible structural schemes, calculates their steel and concrete tonnage,
    finds the lowest-emission materials from the 2050 Materials database,
    and calculates the total manufacturing emissions for each scheme.
    Use this tool when a user asks to 'evaluate', 'compare', or 'analyze' building schemes.
    """
    # a. Generate building schemes dynamically using an LLM
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.3,  # Allow for some creativity in scheme generation
            model_kwargs={"response_mime_type": "application/json"},
        )

        generation_prompt = f"""
        Based on the following building description, generate {number_of_schemes} distinct and plausible structural schemes.
        Description: "{description}"
        
        For each scheme, provide a flat JSON object containing a descriptive 'name' and the following integer parameters:
        - 'grid_spacing_x' (meters, typically between 5 and 15)
        - 'grid_spacing_y' (meters, typically between 5 and 15)
        - 'extents_x' (total building width in meters, must be a multiple of grid_spacing_x and less than 50)
        - 'extents_y' (total building length in meters, must be a multiple of grid_spacing_y and less than 50)
        - 'no_of_floors' (number of stories)
        These parameters should be strictly integers.
        Return ONLY a valid JSON object with a single key "schemes" which is a list of these {number_of_schemes} flat scheme objects. Do not include a nested 'inputs' object. Do not include ```json``` markers or any other text.
        """
        
        response = llm.invoke(generation_prompt)
        schemes_data = json.loads(response.content)
        schemes = schemes_data["schemes"]
        if not isinstance(schemes, list):
            raise ValueError("LLM did not return a list of schemes.")
    except (json.JSONDecodeError, KeyError, ValueError) as e:  # Specific exceptions
        return f"Failed to generate or parse valid building schemes from the LLM. Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred while generating schemes: {e}"

    scheme_results = []

    for scheme in schemes:
        try:
            # Prepare inputs for the schemer tool by extracting numeric parameters.
            # This is more robust than relying on the LLM to create a nested 'inputs' object.
            schemer_inputs = {k: v for k, v in scheme.items() if k != 'name'}

            # a. Use ai_form_schemer to fetch tonnage data
            tonnage_data = mcp_client.ai_form_schemer(**schemer_inputs)
            if not tonnage_data.trustworthy:
                print(f"Warning: Results for {scheme['name']} may not be trustworthy.")

            scheme['inputs'] = schemer_inputs # Store inputs for the final report
            scheme['steel_tonnage'] = tonnage_data.steel_tonnage 
            scheme['concrete_tonnage'] = tonnage_data.concrete_tonnage
            scheme_results.append(scheme)
        except Exception as e:  # Catch-all for issues with a specific scheme
            return f"Error evaluating {scheme['name']} with AI Form Schemer: {e}"

    # b. Fetch low-emission structural steel and concrete products
    try:
        steel_products_output = mcp_client.search_2050_products("structural steel")
        concrete_products_output = mcp_client.search_2050_products("concrete")

        # Find products with the lowest manufacturing emissions
        lowest_emission_steel = min(steel_products_output.products, key=lambda p: p.manufacturing_emissions if p.manufacturing_emissions is not None else float('inf'))
        lowest_emission_concrete = min(concrete_products_output.products, key=lambda p: p.manufacturing_emissions if p.manufacturing_emissions is not None else float('inf'))

    except Exception as e:
        return f"Error searching for materials in 2050 Materials database: {e}"

    # c. & d. Calculate emissions and provide output
    output_lines = [f"Evaluation based on user request: '{description}'\n"]
    output_data = {"schemes": []}  # Initialize a dictionary to store scheme data
    output_lines.append("\n--- Scheme Comparison ---")
    output_lines.append(f"Note: The following calculations use the lowest-emission structural steel and concrete products found in the database.")
    
    for scheme in scheme_results:
        total_steel_emissions = scheme['steel_tonnage'] * lowest_emission_steel.manufacturing_emissions
        total_concrete_emissions = scheme['concrete_tonnage'] * lowest_emission_concrete.manufacturing_emissions
        total_emissions = total_steel_emissions + total_concrete_emissions

        output_lines.append(f"\n## {scheme['name']}")
        output_lines.append(f"   - Scheme Inputs:")
        for key, value in scheme.get('inputs', {}).items():
            output_lines.append(f"     - {key.replace('_', ' ').title()}: {value}")
        output_lines.append(f"   - Steel Tonnage: {scheme['steel_tonnage']:.2f} kg/m²")
        output_lines.append(f"   - Concrete Tonnage: {scheme['concrete_tonnage']:.2f} kg/m²")
        output_lines.append(f"   - Using Steel Product: '{lowest_emission_steel.name}' from {lowest_emission_steel.manufacturing_country} ({lowest_emission_steel.manufacturing_emissions} kgCO2e/{lowest_emission_steel.declared_unit})")
        output_lines.append(f"   - Using Concrete Product: '{lowest_emission_concrete.name}' from {lowest_emission_concrete.manufacturing_country} ({lowest_emission_concrete.manufacturing_emissions} kgCO2e/{lowest_emission_concrete.declared_unit})")
        output_lines.append(f"   - Calculated Steel Emissions: {total_steel_emissions:,.2f} kgCO2e/m²")
        output_lines.append(f"   - Calculated Concrete Emissions: {total_concrete_emissions:,.2f} kgCO2e/m²")
        output_lines.append(f"   - **Total Manufacturing Emissions: {total_emissions:,.2f} kgCO2e/m²")

        # Store scheme data for memory retrieval
        output_data["schemes"].append(scheme)

    # Include the structured data in the output string for memory
    return "\n".join(output_lines) + f"\n\nSCHEME_DATA: {json.dumps(output_data)}"


# Tool 3: Find specific products (like paint)
class ProductSearchInput(BaseModel):
    product_type: str = Field(..., description="The type of product to search for, e.g., 'paint', 'insulation', 'cladding'.")

@tool(args_schema=ProductSearchInput)
def find_low_emission_product(product_type: str) -> str:
    """
    Searches the 2050 Materials database for a specific type of product
    and finds the top 3 options with the lowest manufacturing emissions.
    It returns the single best option to the user and stores all 3 in a hidden data block for follow-up questions.
    Use this for initial questions about specific materials like paint, windows, etc.
    """
    try:
        products_output = mcp_client.search_2050_products(product_type)
        if not products_output.products:
            return f"No products found for '{product_type}'."

        valid_products = [p for p in products_output.products if p.manufacturing_emissions is not None]
        sorted_products = sorted(valid_products, key=lambda p: p.manufacturing_emissions)

        if not sorted_products:
            return f"Found products for '{product_type}', but none had manufacturing emission data."

        top_products = sorted_products[:3]  # Get top 3

        if not top_products:
            return f"No low-emission products found for '{product_type}'."

        # Prepare the response for the user (only the best one)
        best_product = top_products[0]
        user_response = (
            f"The lowest emission product for '{product_type}' is '{best_product.name}' "
            f"from {best_product.city}, {best_product.manufacturing_country} "
            f"with emissions of {best_product.manufacturing_emissions} kgCO2e/{best_product.declared_unit}. "
            "Other options are available if you'd like to see them."
        )

        # Prepare the hidden data block for memory
        product_data_for_memory = [p.dict() for p in top_products]
        hidden_data = {"product_options": product_data_for_memory}

        return user_response + f"\n\nPRODUCT_DATA: {json.dumps(hidden_data)}"

    except Exception as e:
        return f"Error searching for product '{product_type}': {e}"


# Tool 4: Calculator tools
@tool
def add(a: int, b: int) -> int:
    """Adds two integers."""
    return mcp_client.add(a, b)

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts two integers."""
    return mcp_client.subtract(a, b)

@tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return mcp_client.multiply(a, b)

@tool
def divide(a: float, b: float) -> float:
    """Divides two numbers. Cannot divide by zero."""
    if b == 0:
        return "Error: Cannot divide by zero."
    return mcp_client.divide(a, b)

# Consolidate all tools into a list
all_tools = [
    search_building_case_studies,
    evaluate_building_schemes,
    find_low_emission_product,
    add,
    subtract,
    multiply,
    divide,
]