from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union

class SchemeParameters(BaseModel):
    """Parameters for a building scheme"""
    grid_spacing_x: Union[float, str] = Field(description="Grid spacing in X direction (m)")
    grid_spacing_y: Union[float, str] = Field(description="Grid spacing in Y direction (m)")
    extents_x: Union[float, str] = Field(description="Building extent in X direction (m)")
    extents_y: Union[float, str] = Field(description="Building extent in Y direction (m)")
    no_of_floors: Union[int, str] = Field(description="Number of floors")

class SchemeEvaluations(BaseModel):
    """Evaluation metrics for a building scheme"""
    steel_tonnage: Union[float, str] = Field(description="Steel tonnage (tonnes)")
    column_size: Union[str, float] = Field(description="Column size (mm)")
    structural_depth: Union[float, str] = Field(description="Structural depth (mm)")
    concrete_tonnage: Union[float, str] = Field(description="Concrete tonnage (tonnes)")
    total_emissions: Union[float, str] = Field(description="Total emissions (tonnes CO₂e)")

class Scheme(BaseModel):
    """A building scheme with parameters and evaluations"""
    id: int = Field(description="Unique identifier for the scheme")
    name: str = Field(description="Name of the scheme, e.g., 'Simple Rectangular Office'")
    parameters: SchemeParameters
    evaluations: Optional[SchemeEvaluations] = None
    
    # Visualization properties
    width: Optional[float] = None
    height: Optional[float] = None
    depth: Optional[float] = None
    position_x: float = 0
    position_y: float = 0
    position_z: float = 0
    color: str = "#4287f5"  # Default blue color

class SchemeList(BaseModel):
    """List of building schemes"""
    schemes: List[Scheme] = [] 