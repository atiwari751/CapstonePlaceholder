from typing import Dict, List, Optional, Any, Union
import json
import os
import random
from scheme_models import Scheme, SchemeParameters, SchemeEvaluations, SchemeList

# Default colors for schemes
SCHEME_COLORS = [
    "#ff4040",  # Red
    "#40ff40",  # Green
    "#4040ff",  # Blue
    "#ffff40",  # Yellow
    "#ff40ff",  # Purple
    "#40ffff",  # Cyan
    "#ff8040",  # Orange
    "#40ff80",  # Mint
    "#8040ff",  # Violet
    "#ff4080",  # Pink
]

class SchemeService:
    """Service to manage building schemes"""
    
    def __init__(self):
        self.schemes: List[Scheme] = []
    
    def _load_schemes(self) -> None:
        """Load schemes - now just initializes an empty list"""
        self.schemes = []
    
    def _save_schemes(self) -> None:
        """Save schemes - now a no-op since we're not persisting to disk"""
        pass
    
    def get_schemes(self) -> List[Scheme]:
        """Get all schemes"""
        return self.schemes
    
    def clear_schemes(self) -> None:
        """Clear all schemes"""
        self.schemes = []
    
    def create_scheme_from_agent_data(self, agent_data: Dict[str, Any]) -> Scheme:
        """Create a scheme from agent data"""
        # Extract parameters from agent data
        try:
            # Generate a new ID (max existing ID + 1 or 1 if no schemes exist)
            new_id = max([s.id for s in self.schemes], default=0) + 1
            
            # Extract parameters
            parameters = SchemeParameters(
                grid_spacing_x=self._extract_value(agent_data, "grid_spacing_x", "6"),
                grid_spacing_y=self._extract_value(agent_data, "grid_spacing_y", "7"),
                extents_x=self._extract_value(agent_data, "extents_x", "30"),
                extents_y=self._extract_value(agent_data, "extents_y", "24"),
                no_of_floors=self._extract_value(agent_data, "no_of_floors", "3")
            )
            
            # Extract evaluations if available
            evaluations = None
            if any(k in agent_data for k in ["steel_tonnage", "column_size", "structural_depth"]):
                evaluations = SchemeEvaluations(
                    steel_tonnage=self._extract_value(agent_data, "steel_tonnage", "unknown"),
                    column_size=self._extract_value(agent_data, "column_size", "unknown"),
                    structural_depth=self._extract_value(agent_data, "structural_depth", "unknown"),
                    concrete_tonnage=self._extract_value(agent_data, "concrete_tonnage", "unknown"),
                    total_emissions=self._extract_value(agent_data, "total_emissions", "unknown")
                )
            
            # Choose a color
            color = SCHEME_COLORS[new_id % len(SCHEME_COLORS)]
            
            # Calculate dimensions based on parameters - use actual values without scaling down
            # This matches the requested behavior:
            # - length of cuboid = extents_x
            # - width of cuboid = extents_y
            # - height of cuboid = no_of_floors x 3
            extents_x = self._to_float(parameters.extents_x)
            extents_y = self._to_float(parameters.extents_y)
            floors = self._to_int(parameters.no_of_floors)
            
            # Create the scheme with proper dimensions
            scheme = Scheme(
                id=new_id,
                parameters=parameters,
                evaluations=evaluations,
                width=extents_x,  # Use actual extents_x value
                depth=extents_y,  # Use actual extents_y value
                height=floors * 3,  # Height = floors * 3 meters per floor
                color=color
            )
            
            # Debug log
            print(f"Created scheme {new_id} with dimensions: {extents_x}x{extents_y}x{floors*3}")
            
            return scheme
        
        except Exception as e:
            print(f"Error creating scheme from agent data: {e}")
            # Return a default scheme as fallback
            return self._create_default_scheme()
    
    def _to_float(self, value: Union[str, float, int]) -> float:
        """Convert a value to float"""
        if isinstance(value, (float, int)):
            return float(value)
        
        if isinstance(value, str):
            # Remove 'm' suffix if present
            value = value.replace('m', '').strip()
            try:
                return float(value)
            except ValueError:
                pass
        
        return 1.0  # Default value
    
    def _to_int(self, value: Union[str, int, float]) -> int:
        """Convert a value to int"""
        if isinstance(value, int):
            return value
        
        if isinstance(value, float):
            return int(value)
        
        if isinstance(value, str):
            # Remove any non-numeric characters
            value = ''.join(c for c in value if c.isdigit() or c == '.')
            try:
                # First try as int
                return int(float(value))
            except ValueError:
                pass
        
        return 1  # Default value
    
    def _extract_value(self, data: Dict[str, Any], key: str, default: str) -> Union[str, float, int]:
        """Extract a value from agent data with proper type conversion"""
        # Check for direct key match
        if key in data:
            value = data[key]
        else:
            # Check for case-insensitive match or alternative names
            alt_keys = {
                "extents_x": ["width", "building_width", "x_extent", "x_dimension"],
                "extents_y": ["depth", "building_depth", "y_extent", "y_dimension"],
                "grid_spacing_x": ["x_grid", "grid_x", "column_spacing_x"],
                "grid_spacing_y": ["y_grid", "grid_y", "column_spacing_y"],
                "no_of_floors": ["floors", "number_of_floors", "stories", "storeys"]
            }
            
            # Try alternative keys
            found = False
            if key in alt_keys:
                for alt_key in alt_keys[key]:
                    if alt_key in data:
                        value = data[alt_key]
                        found = True
                        break
            
            # Try case-insensitive match
            if not found:
                for k in data:
                    if k.lower() == key.lower():
                        value = data[k]
                        found = True
                        break
            
            # Use default if not found
            if not found:
                return default
        
        # If value is None or empty, return default
        if value is None or (isinstance(value, str) and not value.strip()):
            return default
        
        # If value is already a number, return it
        if isinstance(value, (int, float)):
            return value
        
        # Try to convert string to number
        try:
            # Check if it's an integer
            if isinstance(value, str):
                # Remove any non-numeric characters for better parsing
                clean_value = ''.join(c for c in value if c.isdigit() or c == '.')
                if clean_value.isdigit():
                    return int(clean_value)
                
                # Check if it's a float
                float_val = float(clean_value)
                return float_val
        except:
            # Return as string if conversion fails
            return str(value)
        
        return str(value)
    
    def _create_default_scheme(self) -> Scheme:
        """Create a default scheme when agent data is invalid"""
        new_id = max([s.id for s in self.schemes], default=0) + 1
        
        parameters = SchemeParameters(
            grid_spacing_x=6.0,
            grid_spacing_y=7.0,
            extents_x=30.0,
            extents_y=24.0,
            no_of_floors=3
        )
        
        evaluations = SchemeEvaluations(
            steel_tonnage="unknown",
            column_size="unknown",
            structural_depth="unknown",
            concrete_tonnage="unknown",
            total_emissions="unknown"
        )
        
        color = SCHEME_COLORS[new_id % len(SCHEME_COLORS)]
        
        return Scheme(
            id=new_id,
            parameters=parameters,
            evaluations=evaluations,
            width=30.0,  # Use actual extents_x value
            depth=24.0,  # Use actual extents_y value
            height=9.0,  # 3 floors * 3 meters per floor
            color=color
        )
    
    def add_scheme(self, scheme: Scheme) -> Scheme:
        """Add a scheme to the collection"""
        self.schemes.append(scheme)
        return scheme
    
    def add_schemes_from_agent_results(self, agent_results: Dict[str, Any]) -> List[Scheme]:
        """Process agent results and extract schemes"""
        new_schemes = []
        
        # Look for scheme data in agent results
        for key, result in agent_results.items():
            if not isinstance(result, dict):
                continue
                
            # Check if this result contains scheme data
            if "result" in result and isinstance(result["result"], str):
                try:
                    # Try to parse JSON from the result
                    result_str = result["result"]
                    # Find JSON-like content in the string
                    json_start = result_str.find("{")
                    json_end = result_str.rfind("}")
                    if json_start >= 0 and json_end > json_start:
                        json_str = result_str[json_start:json_end+1]
                        data = json.loads(json_str)
                        
                        # Check if it's a list of schemes
                        if isinstance(data, list):
                            for scheme_data in data:
                                scheme = self.create_scheme_from_agent_data(scheme_data)
                                self.add_scheme(scheme)
                                new_schemes.append(scheme)
                        # Check if it's a single scheme
                        elif isinstance(data, dict):
                            scheme = self.create_scheme_from_agent_data(data)
                            self.add_scheme(scheme)
                            new_schemes.append(scheme)
                except Exception as e:
                    print(f"Error processing result for schemes: {e}")
                    # Not JSON or not scheme data, skip
                    pass
        
        return new_schemes
    
    def update_scheme(self, scheme_id: int, updates: Dict[str, Any]) -> Optional[Scheme]:
        """Update an existing scheme"""
        for i, scheme in enumerate(self.schemes):
            if scheme.id == scheme_id:
                # Update parameters
                if "parameters" in updates:
                    for key, value in updates["parameters"].items():
                        if hasattr(scheme.parameters, key):
                            setattr(scheme.parameters, key, value)
                
                # Update or create evaluations
                if "evaluations" in updates:
                    if scheme.evaluations is None:
                        scheme.evaluations = SchemeEvaluations(**updates["evaluations"])
                    else:
                        for key, value in updates["evaluations"].items():
                            if hasattr(scheme.evaluations, key):
                                setattr(scheme.evaluations, key, value)
                
                # Update visualization properties
                for key in ["position_x", "position_y", "position_z", "color", "width", "height", "depth"]:
                    if key in updates:
                        setattr(scheme, key, updates[key])
                
                return scheme
        
        return None

# Create a singleton instance
scheme_service = SchemeService() 