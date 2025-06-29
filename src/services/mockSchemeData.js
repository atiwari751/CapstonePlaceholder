// Mock data for building schemes
const mockSchemes = [
  {
    id: 1,
    parameters: {
      grid_spacing_x: 8.0,
      grid_spacing_y: 8.0,
      x_extents: [0, 5],
      y_extents: [0, 5],
      num_floors: 5,
      floor_height: 3.5
    },
    evaluations: {
      steel_tonnage: 450,
      column_size: "UC305x305x97",
      beam_size: "UB457x191x82",
      carbon_footprint: 650,
      cost_per_sqm: 2500
    },
    width: 5,
    height: 5,
    depth: 5,
    color: "#4287f5"
  },
  {
    id: 2,
    parameters: {
      grid_spacing_x: 7.5,
      grid_spacing_y: 7.5,
      x_extents: [0, 6],
      y_extents: [0, 6],
      num_floors: 8,
      floor_height: 3.2
    },
    evaluations: {
      steel_tonnage: 720,
      column_size: "UC356x368x129",
      beam_size: "UB533x210x92",
      carbon_footprint: 980,
      cost_per_sqm: 2800
    },
    width: 6,
    height: 8,
    depth: 6,
    color: "#42b883"
  },
  {
    id: 3,
    parameters: {
      grid_spacing_x: 9.0,
      grid_spacing_y: 9.0,
      x_extents: [0, 4],
      y_extents: [0, 4],
      num_floors: 4,
      floor_height: 4.0
    },
    evaluations: {
      steel_tonnage: 380,
      column_size: "UC254x254x73",
      beam_size: "UB406x178x60",
      carbon_footprint: 520,
      cost_per_sqm: 2300
    },
    width: 4,
    height: 4,
    depth: 4,
    color: "#e74c3c"
  },
  {
    id: 4,
    parameters: {
      grid_spacing_x: 8.5,
      grid_spacing_y: 8.5,
      x_extents: [0, 5],
      y_extents: [0, 5],
      num_floors: 6,
      floor_height: 3.8
    },
    evaluations: {
      steel_tonnage: 550,
      column_size: "UC305x305x97",
      beam_size: "UB457x191x82",
      carbon_footprint: 750,
      cost_per_sqm: 2600
    },
    width: 5,
    height: 6,
    depth: 5,
    color: "#9b59b6"
  }
];

// Function to get all schemes
export const getAllSchemes = () => {
  return mockSchemes;
};

// Function to get a scheme by ID
export const getSchemeById = (id) => {
  return mockSchemes.find(scheme => scheme.id === id) || null;
};

// Create a named export object
const schemeService = {
  getAllSchemes,
  getSchemeById
};

export default schemeService; 