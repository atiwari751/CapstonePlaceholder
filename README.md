# Sustainable Building Design Assistant

This project is a web-based, conversational AI assistant designed to help architects and designers create and evaluate sustainable building schemes. It leverages a powerful language model (Google Gemini) through LangChain to understand user requests, generate building designs, evaluate them for carbon emissions, and suggest low-emission construction products.

The application features a split-screen interface with a 3D visualization panel and an interactive chat panel, allowing for a dynamic and intuitive design process.

## Features

-   **Conversational AI:** Interact with the agent using natural language to generate and modify building designs.
-   **3D Visualization:** Instantly view and interact with 3D models of the generated building schemes.
-   **Emissions Evaluation:** The agent can evaluate schemes based on material quantities (steel, concrete) and calculate their estimated manufacturing carbon emissions.
-   **Low-Emission Product Search:** Find and compare alternative, low-carbon building materials from a product database.
-   **Persistent Chat History:** All conversations are saved, allowing you to revisit and continue previous design sessions.

## Architecture

The project follows a modern client-server architecture:

-   **Frontend:** A responsive user interface built with **React**. It handles user interaction, chat display, and 3D rendering using `react-three-fiber`.
-   **Backend:** A robust API server built with **Python** and **FastAPI**. It manages user sessions, orchestrates the AI agent, and serves data to the frontend.
-   **AI Core:** The brain of the application is a **LangChain** agent powered by **Google's Gemini** model. The agent is equipped with custom tools to perform specific tasks like scheme evaluation and product searches.
-   **Session Storage:** Chat history and session data are persisted on the backend using Python's `shelve` module, creating a `session_storage.db` file.

## Setup and Installation

Follow these steps to set up and run the project locally.

### Prerequisites

-   Python (3.9 or higher)
-   Node.js and npm (v16 or higher)
-   A Google Gemini API Key.

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <project dir>
```

### 2. Backend Setup

Navigate to the project root and set up the Python environment.

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root (`/Users/shriti/Downloads/CapstoneProj/FreshCode/.env`) and add your API keys and other credentials.

```dotenv
# .env

# Google Gemini API Key
GEMINI_API_KEY="your-gemini-api-key"

# Environment variables for the MCP Server tools (if used)
# These are required for the evaluate_building_schemes and product search tools.
DEVELOPER_TOKEN="your-developer-token"
API_URL="your-api-url"
API_ENDPOINT_NAME="your-api-endpoint-name"
AZURE_CLIENT_ID="your-azure-client-id"
AZURE_CLIENT_SECRET="your-azure-client-secret"
AZURE_SCOPE="your-azure-scope"
```

### 3. Frontend Setup

In a separate terminal, navigate to the project root and install the Node.js dependencies.

```bash
npm install
```

## Running the Application

1.  **Start the Backend Server:**
    From the project root, with your virtual environment activated, run:
    ```bash
    python3 -m chat_agent.api_server
    ```
    The FastAPI server will start on `http://localhost:8001`.

2.  **Start the Frontend Application:**
    In your other terminal (at the project root), run:
    ```bash
    npm start
    ```
    The React development server will start, and the application will open in your browser at `http://localhost:3000`.

## API Endpoints

The backend exposes the following RESTful API endpoints:

-   `POST /sessions`: Creates a new, empty chat session.
-   `GET /sessions`: Retrieves a list of all past chat sessions.
-   `POST /query`: Submits a user query to an existing session for the agent to process.
-   `GET /session/{session_id}`: Polls for the status and results of a specific session.

## Project Structure

```
.
├── chat_agent/         # Backend FastAPI server and LangChain agent logic
├── public/
├── src/                # Frontend React application source
├── .env                # Environment variables (you must create this)
├── package.json
├── requirements.txt    # Python dependencies
└── README.md           # This file
```