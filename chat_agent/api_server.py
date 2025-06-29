import uuid
import asyncio
import sys
import shelve
import logging
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to the path to allow importing sibling modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

# Agent and tool imports
from .main import create_agent_executor
from .callbacks import SessionCallbackHandler
from scheme_service import scheme_service

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    session_id: str
    message: str

class SessionStatusResponse(BaseModel):
    status: str
    results: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    schemes: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

# --- In-memory Session Storage ---
# sessions: Dict[str, Dict[str, Any]] = {} # Replaced with shelve
sessions = None
SESSION_DB_FILE = "session_storage.db"

# --- FastAPI App Setup ---
app = FastAPI(
    title="Sustainable Building Design Assistant API",
    description="A session-based API to interact with the LangChain agent.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React's default dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the agent executor once on startup
agent_executor = create_agent_executor()

# --- Agent Processing Logic ---
def run_agent_in_background(session_id: str, query: str):
    """
    The function that runs the agent executor.
    This will be executed in a background task.
    """
    try:
        logger.info(f"Starting agent run for session {session_id} with query: '{query}'")
        # Retrieve the session data dictionary from the shelf
        session_data = sessions[session_id]
        
        # Create a callback handler for this specific session
        callback_handler = SessionCallbackHandler(session_data)
        
        # Invoke the agent with the query and the session-specific callback
        agent_executor.invoke(
            {"input": query},
            config={"callbacks": [callback_handler]}
        )

        # IMPORTANT: Re-assign the modified session_data back to the shelf to persist changes.
        # shelve does not track in-place mutations of the objects it stores.
        sessions[session_id] = session_data
        logger.info(f"Agent run completed and session {session_id} saved.")
        
    except Exception as e:
        logger.exception(f"Error during agent execution for session {session_id}: {e}")
        session_data = sessions.get(session_id, {}) # Use .get for safety
        session_data["status"] = "error"
        session_data["error"] = str(e)
        sessions[session_id] = session_data # Persist error state

# --- API Endpoints ---
@app.post("/query", response_model=QueryResponse)
async def create_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Starts a new agent session.
    """
    logger.info(f"Received new query: '{request.query}'")
    session_id = str(uuid.uuid4())
    
    # Initialize session data
    sessions[session_id] = {
        "status": "running",
        "results": {},
        "final_answer": None,
        "schemes": [],
        "error": None,
    }
    logger.info(f"Created new persistent session: {session_id}")
    
    # Add the agent execution to background tasks
    background_tasks.add_task(run_agent_in_background, session_id, request.query)
    
    return {"session_id": session_id, "message": "Query processing started."}

@app.get("/session/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(session_id: str):
    """
    Polls for the status and results of an agent session.
    """
    logger.debug(f"Polling session status for {session_id}")
    if session_id not in sessions:
        logger.warning(f"Session {session_id} not found during status poll.")
        raise HTTPException(status_code=404, detail="Session not found")
        
    session_data = sessions[session_id]

    return session_data

@app.on_event("startup")
async def startup_event():
    """Open the session database on server startup."""
    global sessions
    sessions = shelve.open(SESSION_DB_FILE, writeback=False)
    logger.info("Agent API server started on http://localhost:8001")
    logger.info(f"Session database opened at '{SESSION_DB_FILE}'")

@app.on_event("shutdown")
async def shutdown_event():
    """Close the session database on server shutdown."""
    if sessions is not None:
        sessions.close()
        logger.info("Session database closed.")

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8001 to match the frontend's expectation
    uvicorn.run(app, host="0.0.0.0", port=8001)