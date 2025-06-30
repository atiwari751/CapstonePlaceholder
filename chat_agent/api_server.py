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
from langchain_core.messages import AIMessage, HumanMessage
from scheme_service import scheme_service

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    session_id: str

class ChatMessage(BaseModel):
    type: str
    content: str

class SessionStatusResponse(BaseModel):
    status: str
    results: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    schemes: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = None

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
        session_data = sessions[session_id]

        # Reconstruct chat history from stored dicts into LangChain message objects
        chat_history_for_agent = []
        raw_history = session_data.get("chat_history", [])
        for msg_data in raw_history:
            if msg_data.get("type") == "human":
                chat_history_for_agent.append(HumanMessage(content=msg_data.get("content")))
            elif msg_data.get("type") == "ai":
                chat_history_for_agent.append(AIMessage(content=msg_data.get("content")))

        # Create a callback handler for this specific session
        # It will write updates directly to the shelve DB
        callback_handler = SessionCallbackHandler(sessions, session_id)

        # Invoke the agent with the query and the session-specific callback
        # The history provides context for the conversation
        response = agent_executor.invoke(
            {"input": query, "chat_history": chat_history_for_agent},
            config={"callbacks": [callback_handler]}
        )

        # After the agent runs, manually update the chat history in the session store.
        # This ensures the full conversation, including tool outputs embedded in the
        # AI message, is persisted for the next turn.
        session_data = sessions[session_id] # Re-fetch data modified by callback
        
        # Clean the final answer for UI display, removing the PRODUCT_DATA block.
        # The full response output is still saved to chat_history for agent memory.
        final_answer_from_agent = session_data.get("final_answer")
        if final_answer_from_agent:
            # Remove PRODUCT_DATA for product searches
            if "PRODUCT_DATA:" in final_answer_from_agent:
                final_answer_from_agent = final_answer_from_agent.split("PRODUCT_DATA:")[0].strip()
            # Remove SCHEME_DATA for scheme evaluations
            if "SCHEME_DATA:" in final_answer_from_agent:
                final_answer_from_agent = final_answer_from_agent.split("SCHEME_DATA:")[0].strip()
            session_data["final_answer"] = final_answer_from_agent

        session_data["chat_history"].append(
            {"type": "human", "content": query}
        )
        session_data["chat_history"].append(
            {"type": "ai", "content": response.get("output", "")}
        )
        
        # Persist the updated history and any other changes from the callback
        sessions[session_id] = session_data

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
    session_id = request.session_id

    if session_id and session_id in sessions:
        logger.info(f"Continuing session {session_id} with query: '{request.query}'")
        # Session exists, update its status and reset turn-specific data
        # to prevent showing stale results from the previous turn.
        session_data = sessions[session_id]
        session_data["status"] = "running"
        session_data["results"] = {}
        session_data["final_answer"] = None
        session_data["error"] = None
        sessions[session_id] = session_data # Write back status change
    else:
        session_id = str(uuid.uuid4())
        logger.info(f"Received new query, creating session {session_id}: '{request.query}'")
        # Initialize new session data
        sessions[session_id] = {
            "status": "running",
            "results": {},
            "final_answer": None,
            "schemes": [],
            "error": None,
            "chat_history": [] # Initialize chat history for new sessions
        }

    # Add the agent execution to background tasks
    background_tasks.add_task(run_agent_in_background, session_id, request.query)
    
    return {"session_id": session_id}

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

    # Create a copy to return as JSON. The chat_history is already serializable.
    response_data = session_data.copy()
    # Ensure the chat_history key exists for the Pydantic model, even if empty.
    if "chat_history" not in response_data:
        response_data["chat_history"] = []

    return response_data

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