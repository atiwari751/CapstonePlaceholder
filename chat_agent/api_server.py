import uuid
import asyncio
import sys
import shelve
import logging
import os
import datetime
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
from main import create_agent_executor
from callbacks import SessionCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from scheme_service import scheme_service

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    first_query: str
    last_agent_response: Optional[str] = None

class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]

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
        for msg in raw_history:
            if isinstance(msg, (HumanMessage, AIMessage)):
                chat_history_for_agent.append(msg)
            elif isinstance(msg, dict):
                if msg.get("type") == "human":
                    chat_history_for_agent.append(HumanMessage(content=msg.get("content")))
                elif msg.get("type") == "ai":
                    chat_history_for_agent.append(AIMessage(content=msg.get("content")))

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
        
        agent_output = response.get("output", "")
        
        # --- Memory Enhancement ---
        # The agent sometimes forgets to include PRODUCT_DATA in its final answer after an evaluation.
        # We will inspect the intermediate tool outputs and manually append any missing
        # PRODUCT_DATA blocks to the response that gets saved in the chat history.
        # This makes the agent's memory more robust.
        product_data_blocks = []
        if "intermediate_steps" in response:
            for action, observation in response["intermediate_steps"]:
                if "PRODUCT_DATA:" in str(observation):
                    parts = str(observation).split("PRODUCT_DATA:")
                    for part in parts[1:]:
                        block = f"PRODUCT_DATA: {part.strip()}"
                        product_data_blocks.append(block)
        
        final_response_for_history = agent_output
        for block in product_data_blocks:
            if block not in final_response_for_history:
                final_response_for_history += f"\n\n{block}"
        # --- End Memory Enhancement ---

        # Clean the final answer for UI display, removing any data blocks.
        ui_final_answer = agent_output
        if "PRODUCT_DATA:" in ui_final_answer:
            ui_final_answer = ui_final_answer.split("PRODUCT_DATA:")[0].strip()
        if "SCHEME_DATA:" in ui_final_answer:
            ui_final_answer = ui_final_answer.split("SCHEME_DATA:")[0].strip()
        session_data["final_answer"] = ui_final_answer

        session_data["chat_history"].append(
            {"type": "human", "content": query}
        )
        session_data["chat_history"].append(
            {"type": "ai", "content": final_response_for_history}
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
        session_data = sessions[session_id]
        
        # If this is the first message in a newly created session, update its title
        if session_data.get("status") == "new" and not session_data.get("chat_history"):
            logger.info(f"Updating title for new session {session_id} to '{request.query}'")
            session_data["first_query"] = request.query
        
        session_data["status"] = "running"
        session_data["results"] = {}
        session_data["final_answer"] = None
        session_data["error"] = None
        sessions[session_id] = session_data # Write back status change
    else:
        # This branch is now a fallback for clients that don't pre-create sessions.
        # The preferred flow is to create a session first via POST /sessions.
        raise HTTPException(status_code=400, detail="session_id is required. Please create a session first.")

    # Add the agent execution to background tasks
    background_tasks.add_task(run_agent_in_background, session_id, request.query)
    
    return {"session_id": session_id}

@app.post("/sessions", response_model=SessionInfo, status_code=201)
async def create_new_session():
    """
    Creates a new, empty session and returns its details.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"Explicitly creating new session {session_id}")
    
    session_data = {
        "status": "new", # A new status to indicate it's empty
        "results": {},
        "final_answer": None,
        "schemes": [],
        "error": None,
        "chat_history": [],
        "created_at": datetime.datetime.now().isoformat(),
        "first_query": "(New Chat)" # Placeholder title
    }
    sessions[session_id] = session_data

    return SessionInfo(
        session_id=session_id,
        created_at=session_data["created_at"],
        first_query=session_data["first_query"],
        last_agent_response=""
    )

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

@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    """
    Lists all available sessions with their metadata, sorted by creation date.
    """
    session_list = []
    # Sort by creation date, newest first
    sorted_session_ids = sorted(
        sessions.keys(),
        key=lambda sid: sessions[sid].get("created_at", "1970-01-01"),
        reverse=True
    )

    for sid in sorted_session_ids:
        session_data = sessions[sid]
        
        # Find the last agent response from chat history for a preview
        last_agent_response = ""
        chat_history = session_data.get("chat_history", [])
        for msg in reversed(chat_history): # Iterate backwards to find the last AI message
            is_ai_message = False
            content = ""
            if isinstance(msg, dict):
                if msg.get("type") == "ai":
                    is_ai_message = True
                    content = msg.get("content", "").strip()
            elif isinstance(msg, AIMessage):
                is_ai_message = True
                content = msg.content.strip()

            if is_ai_message and content:
                # Clean the content for preview, removing data blocks
                if "PRODUCT_DATA:" in content:
                    content = content.split("PRODUCT_DATA:")[0].strip()
                last_agent_response = content
                break # Found the last one, so we can stop searching

        session_list.append(
            SessionInfo(
                session_id=sid,
                created_at=session_data.get("created_at", "N/A"),
                first_query=session_data.get("first_query", "Untitled Session"),
                last_agent_response=last_agent_response
            )
        )
    return {"sessions": session_list}

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
        try:
            sessions.close()
            logger.info("Session database closed.")
            # Clean up the session files
            db_path_base = Path(SESSION_DB_FILE).stem
            for f in Path('.').glob(f'{db_path_base}.*'):
                os.remove(f)
                logger.info(f"Removed session file: {f}")
        except Exception as e:
            logger.error(f"Error during session database shutdown and cleanup: {e}")

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8001 to match the frontend's expectation
    uvicorn.run(app, host="0.0.0.0", port=8001)