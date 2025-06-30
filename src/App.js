import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';
// Import mock data from service instead of hardcoding
import ChatMessage from './components/ChatMessage'; // Used for both user and agent messages
import SessionList from './components/SessionList'; // Component to show chat history
import SchemeGrid from './components/SchemeGrid'; // Component for 3D visualization

// API URL for the backend
const API_URL = "http://localhost:8001"; // FastAPI backend URL

function App() {
  // State for schemes/cuboids
  const [schemes, setSchemes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Agent state
  const [currentPrompt, setCurrentPrompt] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [pollingActive, setPollingActive] = useState(false);
  const [allSessions, setAllSessions] = useState([]);

  // Fetch all sessions on initial component mount
  useEffect(() => {
    const fetchAllSessions = async () => {
      setLoading(true);
      try {
        const response = await axios.get(`${API_URL}/sessions`);
        setAllSessions(response.data.sessions);
      } catch (err) {
        console.error("Failed to fetch sessions:", err);
        setError("Failed to fetch session history. The API server may be offline.");
      } finally {
        setLoading(false);
      }
    };
    fetchAllSessions();
  }, []);

  // Function to refresh the session list, e.g., after creating a new one
  const refreshSessionList = useCallback(async () => {
    try {
      const response = await axios.get(`${API_URL}/sessions`);
      setAllSessions(response.data.sessions);
    } catch (err) {
      console.error("Failed to refresh sessions:", err);
    }
  }, []);

  // Poll results function with useCallback to avoid dependency issues
  const pollResults = useCallback(async () => {
    if (!sessionId || !pollingActive) return false;
    
    try {
      const response = await axios.get(`${API_URL}/session/${sessionId}`);
      const data = response.data;
      
      // Update the last message in chat history if it's an agent turn
      setChatHistory(prev => {
        const lastMessage = prev[prev.length - 1];
        if (lastMessage && lastMessage.type === 'agent') {
          // Create a new message object to ensure re-render
          const updatedMessage = {
            ...lastMessage,
            status: data.status,
            results: data.results || {},
            finalAnswer: data.final_answer
          };
          return [...prev.slice(0, -1), updatedMessage];
        }
        return prev;
      });
      
      // Update schemes - completely replace any existing schemes
      // This state is separate and persists until a new chat starts.
      if (data.schemes && Array.isArray(data.schemes)) {
        // Ensure each scheme has a display-friendly name for the grid
        const namedSchemes = data.schemes.map((scheme, index) => ({
          ...scheme,
          name: scheme.name || `Scheme ${index + 1}`
        }));
        setSchemes(namedSchemes);
      }
      
      // Check if processing is complete
      if (data.status === "completed" || data.status === "error") {
        setPollingActive(false);
        setIsProcessing(false); // Re-enable input when polling stops
        return true;
      }
      
      return false;
    } catch (error) {
      console.error("Error polling results:", error);
      setError("Failed to get results. Please try again.");
      setPollingActive(false);
      setIsProcessing(false); // Re-enable input on error
      return true;
    }
  }, [sessionId, pollingActive]);

  // Set up polling effect
  useEffect(() => {
    let interval;
    
    if (pollingActive) {
      interval = setInterval(() => {
        pollResults();
      }, 1000); // Poll every second
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [pollingActive, sessionId, pollResults]);

  // Handle selecting a session from the list
  const handleSelectSession = useCallback(async (selectedSessionId) => {
    if (isProcessing) return; // Don't switch sessions while one is running

    setPollingActive(false); // Stop any active polling
    setIsProcessing(true); // Show loading state
    
    try {
      const response = await axios.get(`${API_URL}/session/${selectedSessionId}`);
      const data = response.data;
      
      setSessionId(selectedSessionId);
      setChatHistory(data.chat_history || []);
      
      const namedSchemes = (data.schemes || []).map((scheme, index) => ({
        ...scheme,
        name: scheme.name || `Scheme ${index + 1}`
      }));
      setSchemes(namedSchemes);
    } catch (err) {
      console.error("Failed to load session:", err);
      setError(`Failed to load session ${selectedSessionId}.`);
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing]);

  // Handle user prompt submission
  const handlePromptSubmit = async (e) => {
    e.preventDefault();
    
    const prompt = currentPrompt.trim();
    if (!prompt || isProcessing) return;
    
    setIsProcessing(true);

    // Determine if this is the start of a new chat
    const isNewChat = !sessionId;

    // Add user's prompt and a placeholder for the agent's response to the history
    const userMessage = { type: 'human', content: prompt };
    const agentPlaceholder = { type: 'agent', sessionId: sessionId, status: 'running', results: {}, finalAnswer: null };

    // If it's a new chat, replace the history; otherwise, append.
    // This prevents carrying over old messages if state updates are batched.
    setChatHistory(prev => 
      isNewChat ? [userMessage, agentPlaceholder] : [...prev, userMessage, agentPlaceholder]
    );
    
    try {
      // If we have a session ID, send it to continue the conversation.
      // Otherwise, the backend will create a new session.
      const payload = { query: prompt };
      if (sessionId) {
        payload.session_id = sessionId;
      }

      const response = await axios.post(`${API_URL}/query`, payload);
      const data = response.data;
      
      // If it was a new session, refresh the session list to include it
      if (!sessionId) {
        await refreshSessionList();
      }

      setSessionId(data.session_id);
      setPollingActive(true);

    } catch (error) {
      console.error("Error processing query:", error);
      const errorMessage = "An error occurred. The API server may be offline.";
      setError(errorMessage);
      // Update the placeholder to show the error instead of adding a new message
      setChatHistory(prev => {
        const newHistory = [...prev];
        const lastMessageIndex = newHistory.length - 1;
        if (lastMessageIndex >= 0 && newHistory[lastMessageIndex].type === 'agent') {
          newHistory[lastMessageIndex].status = 'error';
          newHistory[lastMessageIndex].finalAnswer = errorMessage;
        }
        return newHistory;
      });
      setIsProcessing(false);
      setPollingActive(false);
    }
    
    // Clear the input field
    setCurrentPrompt('');
  };
  
  // Handle starting a new chat view
  const handleNewQuery = () => {
    setSessionId(null);
    setPollingActive(false);
    setIsProcessing(false);
    setChatHistory([]);
    setSchemes([]);
  };

  if (loading) {
    return <div className="loading">Loading schemes...</div>;
  }

  if (error) {
    return <div className="loading">{error}</div>;
  }

  return (
    <div className="app-container">
      {/* Split layout with visualization area and agent area */}
      <div className="split-layout">
        {/* Visualization Area (Left Side) */}
        <div className="visualization-area">
          <div className="vis-header">
            <h3>3D Visualization</h3>
            <p>Click on a scheme to view its details in a side panel. Click and drag to rotate, scroll to zoom.</p>
          </div>
          
          {/* Add scrollable container */}
          <div className="visualization-content">
            {/* Use the SchemeGrid component */}
            <SchemeGrid schemes={schemes} />
          </div>
        </div>
        
        {/* Agent Area (Right Side) */}
        <div className="agent-area">
          <div className="agent-header">
            <h2>AGENT</h2>
          </div>

          <SessionList 
            sessions={allSessions}
            onSelectSession={handleSelectSession}
            onNewChat={handleNewQuery}
            currentSessionId={sessionId}
          />
          
          <div className="agent-content">
            <div className="chat-log">
              {chatHistory.map((msg, index) => {
                if (msg.type === 'human') {
                  // Pass isUser=true and map content to the 'text' prop expected by ChatMessage
                  return <ChatMessage key={index} message={{ text: msg.content }} isUser={true} />;
                }
                // Handle live, in-progress agent messages
                if (msg.type === 'agent') { 
                  // The ChatMessage component handles rendering agent responses, including tool results and the final answer.
                  return <ChatMessage key={index} message={msg} isUser={false} />;
                }
                // Handle historical, completed agent messages from the backend
                if (msg.type === 'ai') {
                  // Transform the historical 'ai' message into the format the ChatMessage component expects for a completed response.
                  const completedMessage = { status: 'completed', finalAnswer: msg.content };
                  return <ChatMessage key={index} message={completedMessage} isUser={false} />;
                }
                return null;
              })}
            </div>
          </div>
          
          {/* User Input Area - moved inside agent-area */}
          <div className="user-input-area">
            <div className="input-form-container">
              <form onSubmit={handlePromptSubmit} className="chat-form">
                <input
                  type="text"
                  value={currentPrompt}
                  onChange={(e) => setCurrentPrompt(e.target.value)}
                  placeholder="Enter your prompt here..."
                  disabled={isProcessing}
                />
                <button type="submit" disabled={isProcessing || !currentPrompt.trim()}>
                  {isProcessing ? "Processing..." : "Send"}
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App; 