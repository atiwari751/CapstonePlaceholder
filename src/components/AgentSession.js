import React, { useState } from 'react';
import './AgentSession.css';

// Component to display a single tool result
const ToolResult = ({ toolName, status, result }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Status indicators
  let statusIndicator = null;
  if (status === 'Finished') statusIndicator = '✅';
  else if (status === 'Running') statusIndicator = '⏳';
  else if (status === 'Error') statusIndicator = '❌';
  
  // Toggle expanded state
  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };
  
  return (
    <div className="tool-result">
      <div className="tool-summary" onClick={toggleExpanded}>
        {isExpanded ? '▼' : '▶'} 
        <div className="tool-info">
          <span className="tool-name">Tool: {toolName}</span>
          <span className="tool-status">
            {statusIndicator && statusIndicator} {status}
          </span>
        </div>
      </div>
      
      {isExpanded && (
        <div className="tool-content">
          {result === "Executing..." ? (
            <div className="info-message">Tool is currently executing...</div>
          ) : (
            <pre className="result-code">{result || 'No result available'}</pre>
          )}
        </div>
      )}
    </div>
  );
};

// Component to display the final answer
const FinalAnswer = ({ answer }) => {
  return (
    <div className="final-answer">
      <h3>Final Answer</h3>
      <div className="answer-content">{answer}</div>
    </div>
  );
};

// Main component for displaying the agent session
const AgentSession = ({ sessionId, status, results, finalAnswer }) => {
  // Sort results by key to ensure they appear in order
  const sortedResults = Object.entries(results || {}).sort((a, b) => a[0].localeCompare(b[0]));

  return (
    <div className="agent-session">
      <div className="session-info">
        <p>Session ID: <code>{sessionId}</code></p>
        <p>Status: <strong>{status.toUpperCase()}</strong></p>
      </div>

      {sortedResults.length > 0 && (
        <div className="results-section">
          <h3>Intermediate Results</h3>
          <div className="results-list">
            {sortedResults.map(([key, result]) => (
              <ToolResult
                key={key}
                toolName={result.tool}
                status={result.status}
                result={result.result}
              />
            ))}
          </div>
        </div>
      )}

      {finalAnswer && (
        <FinalAnswer answer={finalAnswer} />
      )}
    </div>
  );
};

export default AgentSession; 