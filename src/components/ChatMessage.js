import React from 'react';
import './ChatMessage.css';

// Component to display a single tool result within a chat message
const ToolResult = ({ toolName, status, result }) => {
  // Status indicators
  let statusIndicator = 'ℹ️';
  if (status === 'Running') statusIndicator = '⏳';
  else if (status === 'Finished') statusIndicator = '✅';
  else if (status === 'Error') statusIndicator = '❌';

  return (
    <details className="tool-result" open={status === 'Running'}>
      <summary className="tool-summary">
        Tool: {toolName} - {statusIndicator} {status}
      </summary>
      <div className="tool-content">
        {result === "Executing..." ? (
          <div className="info-message">Tool is currently executing...</div>
        ) : (
          <pre className="result-code">{result}</pre>
        )}
      </div>
    </details>
  );
};

const ChatMessage = ({ message, isUser }) => {
  return (
    <div className={`chat-message ${isUser ? 'user-message' : 'agent-message'}`}>
      <div className="message-content">
        {isUser ? (
          <div className="user-text">{message.text}</div>
        ) : (
          <>
            {message.text && <div className="agent-text">{message.text}</div>}
            
            {/* Display tool results if present */}
            {message.results && Object.entries(message.results).length > 0 && (
              <div className="results-section">
                <div className="results-list">
                  {Object.entries(message.results).sort((a, b) => a[0].localeCompare(b[0])).map(([key, result]) => (
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
            
            {/* Display final answer if present */}
            {message.finalAnswer && (
              <div className="final-answer">
                <h4>Agent Response</h4>
                <div className="answer-content">{message.finalAnswer}</div>
              </div>
            )}
          </>
        )}
      </div>
      <div className="message-time">{message.timestamp}</div>
    </div>
  );
};

export default ChatMessage; 