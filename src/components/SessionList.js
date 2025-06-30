import React from 'react';
import './SessionList.css';

const SessionList = ({ sessions, onSelectSession, onNewChat, currentSessionId }) => {
  return (
    <div className="session-list-container">
      <h4>Chat History</h4>
      <button className="new-chat-button-sidebar" onClick={onNewChat}>
        + New Chat
      </button>
      <ul className="session-list">
        {sessions.map(session => (
          <li
            key={session.session_id}
            className={`session-item ${session.session_id === currentSessionId ? 'active' : ''}`}
            onClick={() => onSelectSession(session.session_id)}
          >
            <div className="session-title">{session.first_query}</div>
            {session.last_agent_response && (
              <div className="session-preview">{session.last_agent_response}</div>
            )}
            <div className="session-date">
              {new Date(session.created_at).toLocaleString()}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default SessionList;