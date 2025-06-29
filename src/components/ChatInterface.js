import React, { useState, useRef, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import './ChatInterface.css';

const ChatInterface = ({ onSendMessage, isProcessing }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Function to format current time
  const getCurrentTime = () => {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!inputValue.trim() || isProcessing) return;
    
    // Create new user message
    const newUserMessage = {
      id: Date.now(),
      text: inputValue,
      timestamp: getCurrentTime(),
      isUser: true
    };
    
    // Add user message to chat
    setMessages(prevMessages => [...prevMessages, newUserMessage]);
    
    // Create placeholder for agent response
    const agentPlaceholder = {
      id: Date.now() + 1,
      text: "Thinking...",
      timestamp: getCurrentTime(),
      isUser: false,
      results: {}
    };
    
    // Add agent placeholder to chat
    setMessages(prevMessages => [...prevMessages, agentPlaceholder]);
    
    // Send message to parent component for processing
    onSendMessage(inputValue, (update) => {
      // Update the agent message with results as they come in
      setMessages(prevMessages => {
        // Find the last agent message (placeholder)
        const lastAgentIndex = [...prevMessages].reverse().findIndex(msg => !msg.isUser);
        if (lastAgentIndex === -1) return prevMessages;
        
        // Calculate the actual index in the array
        const actualIndex = prevMessages.length - 1 - lastAgentIndex;
        
        // Create a new array with the updated agent message
        const updatedMessages = [...prevMessages];
        updatedMessages[actualIndex] = {
          ...updatedMessages[actualIndex],
          text: update.finalAnswer ? null : "Processing your request...",
          results: update.results,
          finalAnswer: update.finalAnswer,
          timestamp: getCurrentTime()
        };
        
        return updatedMessages;
      });
    });
    
    // Clear input field
    setInputValue('');
  };

  // Handle starting a new session
  const handleNewSession = () => {
    setMessages([]);
  };

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h2>SCHEMING AGENT</h2>
        <button className="new-session-button" onClick={handleNewSession}>
          New Session
        </button>
      </div>
      
      <div className="chat-messages" ref={chatContainerRef}>
        {messages.length === 0 ? (
          <div className="empty-chat">
            <p>No messages yet. Start a conversation!</p>
          </div>
        ) : (
          messages.map(message => (
            <ChatMessage 
              key={message.id} 
              message={message} 
              isUser={message.isUser} 
            />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form className="chat-input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type your message..."
          disabled={isProcessing}
          className="chat-input"
        />
        <button 
          type="submit" 
          disabled={isProcessing || !inputValue.trim()}
          className="send-button"
        >
          {isProcessing ? "..." : "Send"}
        </button>
      </form>
    </div>
  );
};

export default ChatInterface; 