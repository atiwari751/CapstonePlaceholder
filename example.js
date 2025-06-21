// Example of how to call the API from a React component using fetch

async function sendMessage(userInput, currentHistory) {
  try {
    const response = await fetch('http://localhost:8000/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        input: userInput,
        chat_history: currentHistory, // This should be an array of objects
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    // data will be { output: "...", chat_history: [...] }
    // You can then update your React component's state with this new data
    // to display the agent's response and maintain the conversation.
    console.log('Received from agent:', data);
    return data;

  } catch (error) {
    console.error("Failed to send message to agent:", error);
    // Handle the error in your UI
  }
}

// --- Example Usage in a React component ---
// const [chatHistory, setChatHistory] = useState([]);
// const [userInput, setUserInput] = useState("");

// const handleSend = async () => {
//   const responseData = await sendMessage(userInput, chatHistory);
//   if (responseData) {
//     setChatHistory(responseData.chat_history);
//     setUserInput(""); // Clear the input field
//   }
// };

