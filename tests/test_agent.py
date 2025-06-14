import unittest
import logging
import sys # For diagnostic print
import os  # For diagnostic print
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import subprocess  # For mocking subprocess.Popen

from agent import ConversationalAgent
from module.memory import Memory # We'll be checking interactions with Memory

class MockAgentTool:
    def __init__(self, name="mock_tool"):
        self.name = name
        self.description = "A mock tool for testing agent."
        self.execute_called_with = None

    def get_schema(self): # Needs to exist for Decision module init
        return {"name": self.name, "description": self.description, "parameters": {}}

    def execute(self, tool_input: dict):
        self.execute_called_with = tool_input
        return {"status": "success from mock_tool", "input_received": tool_input}

class TestConversationalAgentPhase3(unittest.TestCase):

    def setUp(self):
        """Set up for each test."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        logging.disable(logging.CRITICAL)
        self.mock_decision_instance = MagicMock()
        self.mock_tool_instance = MockAgentTool(name="sample_tool")
        self.mock_perception_instance = MagicMock() # Mock for the Perception instance
        
        # Patch the Decision class within the agent module for all tests in this class
        self.decision_patcher = patch('agent.Decision', return_value=self.mock_decision_instance)
        self.MockDecisionClass = self.decision_patcher.start()

        # Mock for mcp_server_process
        self.mock_mcp_process_for_agent = MagicMock(spec=subprocess.Popen)
        self.mock_mcp_process_for_agent.stdin = MagicMock()
        self.mock_mcp_process_for_agent.stdout = MagicMock()
        self.mock_mcp_process_for_agent.stderr = MagicMock()
        self.mock_mcp_process_for_agent.poll.return_value = None # Simulate running
        # This is the critical patch. It targets 'Perception' as it's named in the 'agent' module's namespace
        # due to 'from module.perception import Perception' in agent.py
        self.perception_patcher = patch('agent.Perception', return_value=self.mock_perception_instance)
        self.MockPerceptionClass = self.perception_patcher.start()

        # Patch subprocess.Popen for start_mcp_server
        self.popen_patcher = patch('subprocess.Popen', return_value=self.mock_mcp_process_for_agent)
        self.MockPopen = self.popen_patcher.start()

        # Make the tool's execute method an AsyncMock
        self.mock_tool_instance.execute = AsyncMock(return_value={"status": "mock_tool_success"})
        # Make Decision's execute_plan_with_memory an AsyncMock
        self.mock_decision_instance.execute_plan_with_memory = AsyncMock()

        self.agent = ConversationalAgent(tools_list=[self.mock_tool_instance]) # Agent init will call start_mcp_server

    def tearDown(self):
        """Clean up after each test."""
        self.decision_patcher.stop() 
        self.perception_patcher.stop()
        self.popen_patcher.stop()
        logging.disable(logging.NOTSET) # Re-enable logging
        self.loop.close()  # Close the event loop to clean up resources

    def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        self.MockPopen.assert_called_once() # Check if mcp-server was started
        self.assertIsInstance(self.agent.memory, Memory)
        self.assertFalse(self.agent.is_session_active)
        self.MockDecisionClass.assert_called_once_with(tools_list=[self.mock_tool_instance])
        self.MockPerceptionClass.assert_called_once() # Check if Perception was initialized
        self.assertIn("sample_tool", self.agent.tools_execution_map)

    def test_start_session(self):
        
        """Test the start_session method."""
        initial_message = self.agent.start_session()
        self.assertTrue(self.agent.is_session_active)
        self.assertEqual(initial_message, "What do you want to do today?")
        # Check if agent's greeting is in memory
        self.assertEqual(len(self.agent.memory.conversation_history), 1)
        self.assertEqual(self.agent.memory.conversation_history[0]["role"], "agent")
        self.assertEqual(self.agent.memory.conversation_history[0]["final_answer"], "What do you want to do today?")

    def test_start_session_when_already_active(self):
        """Test calling start_session when a session is already active."""
        self.agent.start_session() # First start
        second_start_message = self.agent.start_session() # Attempt to start again
        self.assertTrue(self.agent.is_session_active)
        # In the current implementation, start_session clears memory and restarts.
        # So history will only have the latest greeting.
        self.assertEqual(len(self.agent.memory.conversation_history), 1)
        self.assertEqual(self.agent.memory.conversation_history[0]["final_answer"], "What do you want to do today?")

    def test_end_session(self):
        """Test the end_session method."""
        self.agent.start_session()
        farewell_message = self.agent.end_session()
        self.assertFalse(self.agent.is_session_active)
        self.assertEqual(farewell_message, "That is all, thank you.")
        # Check memory for user's "bye" (added by process_user_query) and agent's "Session ended"
        # For this direct test of end_session, only agent's "Session ended" would be there if not called via process_user_query
        last_turn = self.agent.memory.conversation_history[-1]
        self.assertEqual(last_turn["role"], "agent")
        self.assertEqual(last_turn["content"], "Session ended.")

    def test_end_session_when_not_active(self):
        """Test calling end_session when no session is active."""
        response = self.agent.end_session()
        self.assertEqual(response, "No active session to end.")

    def test_process_user_query_normal(self):
        """Test processing a query where Decision module returns final_answer."""
        self.agent.start_session()
        user_query = "Hello there"
        # Configure mock Perception response
        mock_perception_output = {
            "original_query": user_query, "normalized_query": user_query.lower(),
            "intent": "greeting", "entities": {"mood": "happy"}, "command": None # Example entity
        }
        self.mock_perception_instance.process_user_input.return_value = mock_perception_output

        # Configure mock Decision response
        async def mock_decision_return(*args, **kwargs): # Use side_effect for AsyncMock
            return (
            "Thinking about a direct answer.", # thought
            "final_answer",                    # tool_name
            {},                                # tool_input
            "Hello to you too!",               # speak
            []                                 # memory_actions
            )
        self.mock_decision_instance.execute_plan_with_memory.side_effect = mock_decision_return
        # Ensure check_for_repeated_question returns None for this test case
        async def run_test():
            with patch.object(self.agent.memory, 'check_for_repeated_question', return_value=None) as mock_check_repeat:
                response = await self.agent.process_user_query(user_query) # Await the async method
                mock_check_repeat.assert_called_once_with(user_query)

            expected_final_answer = "Hello to you too!" # This is the 'speak' output
            self.assertEqual(response, expected_final_answer)

            # History should have: 1. Agent greeting, 2. User query, 3. Agent response
            self.assertEqual(len(self.agent.memory.conversation_history), 3)
            self.assertEqual(self.agent.memory.conversation_history[1]["role"], "user")
            self.assertEqual(self.agent.memory.conversation_history[1]["content"], user_query)
            self.assertEqual(self.agent.memory.conversation_history[2]["role"], "agent")
            self.assertEqual(self.agent.memory.conversation_history[2]["final_answer"], expected_final_answer) # Check final_answer
            self.assertEqual(self.agent.memory.conversation_history[2]["content"], "Thinking about a direct answer.") # Check thought process
            self.mock_perception_instance.process_user_input.assert_called_once_with(user_query)
            self.mock_decision_instance.execute_plan_with_memory.assert_called_once()
            
            # Check arguments passed to the async mock
            self.mock_decision_instance.execute_plan_with_memory.assert_awaited_once_with(
                user_query=user_query,
                perception_data=mock_perception_output,
                memory_context=unittest.mock.ANY, # Or more specific if needed
                full_conversation_history=unittest.mock.ANY
            )
        self.loop.run_until_complete(run_test())

    def test_process_user_query_repeated_question(self):
        """Test processing a repeated user query."""
        self.agent.start_session()
        user_query = "What is the weather like?"

        # 1. First time asking: Configure mock Perception and Decision
        mock_perception_output_first_call = {
            "original_query": user_query, "normalized_query": user_query.lower(),
            "intent": "query_weather", "entities": {}, "command": None
        }
        self.mock_perception_instance.process_user_input.return_value = mock_perception_output_first_call

        initial_speak_output = "The weather is currently placeholder."
        # This is what the agent will store as final_answer for the first turn
        expected_first_final_answer_stored_in_memory = f"{initial_speak_output}\n\nWhat do you want to do next?"
        async def run_test():
             # Configure decision mock for the first call
            async def mock_decision_first_call_return(*args, **kwargs): # Use side_effect
                return (
                "Thinking about the weather for the first time.",
                "final_answer", {}, initial_speak_output, []
                )
            self.mock_decision_instance.execute_plan_with_memory.side_effect = mock_decision_first_call_return
            await self.agent.process_user_query(user_query) 

            # 2. Second time asking: Configure Perception mock for the second call
            #    and mock 'check_for_repeated_question' to return the full final_answer
            #    that would have been stored from the agent's previous turn.
            mock_perception_output_second_call = { # Perception is still called
                "original_query": user_query, "normalized_query": user_query.lower(),
                "intent": "query_weather", "entities": {}, "command": None # Could be same or different
            }
            self.mock_perception_instance.process_user_input.return_value = mock_perception_output_second_call

            # Mock check_for_repeated_question to return what was stored as final_answer
            with patch.object(self.agent.memory, 'check_for_repeated_question', return_value=expected_first_final_answer_stored_in_memory) as mock_check_repeat:
                response = await self.agent.process_user_query(user_query) # Second call
                mock_check_repeat.assert_called_once_with(user_query)

                self.assertEqual(response, expected_first_final_answer_stored_in_memory) 
                self.assertEqual(self.mock_decision_instance.execute_plan_with_memory.call_count, 1)
                self.assertEqual(self.mock_perception_instance.process_user_input.call_count, 2) # Called for both user queries
        self.loop.run_until_complete(run_test())

    def test_process_user_query_ends_session(self):
        """Test that 'bye' ends the session via process_user_query."""
        self.agent.start_session()
        async def run_test():
            response = await self.agent.process_user_query("bye") 
            self.assertEqual(response, "That is all, thank you.")
            self.assertFalse(self.agent.is_session_active)
            # Perception would still be called for "bye"
            self.mock_perception_instance.process_user_input.assert_called_with("bye")
        self.loop.run_until_complete(run_test())
    # TODO: Add tests for how ConversationalAgent handles specific commands from Perception
    # if we decide to implement direct handling in the agent before Decision module.
    # For now, all perception output goes to Decision.

if __name__ == '__main__':
    unittest.main()