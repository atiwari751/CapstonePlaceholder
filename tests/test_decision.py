import unittest
import json
import logging
from unittest.mock import patch, MagicMock
import asyncio
 
from module.decision import Decision

# Mock Tool class for testing _format_tools_for_prompt
class MockTool:
    def __init__(self, name, description, schema_params):
        self.name = name
        self.description = description
        self._schema_params = schema_params

    def get_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self._schema_params,
                "required": list(self._schema_params.keys())
            }
        }

class TestDecisionModule(unittest.TestCase):

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        logging.disable(logging.CRITICAL) # Suppress logs during tests
        self.mock_tool1 = MockTool(
            name="test_tool_1",
            description="This is test tool 1.",
            schema_params={"param1": {"type": "string"}, "param2": {"type": "integer"}}
        )
        self.mock_tool2 = MockTool(
            name="test_tool_2",
            description="This is test tool 2.",
            schema_params={"query": {"type": "string"}}
        )
        self.tools_list = [self.mock_tool1, self.mock_tool2]
        # We patch genai.GenerativeModel in the tests that use the LLM

    def tearDown(self):
        logging.disable(logging.NOTSET) # Re-enable logging
        self.loop.close()

    def test_format_tools_for_prompt(self):
        """Test the _format_tools_for_prompt method."""
        decision_module = Decision(tools_list=self.tools_list, model_name="gemini-1.5-flash-latest")
        formatted_string = decision_module._format_tools_for_prompt(self.tools_list)

        self.assertIn("Available Tools:", formatted_string)
        self.assertIn("Name: test_tool_1", formatted_string)
        self.assertIn("Description: This is test tool 1.", formatted_string)
        # Make assertion less brittle to exact spacing, or ensure it matches json.dumps(..., indent=2) output
        # For json.dumps(..., indent=2), the output for a nested property is typically:
        # "param1": {
        #   "type": "string"
        # }
        self.assertIn("\"param1\": {\n      \"type\": \"string\"\n    }", formatted_string)
        self.assertIn("Name: test_tool_2", formatted_string)

    def test_format_tools_for_prompt_no_tools(self):
        """Test _format_tools_for_prompt with an empty tools list."""
        decision_module = Decision(tools_list=[], model_name="gemini-1.5-flash-latest")
        formatted_string = decision_module._format_tools_for_prompt([])
        self.assertEqual(formatted_string, "No tools available.")

    @patch('module.decision.genai.GenerativeModel') # Assuming decision.py uses 'import google.generativeai as genai'
    def test_execute_plan_with_memory_final_answer(self, MockGenerativeModel):
        """Test execute_plan_with_memory when LLM decides on a final_answer."""
        # Configure the mock LLM
        mock_llm_instance = MockGenerativeModel.return_value
        mock_response_content = json.dumps({
            "thought": "User asked a simple question. No tool needed.",
            "tool_name": "final_answer",
            "tool_input": {},
            "memory_actions": [{"action": "store_fact", "key": "mood", "value": "happy"}],
            "speak": "I'm doing well, thank you for asking!"
        })
        mock_llm_instance.generate_content.return_value = MagicMock(
            candidates=[MagicMock(content=MagicMock(parts=[MagicMock(text=mock_response_content)]))]
        )

        async def run_test():
            decision_module = Decision(tools_list=self.tools_list, model_name="gemini-1.5-flash-latest")
            # Await the coroutine
            thought, tool_name, tool_input, speak, memory_actions = await decision_module.execute_plan_with_memory(
                user_query="How are you?",
                perception_data={}, 
                memory_context="Some facts here.",
                full_conversation_history=[{"role": "user", "content": "Hi"}]
            )

            self.assertEqual(thought, "User asked a simple question. No tool needed.")
            self.assertEqual(tool_name, "final_answer")
            self.assertEqual(tool_input, {})
            self.assertEqual(speak, "I'm doing well, thank you for asking!")
            self.assertEqual(memory_actions, [{"action": "store_fact", "key": "mood", "value": "happy"}])
            mock_llm_instance.generate_content.assert_called_once() # If generate_content is sync
            # If generate_content is async, you might need to mock its async version or how it's awaited
        self.loop.run_until_complete(run_test())

    @patch('module.decision.genai.GenerativeModel') # Assuming decision.py uses 'import google.generativeai as genai'
    def test_execute_plan_with_memory_tool_call(self, MockGenerativeModel):
        """Test execute_plan_with_memory when LLM decides to call a tool."""
        mock_llm_instance = MockGenerativeModel.return_value
        mock_response_content = json.dumps({
            "thought": "User wants to use test_tool_1.",
            "tool_name": "test_tool_1",
            "tool_input": {"param1": "hello", "param2": 123},
            "memory_actions": [],
            "speak": "Okay, I will use test_tool_1 for that."
        })
        mock_llm_instance.generate_content.return_value = MagicMock(
            candidates=[MagicMock(content=MagicMock(parts=[MagicMock(text=mock_response_content)]))]
        )

        async def run_test():
            decision_module = Decision(tools_list=self.tools_list, model_name="gemini-1.5-flash-latest")
            # Await the coroutine
            thought, tool_name, tool_input, speak, memory_actions = await decision_module.execute_plan_with_memory(
                user_query="Use test_tool_1 with hello and 123",
                perception_data={},
                memory_context="",
                full_conversation_history=[]
            )

            self.assertEqual(tool_name, "test_tool_1")
            self.assertEqual(tool_input, {"param1": "hello", "param2": 123})
            self.assertEqual(speak, "Okay, I will use test_tool_1 for that.")
            self.assertEqual(memory_actions, [])
            mock_llm_instance.generate_content.assert_called_once()
            # You can also assert that the prompt passed to generate_content contains expected parts
            args, _ = mock_llm_instance.generate_content.call_args
            prompt_sent_to_llm = args[0]
            self.assertIn("Name: test_tool_1", prompt_sent_to_llm)
            self.assertIn("User: Use test_tool_1 with hello and 123", prompt_sent_to_llm)
        self.loop.run_until_complete(run_test())

if __name__ == '__main__':
    unittest.main()