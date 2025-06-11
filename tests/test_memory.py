import unittest
import uuid
import os
import sys
# Add the project root directory (FinalProj) to the Python path
# This assumes test_memory.py is in FinalProj/tests/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from module.memory import Memory # Adjust import path if your project structure is different

class TestMemory(unittest.TestCase):

    def setUp(self):
        """
        This method is called before each test function.
        We create a new Memory instance for each test to ensure isolation.
        """
        self.session_id = str(uuid.uuid4())
        self.memory = Memory(session_id=self.session_id)

    def test_initialization(self):
        """Test basic initialization of the Memory class."""
        self.assertEqual(self.memory.session_id, self.session_id)
        self.assertEqual(self.memory.conversation_history, [])
        self.assertEqual(self.memory.facts, {})
        self.assertEqual(self.memory.schemes_data, {})
        self.assertIsNone(self.memory.current_scheme_id)

    def test_store_and_retrieve_fact(self):
        """Test storing and retrieving a single fact."""
        self.memory.store_fact("test_key", "test_value")
        self.assertEqual(self.memory.retrieve_fact("test_key"), "test_value")

    def test_retrieve_non_existent_fact(self):
        """Test retrieving a fact that doesn't exist."""
        self.assertIsNone(self.memory.retrieve_fact("non_existent_key"))

    def test_overwrite_fact(self):
        """Test overwriting an existing fact."""
        self.memory.store_fact("test_key", "initial_value")
        self.memory.store_fact("test_key", "updated_value")
        self.assertEqual(self.memory.retrieve_fact("test_key"), "updated_value")

    def test_get_all_facts(self):
        """Test retrieving all facts."""
        self.memory.store_fact("key1", "value1")
        self.memory.store_fact("key2", 123)
        all_facts = self.memory.get_all_facts()
        self.assertEqual(len(all_facts), 2)
        self.assertEqual(all_facts["key1"], "value1")
        self.assertEqual(all_facts["key2"], 123)

    def test_add_conversation_turn(self):
        """Test adding a user and agent turn to conversation history."""
        self.memory.add_conversation_turn(role="user", content="Hello Agent")
        self.memory.add_conversation_turn(role="agent", content="Hello User", final_answer="Hello User, how can I help?")

        history = self.memory.get_conversation_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Hello Agent")
        self.assertEqual(history[1]["role"], "agent")
        self.assertEqual(history[1]["content"], "Hello User")
        self.assertEqual(history[1]["final_answer"], "Hello User, how can I help?")

    def test_get_conversation_history_last_n(self):
        """Test retrieving the last N turns of conversation history."""
        self.memory.add_conversation_turn(role="user", content="Turn 1")
        self.memory.add_conversation_turn(role="agent", content="Turn 2")
        self.memory.add_conversation_turn(role="user", content="Turn 3")
        self.memory.add_conversation_turn(role="agent", content="Turn 4")

        last_2_turns = self.memory.get_conversation_history(last_n=2)
        self.assertEqual(len(last_2_turns), 2)
        self.assertEqual(last_2_turns[0]["content"], "Turn 3") # user
        self.assertEqual(last_2_turns[1]["content"], "Turn 4") # agent

    def test_check_for_repeated_question_found(self):
        """Test finding a repeated question."""
        user_query1 = "What is the capital of France?"
        agent_answer1 = "The capital of France is Paris."
        self.memory.add_conversation_turn(role="user", content=user_query1)
        self.memory.add_conversation_turn(role="agent", content="Thinking...", final_answer=agent_answer1)
        self.memory.add_conversation_turn(role="user", content="Tell me about its weather.") # Intermediary
        self.memory.add_conversation_turn(role="agent", content="It's mild.")

        repeated_answer = self.memory.check_for_repeated_question(user_query1)
        self.assertEqual(repeated_answer, agent_answer1)

    def test_check_for_repeated_question_not_found(self):
        """Test when a question is not repeated."""
        self.memory.add_conversation_turn(role="user", content="What is the capital of Germany?")
        self.memory.add_conversation_turn(role="agent", content="The capital of Germany is Berlin.")
        
        repeated_answer = self.memory.check_for_repeated_question("What is the capital of France?")
        self.assertIsNone(repeated_answer)

    def test_check_for_repeated_question_within_lookback(self):
        """Test repeated question within lookback limit."""
        base_query = "Query "
        for i in range(6): # Add 6 user turns, 6 agent turns
            self.memory.add_conversation_turn(role="user", content=f"{base_query}{i}")
            self.memory.add_conversation_turn(role="agent", content=f"Answer {i}", final_answer=f"Final Answer {i}")
        
        # This query was asked (Query 5), should be found with lookback 5
        self.assertEqual(self.memory.check_for_repeated_question(f"{base_query}5", history_lookback=5), f"Final Answer 5")
        # This query was asked (Query 0), should NOT be found with lookback 5 (it's the 6th user query from the end)
        self.assertIsNone(self.memory.check_for_repeated_question(f"{base_query}0", history_lookback=5))

    def test_store_and_retrieve_scheme_data(self):
        """Test storing and retrieving scheme data."""
        scheme_id = "scheme_A"
        data = {"name": "Alpha Scheme", "area": 100}
        self.memory.store_scheme_data(scheme_id, data)
        retrieved_data = self.memory.retrieve_scheme_data(scheme_id)
        self.assertEqual(retrieved_data, data)

    def test_update_scheme_data(self):
        """Test updating existing scheme data."""
        scheme_id = "scheme_B"
        initial_data = {"name": "Beta Scheme", "floors": 2}
        update_data = {"floors": 3, "status": "updated"}
        expected_data = {"name": "Beta Scheme", "floors": 3, "status": "updated"}

        self.memory.store_scheme_data(scheme_id, initial_data)
        self.memory.store_scheme_data(scheme_id, update_data) # Default overwrite=False updates
        retrieved_data = self.memory.retrieve_scheme_data(scheme_id)
        self.assertEqual(retrieved_data, expected_data)

    def test_overwrite_scheme_data(self):
        """Test overwriting scheme data completely."""
        scheme_id = "scheme_C"
        initial_data = {"name": "Charlie Scheme", "param1": "old"}
        new_data = {"name": "Charlie New", "param2": "new"}

        self.memory.store_scheme_data(scheme_id, initial_data)
        self.memory.store_scheme_data(scheme_id, new_data, overwrite=True)
        retrieved_data = self.memory.retrieve_scheme_data(scheme_id)
        self.assertEqual(retrieved_data, new_data)

    def test_get_all_schemes_data(self):
        """Test retrieving all schemes data."""
        self.memory.store_scheme_data("s1", {"name": "Scheme 1"})
        self.memory.store_scheme_data("s2", {"name": "Scheme 2"})
        all_schemes = self.memory.get_all_schemes_data()
        self.assertEqual(len(all_schemes), 2)
        self.assertIn("s1", all_schemes)
        self.assertIn("s2", all_schemes)

    def test_set_and_get_current_scheme(self):
        """Test setting and getting the current scheme."""
        # A scheme needs to exist in schemes_data or as a fact for set_current_scheme to succeed
        self.memory.store_scheme_data("scheme_test_1", {"detail": "some detail"})
        self.assertTrue(self.memory.set_current_scheme("scheme_test_1"))
        self.assertEqual(self.memory.get_current_scheme_id(), "scheme_test_1")
        self.assertEqual(self.memory.retrieve_fact("current_scheme_id"), "scheme_test_1")

    def test_set_current_scheme_non_existent(self):
        """Test setting a non-existent scheme as current."""
        self.assertFalse(self.memory.set_current_scheme("scheme_ghost"))
        self.assertIsNone(self.memory.get_current_scheme_id())

    def test_get_and_update_current_scheme_data(self):
        """Test getting and updating data for the current scheme."""
        scheme_id = "active_scheme"
        self.memory.store_scheme_data(scheme_id, {"status": "initial"})
        self.memory.set_current_scheme(scheme_id)

        current_data = self.memory.get_current_scheme_data()
        self.assertEqual(current_data, {"status": "initial"})

        self.assertTrue(self.memory.update_current_scheme_data({"status": "modified", "new_field": True}))
        updated_data = self.memory.get_current_scheme_data()
        self.assertEqual(updated_data, {"status": "modified", "new_field": True})

    def test_store_and_retrieve_generated_schemes(self):
        """Test storing and retrieving generated schemes."""
        schemes_to_store = [
            {"id_val": "s1", "name": "Generated Scheme 1", "description": "First gen"},
            {"id_val": "s2", "name": "Generated Scheme 2"}
        ]
        generated_ids = self.memory.store_generated_schemes(schemes_to_store)
        self.assertEqual(generated_ids, ["scheme_1", "scheme_2"])
        self.assertEqual(self.memory.retrieve_generated_schemes_summary(), ["scheme_1", "scheme_2"])

        retrieved_details = self.memory.retrieve_all_generated_schemes_details()
        self.assertIsNotNone(retrieved_details)
        self.assertEqual(len(retrieved_details), 2)
        # The store_generated_schemes uses scheme_detail directly, so id_val should be there
        self.assertEqual(retrieved_details[0].get("name"), "Generated Scheme 1")
        self.assertEqual(retrieved_details[1].get("name"), "Generated Scheme 2")
        self.assertIsNotNone(self.memory.retrieve_scheme_data("scheme_1"))

    def test_get_relevant_context_for_decision_basic(self):
        """Test basic structure of get_relevant_context_for_decision."""
        self.memory.store_fact("project_name", "EcoTower")
        self.memory.add_conversation_turn(role="user", content="Hello")
        context = self.memory.get_relevant_context_for_decision(max_context_str_length=200)
        self.assertIsInstance(context, str)
        self.assertIn("EcoTower", context)
        self.assertIn("User: Hello", context)
        self.assertTrue(len(context) <= 200)

    def test_clear_memory(self):
        """Test clearing the memory."""
        self.memory.store_fact("temp_fact", "to_be_cleared")
        self.memory.add_conversation_turn(role="user", content="temp_query")
        self.memory.store_scheme_data("temp_scheme", {"data": "some_data"})
        self.memory.set_current_scheme("temp_scheme") # This also stores a fact

        self.memory.clear()

        self.assertEqual(self.memory.facts, {})
        self.assertEqual(self.memory.conversation_history, [])
        self.assertEqual(self.memory.schemes_data, {})
        self.assertIsNone(self.memory.current_scheme_id)

if __name__ == '__main__':
    unittest.main()
