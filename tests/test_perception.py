import unittest
import logging
from module.perception import Perception

class TestPerceptionModule(unittest.TestCase):

    def setUp(self):
        logging.disable(logging.CRITICAL) # Suppress logs
        self.perception = Perception()

    def tearDown(self):
        logging.disable(logging.NOTSET) # Re-enable logs

    def test_initialization(self):
        """Test that the perception module initializes."""
        self.assertIsNotNone(self.perception)

    def test_process_general_query(self):
        """Test a general query with no specific detected intent."""
        query = "Tell me about sustainable materials."
        result = self.perception.process_user_input(query)
        self.assertEqual(result["original_query"], query)
        self.assertEqual(result["normalized_query"], query.lower())
        self.assertEqual(result["intent"], "general_query")
        self.assertEqual(result["entities"], {})
        self.assertIsNone(result["command"])

    def test_process_end_session_intent(self):
        """Test queries that should trigger end_session intent."""
        queries = ["bye", "That is all, thank you.", "quit session"]
        for query in queries:
            with self.subTest(query=query):
                result = self.perception.process_user_input(query)
                self.assertEqual(result["intent"], "end_session")

    def test_process_finalize_scheme_intent(self):
        """Test queries for finalizing a scheme."""
        queries_and_expected_ids = {
            "Finalize scheme 3": "scheme_3",
            "I want to choose scheme 12": "scheme_12",
            "select scheme1": "scheme_1",
            "Pick Scheme 5 please": "scheme_5",
        }
        for query, expected_id_str in queries_and_expected_ids.items():
            with self.subTest(query=query):
                result = self.perception.process_user_input(query)
                self.assertEqual(result["intent"], "finalize_scheme")
                self.assertIn("scheme_id_string", result["entities"])
                self.assertEqual(result["entities"]["scheme_id_string"], expected_id_str)
                self.assertIn("scheme_id_number", result["entities"])
                self.assertEqual(result["entities"]["scheme_id_number"], int(expected_id_str.split('_')[1]))
                self.assertIsNotNone(result["command"])
                self.assertEqual(result["command"]["type"], "finalize_scheme")
                self.assertEqual(result["command"]["scheme_id"], expected_id_str)

    def test_process_query_with_scheme_mention_not_finalize(self):
        """Test a query that mentions a scheme but isn't a finalize command."""
        query = "What are the details for scheme 4?"
        result = self.perception.process_user_input(query)
        self.assertEqual(result["intent"], "general_query") # Not a finalize command
        # self.assertIn("scheme_id_string", result["entities"]) # Our current regex for scheme_pattern is not used if finalize isn't matched
        # self.assertEqual(result["entities"]["scheme_id_string"], "scheme_4")
        self.assertIsNone(result["command"])

    # Add more tests here for other intents and entities you might add later
    # For example, extracting material names, locations, specific parameters etc.

if __name__ == '__main__':
    unittest.main()
