import unittest
from src.utils.structure_parser import parse_structure_from_text

class TestUtils(unittest.TestCase):
    def test_parse_structure_from_text(self):
        text = """
        src/
            main.py
            utils/
                helper.py
        tests/
            test_main.py
        README.md
        """
        expected_structure = {
            "src": {
                "main.py": None,
                "utils": {
                    "helper.py": None
                }
            },
            "tests": {
                "test_main.py": None
            },
            "README.md": None
        }
        result = parse_structure_from_text(text)
        self.assertEqual(result, expected_structure)

    def test_parse_structure_empty_text(self):
        text = ""
        result = parse_structure_from_text(text)
        self.assertEqual(result, {})

    def test_parse_structure_single_file(self):
        text = "README.md"
        expected_structure = {
            "README.md": None
        }
        result = parse_structure_from_text(text)
        self.assertEqual(result, expected_structure)

    def test_parse_structure_nested_directories(self):
        text = """
        src/
            main/
                app.py
            tests/
                test_app.py
        """
        expected_structure = {
            "src": {
                "main