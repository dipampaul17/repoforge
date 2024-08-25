import unittest
import torch
from src.ai.layout_interpreter import LayoutInterpreter, process_layout_predictions
from src.ai.gpt_interface import interpret_structure, generate_gitignore, generate_dockerfile

class TestAI(unittest.TestCase):
    def setUp(self):
        self.interpreter = LayoutInterpreter(768, 256, 128)
    
    def test_layout_interpreter(self):
        logits = torch.randn(1, 512, 768)
        embeddings = process_layout_predictions(logits, self.interpreter)
        self.assertEqual(embeddings.shape, torch.Size([1, 128]))
    
    @unittest.mock.patch('src.ai.gpt_interface.gpt4_request')
    async def test_interpret_structure(self, mock_gpt4_request):
        mock_gpt4_request.return_value = "src"
        embedding = torch.randn(128)
        result = await interpret_structure(embedding)
        self.assertEqual(result, "src")
    
    @unittest.mock.patch('src.ai.gpt_interface.gpt4_request')
    async def test_generate_gitignore(self, mock_gpt4_request):
        mock_gpt4_request.return_value = "*.pyc\n__pycache__/"
        structure = {"src": {}, "tests": {}}
        result = await generate_gitignore(structure)
        self.assertIn("*.pyc", result)
    
    @unittest.mock.patch('src.ai.gpt_interface.gpt4_request')
    async def test_generate_dockerfile(self, mock_gpt4_request):
        mock_gpt4_request.return_value = "FROM python:3.9\nWORKDIR /app"
        structure = {"src": {}, "requirements.txt": None}
        result = await generate_dockerfile(structure)
        self.assertIn("FROM python", result)

if __name__ == '__main__':
    unittest.main()