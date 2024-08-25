import unittest
from unittest.mock import patch, MagicMock
from src.services.github_service import create_github_repo
from src.services.gitlab_service import create_gitlab_repo
from src.services.bitbucket_service import create_bitbucket_repo
from src.services.azure_devops_service import create_azure_devops_repo

class TestServices(unittest.TestCase):
    def setUp(self):
        self.test_structure = {
            "src": {"main.py": None},
            "tests": {"test_main.py": None},
            "README.md": None
        }

    @patch('src.services.github_service.Github')
    def test_create_github_repo(self, mock_github):
        mock_user = MagicMock()
        mock_repo = MagicMock()
        mock_user.create_repo.return_value = mock_repo
        mock_github.return_value.get_user.return_value = mock_user
        mock_repo.html_url = "https://github.com/test/repo"

        result = create_github_repo("test_repo", self.test_structure)
        self.assertEqual(result, "https://github.com/test/repo")
        mock_user.create_repo.assert_called_once_with("test_repo")

    @patch('src.services.gitlab_service.gitlab.Gitlab')
    def test_create_gitlab_repo(self, mock_gitlab):
        mock_project = MagicMock()
        mock_gitlab.return_value.projects.create.return_value = mock_project
        mock_project.web_url = "https://gitlab.com/test/repo"

        result = create_gitlab_repo("test_repo", self.test_structure)
        self.assertEqual(result, "https://gitlab.com/test/repo")
        mock_gitlab.return_value.projects.create.assert_called_once_with({'name': 'test_repo'})

    @patch('src.services.bitbucket_service.BitbucketClient')
    def test_create_bitbucket_repo(self, mock_bitbucket):
        mock_client = MagicMock()
        mock_bitbucket.return_value = mock_client
        mock_client.repositories.create.return_value = {
            'links': {'html': {'href': 'https://bitbucket.org/test/repo'}}
        }

        result = create_bitbucket_repo("test_repo", self.test_structure)
        self.assertEqual(result, "https://bitbucket.org/test/repo")
        mock_client.repositories.create.assert_called_once_with("test_repo")

    @patch('src.services.azure_devops_service.Connection')
    def test_create_azure_devops_repo(self, mock_connection):
        mock_git_client = MagicMock()
        mock_connection.return_value.clients.get_git_client.return_value = mock_git_client
        mock_project = MagicMock()
        mock_git_client.get_projects.return_value = [mock_project]
        mock_repo = MagicMock()
        mock_git_client.create_repository.return_value = mock_repo
        mock_repo.web_url = "https://dev.azure.com/test/repo"

        result = create_azure_devops_repo("test_repo", self.test_structure)
        self.assertEqual(result, "https://dev.azure.com/test/repo")
        mock_git_client.create_repository.assert_called_once_with("test_repo", mock_project.id)

if __name__ == '__main__':
    unittest.main()