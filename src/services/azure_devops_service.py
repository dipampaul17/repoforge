from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
import os

def create_azure_devops_repo(name, structure):
    credentials = BasicAuthentication('', os.getenv('AZURE_DEVOPS_PAT'))
    connection = Connection(base_url='https://dev.azure.com', creds=credentials)
    git_client = connection.clients.get_git_client()
    
    project = git_client.get_projects()[0]  # Assuming the first project
    repo = git_client.create_repository(name, project.id)

    def create_azure_devops_files(structure, path=''):
        for name, content in structure.items():
            file_path = f"{path}/{name}" if path else name
            if content is None:
                git_client.create_push([{
                    'path': file_path,
                    'content': ''
                }], repo.id, 'main')
            else:
                create_azure_devops_files(content, file_path)

    create_azure_devops_files(structure)
    return repo.web_url