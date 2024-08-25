from bitbucket_api_client import BitbucketClient
import os

def create_bitbucket_repo(name, structure):
    client = BitbucketClient(os.getenv('BITBUCKET_TOKEN'))
    repo = client.repositories.create(name)

    def create_bitbucket_files(structure, path=''):
        for name, content in structure.items():
            file_path = f"{path}/{name}" if path else name
            if content is None:
                client.repositories.files.create(
                    repo['uuid'],
                    'main',
                    file_path,
                    ''
                )
            else:
                create_bitbucket_files(content, file_path)

    create_bitbucket_files(structure)
    return repo['links']['html']['href']