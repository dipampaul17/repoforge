from github import Github
import os

def create_github_repo(name, structure):
    g = Github(os.getenv('GITHUB_TOKEN'))
    user = g.get_user()
    repo = user.create_repo(name)

    def create_github_files(structure, path=''):
        for name, content in structure.items():
            file_path = f"{path}/{name}" if path else name
            if content is None:
                repo.create_file(file_path, f"Create {file_path}", "")
            else:
                create_github_files(content, file_path)

    create_github_files(structure)
    return repo.html_url