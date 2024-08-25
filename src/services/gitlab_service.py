import gitlab
import os

def create_gitlab_repo(name, structure):
    gl = gitlab.Gitlab('https://gitlab.com', private_token=os.getenv('GITLAB_TOKEN'))
    project = gl.projects.create({'name': name})

    def create_gitlab_files(structure, path=''):
        for name, content in structure.items():
            file_path = f"{path}/{name}" if path else name
            if content is None:
                project.files.create({
                    'file_path': file_path,
                    'branch': 'main',
                    'content': '',
                    'commit_message': f"Create {file_path}"
                })
            else:
                create_gitlab_files(content, file_path)

    create_gitlab_files(structure)
    return project.web_url