import aiohttp
import os
from typing import Dict, Any

GITLAB_API_URL = "https://gitlab.com/api/v4"
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")

async def create_gitlab_repo(name: str, structure: Dict[str, Any]) -> str:
    async with aiohttp.ClientSession() as session:
        # Create project
        async with session.post(
            f"{GITLAB_API_URL}/projects",
            headers={"PRIVATE-TOKEN": GITLAB_TOKEN},
            json={"name": name}
        ) as response:
            if response.status != 201:
                raise Exception(f"Failed to create GitLab project: {await response.text()}")
            project_data = await response.json()

        # Create files
        await create_gitlab_files(session, project_data["id"], structure)

        return project_data["web_url"]

async def create_gitlab_files(session: aiohttp.ClientSession, project_id: int, structure: Dict[str, Any], path: str = ""):
    for name, content in structure.items():
        file_path = f"{path}/{name}" if path else name
        if isinstance(content, dict):
            await create_gitlab_files(session, project_id, content, file_path)
        else:
            async with session.post(
                f"{GITLAB_API_URL}/projects/{project_id}/repository/files/{file_path}",
                headers={"PRIVATE-TOKEN": GITLAB_TOKEN},
                json={
                    "branch": "main",
                    "content": content or "",
                    "commit_message": f"Create {file_path}"
                }
            ) as response:
                if response.status != 201:
                    raise Exception(f"Failed to create file {file_path}: {await response.text()}")

async def update_gitlab_file(session: aiohttp.ClientSession, project_id: int, file_path: str, content: str):
    async with session.put(
        f"{GITLAB_API_URL}/projects/{project_id}/repository/files/{file_path}",
        headers={"PRIVATE-TOKEN": GITLAB_TOKEN},
        json={
            "branch": "main",
            "content": content,
            "commit_message": f"Update {file_path}"
        }
    ) as response:
        if response.status != 200:
            raise Exception(f"Failed to update file {file_path}: {await response.text()}")
