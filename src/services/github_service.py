import aiohttp
import os
import base64
from typing import Dict, Any

GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

async def create_github_repo(name: str, structure: Dict[str, Any]) -> str:
    async with aiohttp.ClientSession() as session:
        # Create repository
        async with session.post(
            f"{GITHUB_API_URL}/user/repos",
            headers={"Authorization": f"token {GITHUB_TOKEN}"},
            json={"name": name, "auto_init": True}
        ) as response:
            if response.status != 201:
                raise Exception(f"Failed to create GitHub repository: {await response.text()}")
            repo_data = await response.json()

        # Create files
        await create_github_files(session, repo_data["full_name"], structure)

        return repo_data["html_url"]

async def create_github_files(session: aiohttp.ClientSession, repo_full_name: str, structure: Dict[str, Any], path: str = ""):
    for name, content in structure.items():
        file_path = f"{path}/{name}" if path else name
        if isinstance(content, dict):
            await create_github_files(session, repo_full_name, content, file_path)
        else:
            encoded_content = base64.b64encode(content.encode()).decode() if content else ""
            async with session.put(
                f"{GITHUB_API_URL}/repos/{repo_full_name}/contents/{file_path}",
                headers={"Authorization": f"token {GITHUB_TOKEN}"},
                json={
                    "message": f"Create {file_path}",
                    "content": encoded_content
                }
            ) as response:
                if response.status != 201:
                    raise Exception(f"Failed to create file {file_path}: {await response.text()}")

async def update_github_file(session: aiohttp.ClientSession, repo_full_name: str, file_path: str, content: str, sha: str):
    encoded_content = base64.b64encode(content.encode()).decode()
    async with session.put(
        f"{GITHUB_API_URL}/repos/{repo_full_name}/contents/{file_path}",
        headers={"Authorization": f"token {GITHUB_TOKEN}"},
        json={
            "message": f"Update {file_path}",
            "content": encoded_content,
            "sha": sha
        }
    ) as response:
        if response.status != 200:
            raise Exception(f"Failed to update file {file_path}: {await response.text()}")
