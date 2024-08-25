import aiohttp
import os
from typing import Dict, Any

BITBUCKET_API_URL = "https://api.bitbucket.org/2.0"
BITBUCKET_TOKEN = os.getenv("BITBUCKET_TOKEN")

async def create_bitbucket_repo(name: str, structure: Dict[str, Any]) -> str:
    async with aiohttp.ClientSession() as session:
        # Create repository
        async with session.post(
            f"{BITBUCKET_API_URL}/repositories/{os.getenv('BITBUCKET_USERNAME')}/{name}",
            headers={"Authorization": f"Bearer {BITBUCKET_TOKEN}"},
            json={"scm": "git"}
        ) as response:
            if response.status != 200:
                raise Exception(f"Failed to create Bitbucket repository: {await response.text()}")
            repo_data = await response.json()

        # Create files
        await create_bitbucket_files(session, repo_data["full_name"], structure)

        return repo_data["links"]["html"]["href"]

async def create_bitbucket_files(session: aiohttp.ClientSession, repo_full_name: str, structure: Dict[str, Any], path: str = ""):
    for name, content in structure.items():
        file_path = f"{path}/{name}" if path else name
        if isinstance(content, dict):
            await create_bitbucket_files(session, repo_full_name, content, file_path)
        else:
            async with session.put(
                f"{BITBUCKET_API_URL}/repositories/{repo_full_name}/src",
                headers={"Authorization": f"Bearer {BITBUCKET_TOKEN}"},
                data={file_path: content or ""}
            ) as response:
                if response.status != 201:
                    raise Exception(f"Failed to create file {file_path}: {await response.text()}")

async def update_bitbucket_file(session: aiohttp.ClientSession, repo_full_name: str, file_path: str, content: str):
    async with session.put(
        f"{BITBUCKET_API_URL}/repositories/{repo_full_name}/src",
        headers={"Authorization": f"Bearer {BITBUCKET_TOKEN}"},
        data={file_path: content}
    ) as response:
        if response.status != 201:
            raise Exception(f"Failed to update file {file_path}: {await response.text()}")
