import aiohttp
import os
from typing import Dict, Any

AZURE_DEVOPS_API_URL = f"https://dev.azure.com/{os.getenv('AZURE_DEVOPS_ORG')}"
AZURE_DEVOPS_PAT = os.getenv("AZURE_DEVOPS_PAT")

async def create_azure_devops_repo(name: str, structure: Dict[str, Any]) -> str:
    async with aiohttp.ClientSession() as session:
        # Create project
        project_id = await get_or_create_project(session, name)

        # Create repository
        async with session.post(
            f"{AZURE_DEVOPS_API_URL}/{project_id}/_apis/git/repositories?api-version=6.0",
            headers={"Authorization": f"Basic {AZURE_DEVOPS_PAT}"},
            json={"name": name}
        ) as response:
            if response.status != 201:
                raise Exception(f"Failed to create Azure DevOps repository: {await response.text()}")
            repo_data = await response.json()

        # Create files
        await create_azure_devops_files(session, project_id, repo_data["id"], structure)

        return repo_data["remoteUrl"]

async def get_or_create_project(session: aiohttp.ClientSession, name: str) -> str:
    async with session.get(
        f"{AZURE_DEVOPS_API_URL}/_apis/projects?api-version=6.0",
        headers={"Authorization": f"Basic {AZURE_DEVOPS_PAT}"}
    ) as response:
        projects = await response.json()
        for project in projects["value"]:
            if project["name"] == name:
                return project["id"]

    # Create new project if it doesn't exist
    async with session.post(
        f"{AZURE_DEVOPS_API_URL}/_apis/projects?api-version=6.0",
        headers={"Authorization": f"Basic {AZURE_DEVOPS_PAT}"},
        json={"name": name}
    ) as response:
        if response.status != 202:
            raise Exception(f"Failed to create Azure DevOps project: {await response.text()}")
        project_data = await response.json()
        return project_data["id"]

async def create_azure_devops_files(session: aiohttp.ClientSession, project_id: str, repo_id: str, structure: Dict[str, Any], path: str = ""):
    for name, content in structure.items():
        file_path = f"{path}/{name}" if path else name
        if isinstance(content, dict):
            await create_azure_devops_files(session, project_id, repo_id, content, file_path)
        else:
            async with session.post(
                f"{AZURE_DEVOPS_API_URL}/{project_id}/_apis/git/repositories/{repo_id}/pushes?api-version=6.0",
                headers={"Authorization": f"Basic {AZURE_DEVOPS_PAT}"},
                json={
                    "refUpdates": [{"name": "refs/heads/main", "oldObjectId": "0000000000000000000000000000000000000000"}],
                    "commits": [{
                        "comment": f"Create {file_path}",
                        "changes": [{
                            "changeType": "add",
                            "item": {"path": file_path},
                            "newContent": {"content": content or "", "contentType": "rawtext"}
                        }]
                    }]
                }
            ) as response:
                if response.status != 201:
                    raise Exception(f"Failed to create file {file_path}: {await response.text()}")

async def update_azure_devops_file(session: aiohttp.ClientSession, project_id: str, repo_id: str, file_path: str, content: str):
    async with session.post(
        f"{AZURE_DEVOPS_API_URL}/{project_id}/_apis/git/repositories/{repo_id}/pushes?api-version=6.0",
        headers={"Authorization": f"Basic {AZURE_DEVOPS_PAT}"},
        json={
            "refUpdates": [{"name": "refs/heads/main"}],
            "commits": [{
                "comment": f"Update {file_path}",
                "changes": [{
                    "changeType": "edit",
                    "item": {"path": file_path},
                    "newContent": {"content": content, "contentType": "rawtext"}
                }]
            }]
        }
    ) as response:
        if response.status != 201:
            raise Exception(f"Failed to update file {file_path}: {await response.text()}")
