# RepoForge

RepoForge is an AI-powered tool for generating repository structures from text descriptions or images. It streamlines project setup by creating comprehensive structures, including .gitignore files and Dockerfiles.

## Features

- Parse repository structures from text or images
- Generate appropriate .gitignore files, Dockerfiles, and READMEs
- Create repositories on multiple platforms (GitHub, GitLab, Bitbucket, Azure DevOps)
- Provide a FastAPI-based HTTP server for easy integration

## Installation

```bash
pip install repoforge
```

## Usage

### As a CLI tool

```bash
repoforge create --platform github --name my-project --input "src/\n  main.py\ntests/\n  test_main.py"
```

### As a server

```bash
uvicorn repoforge.main:app --reload
```

### API Usage

```python
import aiohttp
import asyncio

async def create_repo():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/create_repo",
            json={
                "platform": "github",
                "name": "my-new-repo",
                "input_type": "text",
                "text": "src/\n  main.py\ntests/\n  test_main.py"
            }
        ) as response:
            print(await response.json())

asyncio.run(create_repo())
```

## Configuration

Set the following environment variables:

- `GITHUB_TOKEN`: Your GitHub personal access token
- `GITLAB_TOKEN`: Your GitLab personal access token
- `BITBUCKET_TOKEN`: Your Bitbucket app password
- `AZURE_DEVOPS_PAT`: Your Azure DevOps personal access token
- `AZURE_DEVOPS_ORG`: Your Azure DevOps organization name

## Contributing

We welcome contributions! Please fork or clone the repo to get started.

## License

RepoForge is licensed under the MIT License.

## Acknowledgements

RepoForge is built using the following open source libraries:

- [LayoutLMv2](https://github.com/microsoft/unilm/tree/master/layoutlmv2)
- [Sentence Transformers](https://www.sbert.net/)
- [HDBSCAN](https://hdbscan.readthedocs.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
