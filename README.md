# RepoForge

RepoForge is a AI-driven CLI tool that leverages state-of-the-art AI models to generate complete repository structures from text descriptions or images. It simplifies project setup by creating comprehensive directory structures, .gitignore files, and Dockerfiles, allowing developers to focus on writing code.

## Features

- Create repository structures from text or image input
- Support for GitHub, GitLab, Bitbucket, and Azure DevOps
- Generates .gitignore and Dockerfile tailored to your project
- Uses LayoutLMv2 and LSTM for advanced layout interpretation
- HDBSCAN clustering for identifying directory structures
- Semantic understanding with Sentence Transformers
- Robust GPT-4 integration with retry mechanism
- FastAPI server for easy integration with other tools
- Async architecture for efficient request handling

## Installation

```bash
pip install repoforge
```

## Usage

### CLI

```bash
repoforge create --platform github --name my-project --input "src/\n  main.py\ntests/\n  test_main.py"
```

### Server

```bash
uvicorn repoforge.main:app --reload
```

### API

```python
import aiohttp
import asyncio

async def create_repo():
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:8000/create_repo",
            data={
                "platform": "github",
                "name": "my-new-repo",
                "input_type": "text", 
                "text": "src/\n  main.py\ntests/\n  test_main.py"
            }
        ) as response:
            print(await response.json())

asyncio.run(create_repo())
```

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
