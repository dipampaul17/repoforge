# RepoForge

RepoForge is a developer tool that leverages AI to generate repository structures from text descriptions or images. It streamlines project setup by creating comprehensive structures, including .gitignore files and Dockerfiles.

## Key Features

- Parse repo structures from text or images
- AI-powered .gitignore and Dockerfile generation
- Multi-platform support (GitHub, GitLab, Bitbucket, Azure DevOps)
- FastAPI-based HTTP server for seamless integration

## Quick Start

```bash
# Install
pip install repoforge

# Use as a CLI tool
repoforge create --platform github --name my-project --input "src/\n  main.py\ntests/\n  test_main.py"

# Or run as a server
uvicorn repoforge.main:app --reload
```

## API Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/create_repo",
    data={
        "platform": "github",
        "name": "my-new-repo",
        "input_type": "text",
        "text": "src/\n  main.py\ntests/\n  test_main.py"
    }
)
print(response.json()["repo_url"])
```

## Contributing

We welcome contributions! Please fork or clone this repo.

## License

RepoForge is MIT licensed.
