# RepoForge

RepoForge is a developer tool that leverages AI to generate repository structures from text descriptions or images. It streamlines project setup by creating comprehensive structures, including .gitignore files and Dockerfiles.

key features

- advanced layout interpretation using layoutlmv2 and lstm
- clustering with hdbscan
- semantic understanding with sentence transformers
- gpt-4 integration with retry mechanism

quick start
bashCopy# install
pip install repoforge

# use as a cli tool
repoforge create --platform github --name my-project --input "src/\n  main.py\ntests/\n  test_main.py"

# or run as a server
uvicorn repoforge.main:app --reload
api usage
pythonCopyimport aiohttp
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

## contributing

contributions welcome! Please fork or clone this repo.

## license

RepoForge is MIT licensed.
