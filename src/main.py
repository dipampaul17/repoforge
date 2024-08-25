import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import aiohttp
from PIL import Image
import io
import torch
from transformers import LayoutLMv2Processor

from .ai.layout_interpreter import LayoutInterpreter, process_layout_predictions, cluster_embeddings, interpret_embedding, build_directory_structure
from .ai.gpt_interface import generate_gitignore, generate_dockerfile, generate_readme
from .utils.structure_parser import parse_structure_from_text
from .services.github_service import create_github_repo
from .services.gitlab_service import create_gitlab_repo
from .services.bitbucket_service import create_bitbucket_service
from .services.azure_devops_service import create_azure_devops_repo

app = FastAPI()

# Initialize models
layout_interpreter = LayoutInterpreter()
layout_interpreter.load_state_dict(torch.load('models/layout_interpreter.pth'))
layout_interpreter.eval()

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")

class RepoRequest(BaseModel):
    platform: str
    name: str
    input_type: str
    text: Optional[str] = None

@app.post("/create_repo")
async def create_repo(
    request: RepoRequest,
    file: Optional[UploadFile] = File(None)
):
    try:
        if request.input_type == 'image':
            if not file:
                raise HTTPException(status_code=400, detail="Image file is required for image input type")
            image = Image.open(io.BytesIO(await file.read()))
            encoding = processor(image, return_tensors="pt")
            with torch.no_grad():
                output = layout_interpreter(**encoding)
            embeddings = output.last_hidden_state.squeeze(0)
            clusters = cluster_embeddings(embeddings)
            interpreted_clusters = {interpret_embedding(emb): None for emb in embeddings[clusters != -1]}
            structure = build_directory_structure(interpreted_clusters)
        elif request.input_type == 'text':
            if not request.text:
                raise HTTPException(status_code=400, detail="Text input is required for text input type")
            structure = parse_structure_from_text(request.text)
        else:
            raise HTTPException(status_code=400, detail="Invalid input type")

        gitignore = await generate_gitignore(structure)
        dockerfile = await generate_dockerfile(structure)
        readme = await generate_readme(structure)

        structure['.gitignore'] = gitignore
        structure['Dockerfile'] = dockerfile
        structure['README.md'] = readme

        if request.platform == 'github':
            repo_url = await create_github_repo(request.name, structure)
        elif request.platform == 'gitlab':
            repo_url = await create_gitlab_repo(request.name, structure)
        elif request.platform == 'bitbucket':
            repo_url = await create_bitbucket_repo(request.name, structure)
        elif request.platform == 'azure':
            repo_url = await create_azure_devops_repo(request.name, structure)
        else:
            raise HTTPException(status_code=400, detail="Unsupported platform")

        return JSONResponse(content={"repo_url": repo_url, "structure": structure})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
