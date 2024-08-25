import asyncio
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from .ai.layout_interpreter import LayoutInterpreter, process_layout_predictions
from .ai.gpt_interface import interpret_structure, generate_gitignore, generate_dockerfile
from .utils.structure_parser import parse_structure_from_text
from .services.github_service import create_github_repo
import torch

app = FastAPI()

layout_interpreter = LayoutInterpreter(768, 256, 128)
layout_interpreter.load_state_dict(torch.load('layout_interpreter.pth'))

@app.post("/create_repo")
async def create_repo(
    platform: str = Form(...),
    name: str = Form(...),
    input_type: str = Form(...),
    file: UploadFile = File(None),
    text: str = Form(None)
):
    try:
        if input_type == 'image':
            # Process image with LayoutLMv2 (not implemented here)
            logits = torch.randn(1, 512, 768)  # Placeholder
            embeddings = process_layout_predictions(logits, layout_interpreter)
            structure = {await interpret_structure(emb): None for emb in embeddings}
        elif input_type == 'text':
            structure = parse_structure_from_text(text)
        else:
            return JSONResponse(status_code=400, content={"error": "Invalid input type"})

        gitignore = await generate_gitignore(structure)
        dockerfile = await generate_dockerfile(structure)

        structure['.gitignore'] = gitignore
        structure['Dockerfile'] = dockerfile

        if platform == 'github':
            repo_url = create_github_repo(name, structure)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported platform"})

        return {"repo_url": repo_url}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)