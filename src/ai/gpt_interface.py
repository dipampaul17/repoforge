import aiohttp
import os
import json
from typing import Dict, Any

GPT_MODEL = "gpt-4"
API_URL = "https://api.openai.com/v1/chat/completions"

async def gpt4_request(prompt: str, system_message: str) -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GPT_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=data) as response:
            if response.status != 200:
                raise Exception(f"GPT-4 API request failed with status {response.status}")
            response_data = await response.json()
            return response_data['choices'][0]['message']['content'].strip()

async def interpret_structure(embedding: Any) -> str:
    prompt = f"Interpret this embedding as a directory or file name: {embedding.tolist()}"
    system_message = "You are an AI that interprets embeddings of directory structures. Respond with only the interpreted name, nothing else."
    return await gpt4_request(prompt, system_message)

async def generate_gitignore(structure: Dict[str, Any]) -> str:
    prompt = f"Generate a .gitignore file for a project with the following structure:\n{json.dumps(structure, indent=2)}"
    system_message = "You are an AI that generates .gitignore files for software projects. Respond with only the content of the .gitignore file, no explanations."
    return await gpt4_request(prompt, system_message)

async def generate_dockerfile(structure: Dict[str, Any]) -> str:
    prompt = f"Generate a Dockerfile for a project with the following structure:\n{json.dumps(structure, indent=2)}"
    system_message = "You are an AI that generates Dockerfiles for software projects. Respond with only the content of the Dockerfile, no explanations."
    return await gpt4_request(prompt, system_message)

async def generate_readme(structure: Dict[str, Any]) -> str:
    prompt = f"Generate a README.md file for a project with the following structure:\n{json.dumps(structure, indent=2)}"
    system_message = "You are an AI that generates README.md files for software projects. Create a concise, informative README."
    return await gpt4_request(prompt, system_message)
