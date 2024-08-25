import aiohttp
import os

async def gpt4_request(prompt, system_message):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500
            }
        ) as response:
            data = await response.json()
            return data['choices'][0]['message']['content'].strip()

async def interpret_structure(embedding):
    prompt = f"Interpret this embedding as a directory or file name: {embedding.tolist()}"
    system_message = "You are an AI that interprets embeddings of directory structures."
    return await gpt4_request(prompt, system_message)

async def generate_gitignore(structure):
    prompt = f"Generate a .gitignore for this structure:\n{structure}"
    system_message = "You are an AI that generates .gitignore files for software projects."
    return await gpt4_request(prompt, system_message)

async def generate_dockerfile(structure):
    prompt = f"Generate a Dockerfile for this structure:\n{structure}"
    system_message = "You are an AI that generates Dockerfiles for software projects."
    return await gpt4_request(prompt, system_message)