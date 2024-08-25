from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="repoforge",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="An AI-powered tool for generating repository structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/repoforge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        "torch>=1.9.0",
        "transformers>=4.9.2",
        "pygithub>=1.55",
        "python-dotenv>=0.19.0",
        "aiohttp>=3.7.4",
    ],
    entry_points={
        "console_scripts": [
            "repoforge=repoforge.main:main",
        ],
    },
)