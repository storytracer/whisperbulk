from setuptools import setup, find_packages

setup(
    name="whisperbulk",
    version="0.1.0",
    description="A CLI tool for bulk transcribing audio files using OpenAI's Whisper API",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    py_modules=["whisperbulk"],
    install_requires=[
        "click",
        "openai",
        "tenacity",
        "python-dotenv",
        "tqdm",
        "aiofiles",
        "aiohttp",
        "aiobotocore",
    ],
    entry_points={
        "console_scripts": [
            "whisperbulk=whisperbulk:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)