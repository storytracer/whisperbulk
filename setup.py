from setuptools import setup, find_packages

setup(
    name="whisperbulk",
    version="0.1.0",
    description="A CLI tool for bulk transcribing audio files using OpenAI's Whisper API",
    author="Sebastian Majstorovic",
    author_email="storytracer@gmail.com",
    packages=find_packages(),
    py_modules=["whisperbulk"],
    install_requires=[
        "click",
        "openai",  # Ensure we use the version with the latest async API support
        "tenacity",
        "python-dotenv",
        "tqdm",
        "aiofiles",
        "srt",  # For SRT subtitle file generation
        "aiobotocore",  # For async S3 operations
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