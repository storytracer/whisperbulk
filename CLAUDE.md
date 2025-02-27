# WhisperBulk Development Guidelines

## Commands

* Install: `pip install -e .`
* Run: `whisperbulk INPUT_PATH OUTPUT_PATH [options]`
* Test: `pytest tests/`
* Lint: `flake8 whisperbulk.py`
* Type check: `mypy whisperbulk.py`

## Code Style

* Follow PEP 8 guidelines
* Use type hints for all function parameters and return values
* Use docstrings for all functions, classes, and modules
* Handle errors with proper exception handling
* Use async/await for concurrent operations
* Use click for CLI argument parsing
* Import order: standard library → third-party → local modules
* Use f-strings for string formatting
* Use pathlib.Path for file path handling when appropriate

## Dependencies

* openai - API client for OpenAI services
* click - CLI framework
* tenacity - Retry mechanism
* smart_open - Unified interface for file/S3 operations
* dotenv - Environment variable management
* tqdm - Progress bar
* boto3 - AWS SDK (for S3 operations)