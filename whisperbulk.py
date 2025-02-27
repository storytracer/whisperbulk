#!/usr/bin/env python3
"""
WhisperBulk: A CLI tool for bulk transcribing audio files using 
various Whisper model implementations.
"""
import os
import sys
import asyncio
import logging
import json
import datetime
from pathlib import Path
from typing import List, Optional, Union, Any, Dict, Literal
from urllib.parse import urlparse

import click
import dotenv
import openai
import tenacity
from smart_open import open
from tqdm import tqdm

# Constants
AUDIO_EXTENSIONS = (".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm")

# Load environment variables from .env file
dotenv.load_dotenv()

# Setup logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "whisperbulk.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=log_file,
    filemode='a'
)
logger = logging.getLogger("whisperbulk")

# Configure OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# Configure Async OpenAI client
async_client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(Exception),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    stop=tenacity.stop_after_attempt(5),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying {retry_state.attempt_number}/5 after exception: "
        f"{retry_state.outcome.exception() if retry_state.outcome else 'Unknown error'}"
    ),
)
async def transcribe_file(file_path: Union[str, Path], model: str) -> Any:
    """Transcribe a single audio file using the specified Whisper model with retry logic."""
    logger.info(f"Transcribing {file_path} with model {model}")

    # Use the async OpenAI client directly
    # File reading still needs to be done in a thread pool since it's blocking I/O
    loop = asyncio.get_running_loop()
    file_content = await loop.run_in_executor(None, lambda: _read_file(file_path))
    
    # Now call the API using the async client
    response = await async_client.audio.transcriptions.create(
        file=file_content,
        model=model,
        response_format="verbose_json"
    )
    
    return response

def _read_file(file_path: Union[str, Path]) -> Any:
    """Read file from disk - helper function for thread pool executor."""
    # Make sure we return a file object that OpenAI's API can accept
    return open(str(file_path), "rb")


def format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm."""
    millisec = int(seconds * 1000) % 1000
    seconds = int(seconds)
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millisec:03d}"


def convert_to_srt(segments: List[Dict]) -> str:
    """Convert segments to SRT subtitle format."""
    srt_content = []
    for i, segment in enumerate(segments, 1):
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "").strip()
        
        if text:
            srt_content.append(f"{i}\n")
            srt_content.append(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            srt_content.append(f"{text}\n\n")
    
    return "".join(srt_content)


def save_transcription(
    result: Any, 
    output_path: Path, 
    format: Literal["json", "txt", "srt"] = "txt"
) -> None:
    """Save transcription results to a file in the specified format."""
    # Extract data from different result types
    data = None
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    elif isinstance(result, dict):
        data = result
    elif isinstance(result, str):
        data = {"text": result}
    else:
        data = {"text": str(result)}
    
    with open(str(output_path), "w", encoding="utf-8") as f:
        if format == "json":
            # Save as JSON
            json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == "txt":
            # Save as plain text
            f.write(data.get("text", str(data)))
        elif format == "srt":
            # Save as SRT subtitle format
            segments = data.get("segments", [])
            if segments:
                f.write(convert_to_srt(segments))
            else:
                # Fallback if no segments available
                f.write(data.get("text", str(data)))

    logger.info(f"Saved transcription to {output_path} in {format} format")


async def process_file(
    file_path: Union[str, Path],
    output_dir: Union[str, Path, None] = None,
    model: str = "whisper-1",
    formats: List[Literal["json", "txt", "srt"]] = ["txt"]
) -> None:
    """Process a single file and save the transcription in all requested formats."""
    try:
        # Use await with the async transcribe_file function
        result = await transcribe_file(file_path, model)
        file_path_obj = Path(file_path)
        
        # Save the result in each requested format
        for format in formats:
            # Determine output path
            if output_dir:
                output_path = Path(output_dir) / f"{file_path_obj.stem}.{format}"
            else:
                output_path = file_path_obj.with_suffix(f".{format}")

            # Create a bound function for the save operation
            save_task = lambda: save_transcription(result, output_path, format)
            
            # Run the file saving in a thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, save_task)

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise


async def process_files(
    files: List[Union[str, Path]],
    output_dir: Optional[str] = None,
    concurrency: int = 5,
    model: str = "whisper-1",
    formats: List[Literal["json", "txt", "srt"]] = ["txt"]
) -> None:
    """Process multiple files concurrently."""
    # Use a semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(concurrency)

    async def _process_with_semaphore(file_path):
        async with semaphore:
            return await process_file(file_path, output_dir, model, formats)

    # Set up progress bar
    progress = tqdm(
        total=len(files),
        desc="Transcribing files",
        position=0,
        leave=True
    )
    
    async def process_and_update(file_path):
        try:
            await _process_with_semaphore(file_path)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            # We're already using return_exceptions=True in gather, but we still 
            # need to handle exceptions here to ensure progress bar updates
        finally:
            # Always update progress, even if file processing fails
            progress.update(1)
    
    # Create tasks for all files and process them concurrently
    tasks = [process_and_update(file_path) for file_path in files]
    await asyncio.gather(*tasks, return_exceptions=True)
    progress.close()


def is_audio_file(filename: str) -> bool:
    """Check if a file is a supported audio format."""
    return filename.lower().endswith(AUDIO_EXTENSIONS)


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 URI."""
    parsed = urlparse(path)
    return parsed.scheme == "s3"


def ensure_output_dir(output_dir: Optional[str]) -> None:
    """Ensure the output directory exists if it's a local path."""
    if output_dir and not is_s3_path(output_dir):
        os.makedirs(output_dir, exist_ok=True)


def collect_local_files(input_path: str, recursive: bool) -> List[str]:
    """Collect local audio files to process."""
    files = []

    if os.path.isfile(input_path):
        files.append(input_path)
    elif os.path.isdir(input_path):
        if recursive:
            for root, _, filenames in os.walk(input_path):
                files.extend([
                    os.path.join(root, filename)
                    for filename in filenames if is_audio_file(filename)
                ])
        else:
            files = [
                os.path.join(input_path, filename)
                for filename in os.listdir(input_path)
                if (os.path.isfile(os.path.join(input_path, filename)) and
                    is_audio_file(filename))
            ]

            if not files:
                logger.warning(
                    f"{input_path} is a directory. "
                    f"Use --recursive to process subdirectories."
                )

    return files


def collect_s3_files(s3_uri: str) -> List[str]:
    """Collect audio files from an S3 bucket."""
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError

        files = []
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')

        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                files.extend([
                    f"s3://{bucket}/{obj['Key']}"
                    for obj in page['Contents'] if is_audio_file(obj['Key'])
                ])

        return files

    except (ImportError, NoCredentialsError, Exception) as e:
        logger.error(f"Error accessing S3: {e}")
        sys.exit(1)


def check_requirements() -> None:
    """Check required environment variables."""
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)


def check_aws_credentials(uses_s3: bool) -> None:
    """Warn if AWS credentials are missing when using S3."""
    if uses_s3 and not (
        os.environ.get("AWS_ACCESS_KEY_ID") and
        os.environ.get("AWS_SECRET_ACCESS_KEY")
    ):
        logger.warning("AWS credentials not found in environment variables")


@click.command()
@click.argument("input", required=True, type=str)
@click.argument("output", required=True, type=str)
@click.option(
    "--concurrency", "-c", default=5,
    help="Number of concurrent transcription requests"
)
@click.option(
    "--recursive", "-r", is_flag=True,
    help="Recursively process directories"
)
@click.option(
    "--verbose", "-v", is_flag=True,
    help="Enable verbose logging to the log file"
)
@click.option(
    "--log-file", type=str, default=None,
    help="Path to log file (default: logs/whisperbulk.log)"
)
@click.option(
    "--model", "-m", default="Systran/faster-whisper-small",
    help="Model to use for transcription"
)
@click.option(
    "--format", "-f", default=["txt"], multiple=True, type=click.Choice(["json", "txt", "srt"]),
    help="Output format(s) for transcriptions (can be used multiple times)"
)
def main(input, output, concurrency, recursive, verbose, log_file, model, format):
    """Bulk transcribe audio files using Whisper models.

    INPUT is the source directory or file (or s3:// URI).
    OUTPUT is the destination directory (or s3:// URI) for transcriptions.

    Supports local paths and S3 URIs (s3://bucket/path).

    Example usage:

        whisperbulk ./audio_files ./transcriptions -c 10 -r
        whisperbulk s3://mybucket/audio s3://mybucket/transcriptions -m openai/whisper-1 -f srt
        
    You can specify multiple output formats:
    
        whisperbulk ./audio_files ./transcriptions -f txt -f srt -f json
    """
    # Configure logging
    if log_file:
        # If user provided custom log file path
        custom_log_dir = os.path.dirname(log_file)
        if custom_log_dir:
            os.makedirs(custom_log_dir, exist_ok=True)
        
        # Reconfigure the logging to use the custom file
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=log_file,
            filemode='a'
        )
    elif verbose:
        # Just change the log level if using default log file
        logger.setLevel(logging.DEBUG)

    # Validate inputs and environment
    check_requirements()

    input_uses_s3 = is_s3_path(input)
    output_uses_s3 = is_s3_path(output)
    check_aws_credentials(input_uses_s3 or output_uses_s3)

    # Ensure output directory exists
    ensure_output_dir(output)

    # Collect files to process
    files_to_process = []

    if input_uses_s3:
        files_to_process = collect_s3_files(input)
    else:
        files_to_process = collect_local_files(input, recursive)

    if not files_to_process:
        logger.error("No audio files found to process")
        sys.exit(1)

    logger.info(
        f"Found {len(files_to_process)} files to process "
        f"with concurrency {concurrency}"
    )

    # Process files - convert tuple to list if necessary
    formats = list(format) if isinstance(format, tuple) else format
    
    # Ensure we have at least one format
    if not formats:
        formats = ["txt"]
        
    asyncio.run(process_files(files_to_process, output, concurrency, model, formats))

    logger.info(f"All files processed successfully in format(s): {', '.join(formats)}")


if __name__ == "__main__":
    main()
