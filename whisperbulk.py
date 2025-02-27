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
import tempfile
import aiofiles
from pathlib import Path
from typing import List, Optional, Union, Any, Dict, Literal, BinaryIO
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

import click
import dotenv
from openai import AsyncOpenAI
import tenacity
from tqdm import tqdm

# Constants
AUDIO_EXTENSIONS = (".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm")

# Load environment variables from .env file
dotenv.load_dotenv()

# Setup logging
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "whisperbulk.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=log_file,
    filemode='a'
)
logger = logging.getLogger("whisperbulk")

openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class StorageManager(ABC):
    """Abstract base class for storage operations."""
    
    @abstractmethod
    async def exists(self, path: Union[str, Path]) -> bool:
        """Check if a file exists."""
        pass
    
    @abstractmethod
    async def read_binary(self, path: Union[str, Path]) -> bytes:
        """Read a file as binary data."""
        pass
    
    @abstractmethod
    async def write_text(self, path: Union[str, Path], content: str) -> None:
        """Write string content to a file."""
        pass
    
    @abstractmethod
    async def list_files(self, path: Union[str, Path], 
                         recursive: bool, pattern: Optional[str] = None) -> List[str]:
        """List files in a directory matching the pattern."""
        pass


class LocalStorageManager(StorageManager):
    """Manages local file operations using asyncio."""
    
    async def exists(self, path: Union[str, Path]) -> bool:
        """Check if a local file exists."""
        return Path(str(path)).exists()
    
    async def read_binary(self, path: Union[str, Path]) -> bytes:
        """Read a local file as binary data."""
        async with aiofiles.open(str(path), 'rb') as f:
            return await f.read()
    
    async def write_text(self, path: Union[str, Path], content: str) -> None:
        """Write string content to a local file."""
        # Ensure directory exists
        file_path = Path(str(path))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write content
        async with aiofiles.open(str(file_path), 'w', encoding='utf-8') as f:
            await f.write(content)
    
    async def list_files(self, path: Union[str, Path], 
                         recursive: bool, pattern: Optional[str] = None) -> List[str]:
        """List local files matching the pattern."""
        path_obj = Path(str(path))
        
        if path_obj.is_file():
            return [str(path_obj)]
        
        files = []
        
        # Handle pattern or use default audio extensions
        patterns = []
        if pattern:
            patterns = [pattern]
        else:
            # Use audio extensions for filtering
            patterns = [f"*{ext}" for ext in AUDIO_EXTENSIONS]
        
        # Use pathlib glob or rglob based on recursive flag
        for pattern in patterns:
            if recursive:
                # Use ** for recursive search with pathlib
                if path_obj.is_dir():
                    glob_pattern = f"**/{pattern}"
                    files.extend([str(p) for p in path_obj.glob(glob_pattern)])
            else:
                # Non-recursive glob
                files.extend([str(p) for p in path_obj.glob(pattern)])
        
        return files


class S3StorageManager(StorageManager):
    """Manages S3 operations using aiobotocore."""
    
    def __init__(self):
        """Initialize the S3 storage manager."""
        # Lazy import to avoid dependency issues
        import aiobotocore.session
        self.session = aiobotocore.session.get_session()
    
    @asynccontextmanager
    async def _get_client(self):
        """Create and yield an S3 client using aiobotocore."""
        async with self.session.create_client('s3') as client:
            yield client
    
    async def exists(self, path: Union[str, Path]) -> bool:
        """Check if a file exists in S3."""
        parsed = urlparse(str(path))
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        try:
            async with self._get_client() as client:
                await client.head_object(Bucket=bucket, Key=key)
                return True
        except Exception:
            return False
    
    async def read_binary(self, path: Union[str, Path]) -> bytes:
        """Read a file from S3 as binary data."""
        parsed = urlparse(str(path))
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        async with self._get_client() as client:
            response = await client.get_object(Bucket=bucket, Key=key)
            async with response['Body'] as stream:
                return await stream.read()
    
    async def write_text(self, path: Union[str, Path], content: str) -> None:
        """Write string content to an S3 file."""
        parsed = urlparse(str(path))
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        
        async with self._get_client() as client:
            await client.put_object(
                Bucket=bucket,
                Key=key,
                Body=content.encode('utf-8'),
                ContentType='text/plain'
            )
    
    async def list_files(self, path: Union[str, Path], 
                          recursive: bool, pattern: Optional[str] = None) -> List[str]:
        """List files in an S3 bucket/prefix."""
        parsed = urlparse(str(path))
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')
        
        if not recursive and prefix and not prefix.endswith('/'):
            # If not recursive, we only want objects in this "directory"
            prefix = f"{prefix}/"
        
        files = []
        async with self._get_client() as client:
            paginator = client.get_paginator('list_objects_v2')
            
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if 'Contents' in page:
                    # Filter results based on pattern or audio extensions
                    for obj in page['Contents']:
                        key = obj['Key']
                        
                        # Skip "directory" objects
                        if key.endswith('/'):
                            continue
                        
                        # If pattern is None, filter by audio extensions
                        if pattern is None and not any(key.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
                            continue
                            
                        # If pattern is provided, do basic matching
                        if pattern and not self._matches_pattern(key, pattern):
                            continue
                            
                        files.append(f"s3://{bucket}/{key}")
        
        return files
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Basic pattern matching for S3 keys."""
        import fnmatch
        filename = key.split('/')[-1]
        return fnmatch.fnmatch(filename, pattern)


def get_storage_manager(path: Union[str, Path]) -> StorageManager:
    """Factory function to get the appropriate storage manager."""
    if isinstance(path, str) and is_s3_path(path):
        try:
            return S3StorageManager()
        except ImportError:
            logger.error("aiobotocore is required for S3 operations. Install with 'pip install aiobotocore'")
            sys.exit(1)
    else:
        return LocalStorageManager()


# We don't need a custom file-like class anymore as we'll use temporary files


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
    
    # For S3 files or other remote files, we need to download them first
    if isinstance(file_path, str) and is_s3_path(file_path):
        # Use the storage manager to read the file
        storage = get_storage_manager(file_path)
        file_data = await storage.read_binary(file_path)
        
        # Create a temporary file
        file_name = Path(str(file_path)).name
        with tempfile.NamedTemporaryFile(suffix=file_name, delete=False) as temp_file:
            temp_file.write(file_data)
            temp_file_path = Path(temp_file.name)
        
        try:
            # Use the local file path for transcription
            with open(temp_file_path, "rb") as file_obj:
                response = await openai_client.audio.transcriptions.create(
                    file=file_obj, 
                    model=model,
                    response_format="verbose_json"
                )
            return response
        finally:
            # Clean up the temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
    else:
        # For local files, we can use them directly
        with open(str(file_path), "rb") as file_obj:
            response = await openai_client.audio.transcriptions.create(
                file=file_obj,
                model=model,
                response_format="verbose_json"
            )
        return response


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


async def save_transcription(
    result: Any, 
    output_path: Union[str, Path], 
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
    
    # Prepare content based on format
    content = ""
    if format == "json":
        # Convert to JSON string
        content = json.dumps(data, indent=2, ensure_ascii=False)
    elif format == "txt":
        # Get plain text
        content = data.get("text", str(data))
    elif format == "srt":
        # Convert to SRT subtitle format
        segments = data.get("segments", [])
        if segments:
            content = convert_to_srt(segments)
        else:
            # Fallback if no segments available
            content = data.get("text", str(data))
    
    # Use the appropriate storage manager to write the file
    storage = get_storage_manager(output_path)
    await storage.write_text(output_path, content)

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
        file_path_obj = Path(str(file_path))
        
        # Save the result in each requested format
        save_tasks = []
        for format in formats:
            # Determine output path
            if output_dir:
                if isinstance(output_dir, str) and is_s3_path(output_dir):
                    # Handle S3 output path
                    parsed = urlparse(output_dir)
                    bucket = parsed.netloc
                    prefix = parsed.path.lstrip('/')
                    if not prefix.endswith('/'):
                        prefix += '/'
                    output_path = f"s3://{bucket}/{prefix}{file_path_obj.stem}.{format}"
                else:
                    # Handle local output path
                    output_path = str(Path(output_dir) / f"{file_path_obj.stem}.{format}")
            else:
                # Store alongside input file
                if isinstance(file_path, str) and is_s3_path(file_path):
                    # For S3 input, keep in same bucket/prefix
                    parsed = urlparse(file_path)
                    bucket = parsed.netloc
                    key = parsed.path.lstrip('/')
                    
                    # Use pathlib to handle the path components
                    key_path = Path(key)
                    dir_part = str(key_path.parent) if key_path.parent != Path('.') else ""
                    stem = file_path_obj.stem
                    new_key = f"{dir_part}/{stem}.{format}" if dir_part else f"{stem}.{format}"
                    output_path = f"s3://{bucket}/{new_key}"
                else:
                    # For local input, use same directory
                    output_path = str(file_path_obj.with_suffix(f".{format}"))

            # Add the save task to our list
            save_tasks.append(save_transcription(result, output_path, format))
        
        # Run all save tasks concurrently
        await asyncio.gather(*save_tasks)

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise


async def should_process_file(
    file_path: Union[str, Path],
    output_dir: Optional[str],
    formats: List[Literal["json", "txt", "srt"]]
) -> bool:
    """Check if a file needs to be processed by verifying if output files exist."""
    file_path_obj = Path(str(file_path))
    file_stem = file_path_obj.stem
    
    # Create a list of expected output paths
    output_paths = []
    
    if output_dir:
        if isinstance(output_dir, str) and is_s3_path(output_dir):
            # Handle S3 output path
            parsed = urlparse(output_dir)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip('/')
            if not prefix.endswith('/'):
                prefix += '/'
            
            # Generate S3 output paths for each format
            for format in formats:
                output_path = f"s3://{bucket}/{prefix}{file_stem}.{format}"
                output_paths.append(output_path)
        else:
            # Handle local output paths
            for format in formats:
                output_path = str(Path(output_dir) / f"{file_stem}.{format}")
                output_paths.append(output_path)
    else:
        # Store alongside input file
        if isinstance(file_path, str) and is_s3_path(file_path):
            # For S3 input, keep in same bucket/prefix
            parsed = urlparse(file_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            # Use pathlib to handle the path components
            key_path = Path(key)
            dir_part = str(key_path.parent) if key_path.parent != Path('.') else ""
            stem = Path(file_stem).name
            
            # Generate S3 output paths for each format
            for format in formats:
                new_key = f"{dir_part}/{stem}.{format}" if dir_part else f"{stem}.{format}"
                output_path = f"s3://{bucket}/{new_key}"
                output_paths.append(output_path)
        else:
            # For local input, use same directory
            for format in formats:
                output_path = str(file_path_obj.with_suffix(f".{format}"))
                output_paths.append(output_path)
    
    # Check if all expected output files exist
    for output_path in output_paths:
        storage = get_storage_manager(output_path)
        if not await storage.exists(output_path):
            # If any output file doesn't exist, we should process this file
            return True
    
    # All output files exist, so we can skip this file
    logger.info(f"Skipping {file_path} as all output files already exist")
    return False

async def process_files(
    files: List[Union[str, Path]],
    output_dir: Optional[str] = None,
    concurrency: int = 5,
    model: str = "whisper-1",
    formats: List[Literal["json", "txt", "srt"]] = ["txt"],
    force: bool = False
) -> None:
    """Process multiple files concurrently, skipping files that already have outputs unless force=True."""
    # Filter files that need processing, or include all if force=True
    if force:
        files_to_process = files
        logger.info("Force flag enabled - processing all files regardless of existing outputs")
    else:
        # Check each file to see if it needs processing
        needs_processing_tasks = []
        for file_path in files:
            needs_processing_tasks.append(should_process_file(file_path, output_dir, formats))
        
        # Wait for all checks to complete
        needs_processing_results = await asyncio.gather(*needs_processing_tasks)
        
        # Filter files based on results
        files_to_process = [
            file_path for file_path, needs_processing 
            in zip(files, needs_processing_results) if needs_processing
        ]
    
    if not files_to_process:
        logger.info("No files need processing - all output files already exist")
        return
        
    logger.info(f"Processing {len(files_to_process)} of {len(files)} files (skipping {len(files) - len(files_to_process)} existing outputs)")
    
    # Use a semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(concurrency)

    async def _process_with_semaphore(file_path):
        async with semaphore:
            return await process_file(file_path, output_dir, model, formats)

    # Set up progress bar
    progress = tqdm(
        total=len(files_to_process),
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
    tasks = [process_and_update(file_path) for file_path in files_to_process]
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
        Path(output_dir).mkdir(parents=True, exist_ok=True)


async def collect_files(input_path: str, recursive: bool) -> List[str]:
    """Collect audio files from any supported storage system."""
    # Use the appropriate storage manager
    storage = get_storage_manager(input_path)
    
    try:
        # List files with the storage manager
        files = await storage.list_files(input_path, recursive)
        
        if not files and isinstance(storage, LocalStorageManager):
            path = Path(input_path)
            if path.is_dir():
                logger.warning(
                    f"{input_path} is a directory with no audio files. "
                    f"Use --recursive to process subdirectories if needed."
                )
        
        logger.info(f"Found {len(files)} audio files to process")
        return files
        
    except Exception as e:
        logger.error(f"Error collecting files from {input_path}: {e}")
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
@click.option(
    "--force", is_flag=True,
    help="Force processing of all files, even if output files already exist"
)
def main(input, output, concurrency, recursive, verbose, log_file, model, format, force):
    """Bulk transcribe audio files using Whisper models.

    INPUT is the source directory or file (or s3:// URI).
    OUTPUT is the destination directory (or s3:// URI) for transcriptions.

    Supports local paths and S3 URIs (s3://bucket/path).

    By default, the tool runs in resumable mode and only processes files that 
    don't already have corresponding output files for all requested formats.
    Use --force to process all files regardless of existing outputs.

    Example usage:

        whisperbulk ./audio_files ./transcriptions -c 10 -r
        whisperbulk s3://mybucket/audio s3://mybucket/transcriptions -m openai/whisper-1 -f srt
        
    You can specify multiple output formats:
    
        whisperbulk ./audio_files ./transcriptions -f txt -f srt -f json
        
    To reprocess all files regardless of existing outputs:
    
        whisperbulk ./audio_files ./transcriptions --force
    """
    # Configure logging
    if log_file:
        # If user provided custom log file path
        log_path = Path(log_file)
        if log_path.parent != Path('.'):
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Reconfigure the logging to use the custom file
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=log_path,
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

    # Ensure local output directory exists
    if output and not is_s3_path(output):
        Path(output).mkdir(parents=True, exist_ok=True)

    # Use asyncio to run the entire pipeline
    async def run_pipeline():
        # Collect files to process
        files_to_process = await collect_files(input, recursive)
        
        if not files_to_process:
            logger.error("No audio files found to process")
            sys.exit(1)
            
        logger.info(
            f"Found {len(files_to_process)} files to process "
            f"with concurrency {concurrency}"
        )
        
        # Process files - convert tuple to list if necessary
        formats_list = list(format) if isinstance(format, tuple) else format
        
        # Ensure we have at least one format
        if not formats_list:
            formats_list = ["txt"]
            
        # Log resumable mode (unless force is enabled)
        if not force:
            logger.info("Running in resumable mode - will skip files that already have outputs")
            
        # Process the files
        await process_files(files_to_process, output, concurrency, model, formats_list, force)
        
        logger.info(f"All files processed successfully in format(s): {', '.join(formats_list)}")
    
    # Run the async pipeline
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
