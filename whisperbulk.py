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
import srt
import datetime
from pathlib import Path
from typing import List, Optional, Union, Any, Dict, Literal, Set
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
DERIVATIVE_FORMATS = ["txt", "srt"]

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


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(Exception),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    stop=tenacity.stop_after_attempt(5),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying {retry_state.attempt_number}/5 after exception: "
        f"{retry_state.outcome.exception() if retry_state.outcome else 'Unknown error'}"
    ),
)
async def transcribe_file(file_path: Union[str, Path], model: str) -> Dict:
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
            return response.model_dump() if hasattr(response, "model_dump") else response
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
        return response.model_dump() if hasattr(response, "model_dump") else response


class DerivativeConverter:
    """
    Handles conversion from JSON transcription data to derivative formats.
    """
    
    @staticmethod
    def to_text(transcription_data: Dict) -> str:
        """Convert JSON transcription data to plain text."""
        return transcription_data.get("text", "")
    
    @staticmethod
    def to_srt(transcription_data: Dict) -> str:
        """Convert JSON transcription data to SRT subtitle format."""
        segments = transcription_data.get("segments", [])
        if not segments:
            return transcription_data.get("text", "")
        
        subtitle_entries = []
        
        for i, segment in enumerate(segments, 1):
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            
            if text:
                # Convert seconds to timedelta objects as required by srt library
                start_time = datetime.timedelta(seconds=start)
                end_time = datetime.timedelta(seconds=end)
                
                # Create a subtitle entry
                subtitle = srt.Subtitle(
                    index=i,
                    start=start_time,
                    end=end_time,
                    content=text
                )
                
                subtitle_entries.append(subtitle)
        
        # Compose the full SRT content
        return srt.compose(subtitle_entries)


async def save_json_transcription(
    transcription_data: Dict,
    output_path: Union[str, Path]
) -> None:
    """Save transcription results as JSON."""
    # Format as JSON string
    content = json.dumps(transcription_data, indent=2, ensure_ascii=False)
    
    # Use the appropriate storage manager to write the file
    storage = get_storage_manager(output_path)
    await storage.write_text(output_path, content)
    
    logger.info(f"Saved JSON transcription to {output_path}")


async def save_derivative(
    transcription_data: Dict,
    output_path: Union[str, Path],
    format_name: Literal["txt", "srt"]
) -> None:
    """Save transcription results in a derivative format (txt or srt)."""
    # Convert to the requested format
    if format_name == "txt":
        content = DerivativeConverter.to_text(transcription_data)
    elif format_name == "srt":
        content = DerivativeConverter.to_srt(transcription_data)
    else:
        raise ValueError(f"Unsupported derivative format: {format_name}")
    
    # Use the appropriate storage manager to write the file
    storage = get_storage_manager(output_path)
    await storage.write_text(output_path, content)
    
    logger.info(f"Saved {format_name.upper()} derivative to {output_path}")


def get_output_path(
    file_path: Union[str, Path],
    fmt: str,
    output_dir: Optional[Union[str, Path]] = None,
    input_base_path: str = ""
) -> str:
    """
    Generate the output path for a given file and format, preserving directory structure.
    
    Args:
        file_path: The input audio file path
        fmt: The format extension (json, txt, srt)
        output_dir: Optional output directory, if not specified will save alongside input
        input_base_path: Original input path to correctly preserve directory structure
        
    Returns:
        The full output path for the given format
    """
    file_path_obj = Path(str(file_path))
    file_stem = file_path_obj.stem
    
    # Get relative path for preserving directory structure
    def get_relative_path():
        if isinstance(file_path, str) and is_s3_path(file_path):
            parsed = urlparse(file_path)
            key = parsed.path.lstrip('/')
            input_path_obj = Path(key)
            # Get parent directories to preserve structure
            return input_path_obj.parent
        else:
            # For local paths, get input directory
            input_base = Path(input_base_path)
            if input_base.is_file():
                input_base = input_base.parent
            # Get relative path from input base to file
            try:
                # Only compute relative path if file_path is within input directory
                rel_path = Path(file_path).resolve().relative_to(input_base.resolve())
                return rel_path.parent
            except ValueError:
                # If file isn't within input directory, just use filename
                return Path('.')
    
    if output_dir:
        if isinstance(output_dir, str) and is_s3_path(output_dir):
            # Handle S3 output path
            parsed = urlparse(output_dir)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip('/')
            if not prefix.endswith('/'):
                prefix += '/'
            
            # Preserve directory structure relative to input path
            rel_path = get_relative_path()
            if rel_path != Path('.'):
                prefix += f"{rel_path}/"
                
            return f"s3://{bucket}/{prefix}{file_stem}.{fmt}"
        else:
            # Handle local output path
            rel_path = get_relative_path()
            output_path = Path(output_dir)
            if rel_path != Path('.'):
                output_path = output_path / rel_path
                
            # Ensure the directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            
            return str(output_path / f"{file_stem}.{fmt}")
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
            new_key = f"{dir_part}/{stem}.{fmt}" if dir_part else f"{stem}.{fmt}"
            return f"s3://{bucket}/{new_key}"
        else:
            # For local input, use same directory
            return str(file_path_obj.with_suffix(f".{fmt}"))


async def process_file(
    file_path: Union[str, Path],
    output_dir: Union[str, Path, None] = None,
    model: str = "whisper-1",
    derivatives: Optional[List[Literal["txt", "srt"]]] = None,
    input_base_path: str = ""
) -> None:
    """
    Process a single file: transcribe to JSON and optionally create derivative formats.
    
    Args:
        file_path: Path to the audio file to process
        output_dir: Directory to save outputs (if None, saves alongside input)
        model: Whisper model to use for transcription
        derivatives: Derivative formats to generate (txt, srt)
        input_base_path: Original input path for preserving directory structure
    """
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Set empty list if derivatives is None
        derivatives_to_process = derivatives or []
        
        # Get JSON output path and check if it exists
        json_path = get_output_path(file_path, "json", output_dir, input_base_path)
        json_storage = get_storage_manager(json_path)
        json_exists = await json_storage.exists(json_path)
        
        # Initialize results and tasks
        transcription_data = None
        save_tasks = []
        
        # STEP 1: Try to use existing JSON if possible
        if json_exists:
            logger.info(f"Found existing JSON file: {json_path}")
            # Load and parse the JSON file
            try:
                json_content = await json_storage.read_binary(json_path)
                json_text = json_content.decode('utf-8')
                transcription_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing existing JSON file {json_path}: {e}")
                transcription_data = None  # Force re-transcription
        
        # STEP 2: If no valid JSON exists, perform transcription
        if not json_exists or transcription_data is None:
            logger.info(f"Transcribing: {file_path}")
            transcription_data = await transcribe_file(file_path, model)
            
            # Save JSON result
            save_tasks.append(save_json_transcription(
                transcription_data, 
                json_path
            ))
        
        # STEP 3: Create derivative formats if requested
        for fmt in derivatives_to_process:
            output_path = get_output_path(file_path, fmt, output_dir, input_base_path)
            
            # Check if this derivative already exists
            output_storage = get_storage_manager(output_path)
            if not await output_storage.exists(output_path):
                logger.info(f"Creating {fmt} derivative: {output_path}")
                save_tasks.append(save_derivative(
                    transcription_data,
                    output_path,
                    fmt
                ))
        
        # Run all save tasks concurrently for better performance
        if save_tasks:
            await asyncio.gather(*save_tasks)
            logger.info(f"Completed processing {file_path}")
        else:
            logger.info(f"No new outputs to generate for {file_path}")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        raise


async def scan_output_files(output_dir: Optional[str], formats: List[str]) -> Dict[str, bool]:
    """
    Efficiently scan and cache all existing output files in the output directory.
    
    Args:
        output_dir: Directory to scan for existing outputs
        formats: List of format extensions to look for
        
    Returns:
        Dictionary mapping file paths to True for quick lookup
    """
    existing_files: Dict[str, bool] = {}
    
    if not output_dir:
        # If no output directory is specified, we can't pre-cache the files
        # This typically means outputs are alongside inputs, which is handled differently
        return existing_files
    
    logger.info(f"Scanning existing output files in {output_dir}")
    progress = tqdm(desc="Scanning existing output files", unit="files", leave=True)
    
    storage = get_storage_manager(output_dir)
    
    # Collect files for all formats we're interested in
    for fmt in formats:
        pattern = f"*.{fmt}"
        try:
            files = await storage.list_files(output_dir, recursive=True, pattern=pattern)
            for file_path in files:
                existing_files[file_path] = True
                progress.update(1)
        except Exception as e:
            logger.warning(f"Error listing existing {fmt} files in {output_dir}: {e}")
    
    progress.close()
    logger.info(f"Found {len(existing_files)} existing output files in {output_dir}")
    return existing_files


def check_file_needs_processing(
    file_path: Union[str, Path],
    derivatives: Optional[List[str]],
    existing_files_cache: Dict[str, bool],
    output_dir: Optional[Union[str, Path]],
    input_base_path: str = ""
) -> tuple[bool, bool, Set[str]]:
    """
    Check if a file needs transcription and/or derivative creation.
    
    Args:
        file_path: Path to the audio file to check
        derivatives: List of derivative formats to generate
        existing_files_cache: Cache of existing output files
        output_dir: Output directory for generated files
        input_base_path: Original input path for preserving structure
        
    Returns:
        Tuple of (needs_transcription, needs_derivatives, missing_derivatives) where:
        - needs_transcription: True if JSON transcription needs to be generated
        - needs_derivatives: True if any derivative format needs to be generated
        - missing_derivatives: Set of derivative formats that need to be generated
    """
    # If no cache (force mode), process everything
    if not existing_files_cache:
        return True, bool(derivatives), set(derivatives or [])
    
    # Get path to JSON transcription
    json_path = get_output_path(file_path, "json", output_dir, input_base_path)
    needs_transcription = json_path not in existing_files_cache
    
    # Check which derivatives are missing
    missing_derivatives = set()
    if derivatives:
        for fmt in derivatives:
            output_path = get_output_path(file_path, fmt, output_dir, input_base_path)
            if output_path not in existing_files_cache:
                missing_derivatives.add(fmt)
    
    needs_derivatives = len(missing_derivatives) > 0
    
    return needs_transcription, needs_derivatives, missing_derivatives


async def process_files(
    files: List[str],
    output_dir: Optional[str] = None,
    concurrency: int = 5,
    model: str = "whisper-1",
    derivatives: Optional[List[Literal["txt", "srt"]]] = None,
    force: bool = False,
    input_base_path: str = ""
) -> None:
    """
    Process multiple files concurrently, skipping files that already have outputs unless force=True.
    
    Args:
        files: List of audio file paths to process
        output_dir: Directory to save output files
        concurrency: Number of concurrent transcription jobs
        model: Whisper model to use for transcription
        derivatives: Derivative formats to generate
        force: If True, process all files regardless of existing outputs
        input_base_path: Original input path for preserving directory structure
    """
    # Ensure derivatives is a list or None
    validated_derivatives = derivatives or []
    
    # Validate derivatives against supported list
    for fmt in validated_derivatives:
        if fmt not in DERIVATIVE_FORMATS:
            logger.warning(f"Ignoring unsupported derivative format: {fmt}")
            validated_derivatives.remove(fmt)
    
    # Step 1: Scan for existing output files if not in force mode
    scan_formats = ["json"] + validated_derivatives
    existing_files_cache = {}
    if not force and output_dir:
        existing_files_cache = await scan_output_files(output_dir, scan_formats)
    
    # Step 2: Determine which files need processing
    files_to_process = []
    processing_info = []  # Store (file_path, needs_transcription, missing_derivatives)
    
    # Set up progress bar for checking files
    check_progress = tqdm(
        total=len(files),
        desc="Checking files to process",
        position=0,
        leave=True
    )
    
    # Check files in batches to avoid memory issues with large file lists
    for file_path in files:
        # Check if file needs processing and which formats are missing
        needs_transcription, needs_derivatives, missing_derivatives = check_file_needs_processing(
            file_path, validated_derivatives, existing_files_cache, 
            output_dir, input_base_path
        )
        
        if needs_transcription or needs_derivatives:
            files_to_process.append(file_path)
            processing_info.append((file_path, needs_transcription, missing_derivatives))
        
        check_progress.update(1)
    
    # Close progress bar
    check_progress.close()
    
    # If nothing to process, we're done
    if not files_to_process:
        logger.info("No files need processing - all output files already exist")
        return
        
    logger.info(f"Processing {len(files_to_process)} of {len(files)} files "
               f"(skipping {len(files) - len(files_to_process)} with existing outputs)")
    
    # Step 3: Process files concurrently with semaphore to limit API calls
    semaphore = asyncio.Semaphore(concurrency)

    async def _process_with_semaphore(file_path, needs_transcription, missing_derivatives):
        async with semaphore:
            # If we don't need to transcribe or generate derivatives, skip this file
            if not needs_transcription and not missing_derivatives:
                return
                
            # Process the file with the required derivatives
            return await process_file(
                file_path, 
                output_dir, 
                model, 
                list(missing_derivatives) if not needs_transcription else validated_derivatives,
                input_base_path
            )

    # Set up progress bar for processing
    progress = tqdm(
        total=len(files_to_process),
        desc="Transcribing files",
        position=0,
        leave=True
    )
    
    async def process_and_update(file_info):
        file_path, needs_transcription, missing_derivatives = file_info
        try:
            await _process_with_semaphore(file_path, needs_transcription, missing_derivatives)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
        finally:
            # Always update progress, even if file processing fails
            progress.update(1)
    
    # Create tasks for all files and process them concurrently
    tasks = [process_and_update(info) for info in processing_info]
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
    """
    Collect audio files from any supported storage system.
    
    Args:
        input_path: The input directory or file path (local or S3)
        recursive: Whether to search subdirectories
        
    Returns:
        List of audio file paths found
    """
    # Use the appropriate storage manager
    storage = get_storage_manager(input_path)
    
    try:
        # For S3 storage, we need to use pagination which naturally gives us incremental updates
        if isinstance(storage, S3StorageManager):
            s3_files: List[str] = []
            parsed = urlparse(input_path)
            bucket = parsed.netloc
            prefix = parsed.path.lstrip('/')
            
            if not recursive and prefix and not prefix.endswith('/'):
                # If not recursive, we only want objects in this "directory"
                prefix = f"{prefix}/"
            
            print(f"Searching for audio files in s3://{bucket}/{prefix}...")
            
            # Create progress bar without a total (unknown at start)
            progress = tqdm(
                desc="Discovering audio files",
                unit="files",
                leave=True
            )
            
            async with storage._get_client() as client:
                paginator = client.get_paginator('list_objects_v2')
                
                async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    if 'Contents' in page:
                        # Filter results based on audio extensions
                        for obj in page['Contents']:
                            key = obj['Key']
                            
                            # Skip "directory" objects
                            if key.endswith('/'):
                                continue
                            
                            # Filter by audio extensions
                            if any(key.lower().endswith(ext) for ext in AUDIO_EXTENSIONS):
                                s3_files.append(f"s3://{bucket}/{key}")
                                progress.update(1)
            
            # Close the progress bar
            progress.close()
            
            if not s3_files:
                print("No audio files found.")
                
            return s3_files
                
        else:  # LocalStorageManager
            path_obj = Path(str(input_path))
            
            # If it's a single file, just check if it's audio and return
            if path_obj.is_file():
                if is_audio_file(str(path_obj)):
                    return [str(path_obj)]
                else:
                    return []
            
            print(f"Searching for audio files in {input_path}...")
            local_files: List[str] = []
            
            # Create a list of patterns for audio files
            patterns = [f"*{ext}" for ext in AUDIO_EXTENSIONS]
            
            # Create progress bar without a total (unknown at start)
            progress = tqdm(
                desc="Discovering audio files",
                unit="files",
                leave=True
            )
            
            # Process directory
            if recursive:
                for pattern in patterns:
                    # Use ** for recursive search with pathlib
                    glob_pattern = f"**/{pattern}"
                    for file_path in path_obj.glob(glob_pattern):
                        if file_path.is_file():  # Ensure it's a file
                            local_files.append(str(file_path))
                            progress.update(1)
            else:
                for pattern in patterns:
                    for file_path in path_obj.glob(pattern):
                        if file_path.is_file():  # Ensure it's a file
                            local_files.append(str(file_path))
                            progress.update(1)
            
            # Close the progress bar
            progress.close()
            
            if not local_files and path_obj.is_dir():
                print("No audio files found.")
                logger.warning(
                    f"{input_path} is a directory with no audio files. "
                    f"Use --recursive to process subdirectories if needed."
                )
            
            logger.info(f"Found {len(local_files)} audio files to process")
            return local_files
        
    except Exception as e:
        logger.error(f"Error collecting files from {input_path}: {e}")
        print(f"Error searching for files: {e}")
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
@click.argument("input_path", required=True, type=str)
@click.argument("output_path", required=True, type=str)
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
    "--model", "-m", default="whisper-1",
    help="Model to use for transcription"
)
@click.option(
    "--derivative", "-d", multiple=True, 
    type=click.Choice(DERIVATIVE_FORMATS),
    help="Derivative format(s) to generate from JSON transcriptions (can be used multiple times)"
)
@click.option(
    "--force", is_flag=True,
    help="Force processing of all files, even if output files already exist"
)
def main(input_path, output_path, concurrency, recursive, verbose, log_file, model, derivative, force):
    """Bulk transcribe audio files using Whisper models.

    INPUT_PATH is the source directory or file (or s3:// URI).
    OUTPUT_PATH is the destination directory (or s3:// URI) for transcriptions.

    Supports local paths and S3 URIs (s3://bucket/path).

    By default, the tool runs in resumable mode and only processes files that 
    don't already have corresponding output files for all requested formats.
    Use --force to process all files regardless of existing outputs.

    The tool generates JSON transcriptions and optionally creates derivative 
    formats (txt or srt) from these transcriptions.

    Example usage:

        whisperbulk ./audio_files ./transcriptions -c 10 -r
        whisperbulk s3://mybucket/audio s3://mybucket/transcriptions -m whisper-1 -d srt
        
    You can specify multiple derivative formats:
    
        whisperbulk ./audio_files ./transcriptions -d txt -d srt
        
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

    input_uses_s3 = is_s3_path(input_path)
    output_uses_s3 = is_s3_path(output_path)
    check_aws_credentials(input_uses_s3 or output_uses_s3)

    # Ensure local output directory exists
    if output_path and not is_s3_path(output_path):
        Path(output_path).mkdir(parents=True, exist_ok=True)

    # Use asyncio to run the entire pipeline
    async def run_pipeline():
        # Collect files to process
        files_to_process = await collect_files(input_path, recursive)
        
        if not files_to_process:
            logger.error("No audio files found to process")
            sys.exit(1)
            
        logger.info(
            f"Found {len(files_to_process)} files to process "
            f"with concurrency {concurrency}"
        )
        
        # Process files - convert tuple to list if necessary
        derivatives_list = list(derivative) if isinstance(derivative, tuple) else derivative
        
        # Log resumable mode (unless force is enabled)
        if not force:
            logger.info("Running in resumable mode - will skip files that already have outputs")
            
        # Process the files
        await process_files(
            files_to_process, 
            output_path, 
            concurrency, 
            model, 
            derivatives_list, 
            force, 
            input_path
        )
        
        # Log completion message with appropriate format information
        if derivatives_list:
            derivative_formats = ', '.join(derivatives_list)
            logger.info(f"All files processed successfully with JSON transcriptions and {derivative_formats} derivatives")
        else:
            logger.info("All files processed successfully with JSON transcriptions")
    
    # Run the async pipeline
    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()