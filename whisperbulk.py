#!/usr/bin/env python3
"""
WhisperBulk: A CLI tool for bulk transcribing audio files using
various Whisper model implementations.
"""
import sys
import asyncio
import logging
import json
import aiofiles
import srt
import datetime
import re
from typing import List, Optional, Union, Any, Dict, Literal, Set, Tuple
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

import click
import dotenv
from openai import AsyncOpenAI
import tenacity
from tqdm import tqdm
from upath import UPath
from aiobotocore.session import get_session

# Constants
AUDIO_EXTENSIONS = ("mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm")
DERIVATIVE_FORMATS = ("txt", "srt")
PROGRESS_BAR_WIDTH = 75
DEFAULT_MODEL = "Systran/faster-whisper-medium"

# Load environment variables from .env file
dotenv.load_dotenv()

# Setup logging
log_dir = UPath(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "whisperbulk.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=str(log_file),
    filemode='a'
)
logger = logging.getLogger("whisperbulk")

# Initialize OpenAI client with API key from environment
openai_client = AsyncOpenAI()


class FileUtils:
    """Static utility methods for file operations."""
    
    @staticmethod
    def get_extension(path: UPath) -> str:
        """Get the file extension without the leading period."""
        return path.suffix.lower().lstrip('.')
    
    @staticmethod
    def is_audio_file(path: UPath) -> bool:
        """Check if a file is a supported audio format."""
        suffix = FileUtils.get_extension(path)
        return suffix in AUDIO_EXTENSIONS
    
    @staticmethod
    def is_output_file(path: UPath, output_files: Dict[str, Set[UPath]]) -> bool:
        """Check if a file is a supported output format and add it to the output files dictionary."""
        suffix = FileUtils.get_extension(path)
        if suffix == "json" or suffix in DERIVATIVE_FORMATS:
            output_files[suffix].add(path)
            return True
        return False
    
    @staticmethod
    async def list_s3_objects_paginated(bucket, prefix):
        """List all objects in an S3 bucket with the given prefix using aiobotocore.
        Returns results page by page for responsive UI updates.
        
        Args:
            bucket: The S3 bucket name
            prefix: The prefix to filter objects by
            
        Yields:
            Lists of S3 object keys, one list per page of results
        """
        session = get_session()
        async with session.create_client('s3') as client:
            paginator = client.get_paginator('list_objects_v2')
            
            # Process each page separately to allow for progress updates
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                page_keys = []
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Skip directories (objects ending with '/')
                        if not obj['Key'].endswith('/'):
                            page_keys.append(obj['Key'])
                
                if page_keys:
                    yield page_keys
                    
    @staticmethod
    async def list_s3_objects_simple_parallel(bucket, prefix, delimiter='/', concurrency=10):
        """List all objects in an S3 bucket using a simplified parallel approach.
        
        This method:
        1. Gets all top-level prefixes
        2. Creates a queue of prefixes to process
        3. Processes prefixes in parallel with a semaphore
        4. Yields each page of results immediately to keep the progress bar responsive
        
        Args:
            bucket: The S3 bucket name
            prefix: The prefix to filter objects by
            delimiter: The delimiter to use (default: '/')
            concurrency: Maximum number of concurrent tasks (default: 10)
            
        Yields:
            Lists of S3 object keys found in batches (page by page)
        """
        # Step 1: Get all top-level prefixes first
        top_level_prefixes = []
        
        # Create a session for the initial listing
        session = get_session()
        async with session.create_client('s3') as client:
            paginator = client.get_paginator('list_objects_v2')
            
            # Process the root prefix first
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter=delimiter):
                # Yield files directly in the root prefix immediately
                if 'Contents' in page:
                    page_keys = []
                    for obj in page['Contents']:
                        if not obj['Key'].endswith('/'):
                            page_keys.append(obj['Key'])
                    if page_keys:
                        yield page_keys
                
                # Collect all top-level prefixes
                if 'CommonPrefixes' in page:
                    for common_prefix in page['CommonPrefixes']:
                        top_level_prefixes.append(common_prefix['Prefix'])
        
        # If we have prefixes to process
        if top_level_prefixes:
            logger.info(f"Processing {len(top_level_prefixes)} top-level prefixes in parallel")
            
            # Step 2 & 3: Create a queue and process with a semaphore
            queue = asyncio.Queue()
            for p in top_level_prefixes:
                await queue.put(p)
            
            # Create a semaphore to limit concurrency
            sem = asyncio.Semaphore(concurrency)
            
            # Create a channel to communicate results back to the main task
            results_queue = asyncio.Queue()
            
            # Process prefixes from the queue
            async def worker():
                while True:
                    try:
                        # Get a prefix from the queue
                        current_prefix = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except asyncio.TimeoutError:
                        # No more prefixes in the queue
                        if queue.empty():
                            break
                        continue
                    
                    try:
                        # Process the prefix with the semaphore
                        async with sem:
                            # Process each page of results
                            session = get_session()
                            async with session.create_client('s3') as client:
                                paginator = client.get_paginator('list_objects_v2')
                                # Process each page and send to results queue immediately
                                async for page in paginator.paginate(Bucket=bucket, Prefix=current_prefix):
                                    page_keys = []
                                    if 'Contents' in page:
                                        for obj in page['Contents']:
                                            if not obj['Key'].endswith('/'):
                                                page_keys.append(obj['Key'])
                                        if page_keys:
                                            # Put the keys in the results queue for yielding
                                            await results_queue.put(page_keys)
                    except Exception as e:
                        logger.warning(f"Error processing prefix {current_prefix}: {e}")
                    finally:
                        queue.task_done()
            
            # Start workers
            workers = []
            for _ in range(concurrency):
                task = asyncio.create_task(worker())
                workers.append(task)
            
            # We don't actually need a consumer since the main loop is consuming
            
            # While work is being done, yield results directly from the results queue
            done_checking = False
            while not done_checking:
                try:
                    # Get results with a timeout
                    result = await asyncio.wait_for(results_queue.get(), timeout=0.1)
                    # Forward to main loop
                    yield result
                    results_queue.task_done()
                except asyncio.TimeoutError:
                    # Check if all work is done
                    if queue.empty() and queue.qsize() == 0 and all(w.done() for w in workers):
                        done_checking = True
            
            # Cancel any remaining workers
            for worker_task in workers:
                worker_task.cancel()
            
            # Wait for tasks to be cancelled
            await asyncio.gather(*workers, return_exceptions=True)
    
    
    
    @staticmethod
    def parse_s3_path(s3_path):
        """Parse an S3 path into bucket and key components.
        
        Args:
            s3_path: An S3 path in the format s3://bucket/key
            
        Returns:
            Tuple of (bucket, key)
        """
        # Remove s3:// prefix and split by first slash
        path_str = str(s3_path)
        if not path_str.startswith('s3://'):
            raise ValueError(f"Not an S3 path: {path_str}")
            
        path = path_str.replace('s3://', '')
        parts = path.split('/', 1)
        
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        return bucket, key

    @staticmethod
    async def riterdir(path: UPath):
        """Recursively iterate through all files in a directory.
        
        This is a replacement for the non-existent riterdir method in UPath.
        Uses optimized path for S3 with simple parallel async aiobotocore calls.
        
        Args:
            path: The UPath directory to recursively iterate through
            
        Yields:
            UPath objects for all files found recursively
        """
        # Special optimized path for S3
        if path.protocol == 's3':
            logger.info(f"Using optimized simple parallel S3 listing for {path}")
            # Parse the S3 path to get bucket and prefix
            bucket, prefix = FileUtils.parse_s3_path(path)
            
            # Use simplified parallel listing for better progress bar responsiveness
            async for page_keys in FileUtils.list_s3_objects_simple_parallel(bucket, prefix, concurrency=10):
                # Create S3Path objects for each key in the current page
                for key in page_keys:
                    yield UPath(f"s3://{bucket}/{key}")
        else:
            # For non-S3 paths, use the standard fsspec implementation
            fs = path.fs
            for file_path in fs.find(str(path), maxdepth=None, withdirs=False):
                yield UPath(file_path)


class FileManager:
    """
    Manages file discovery and tracking for both input and output paths.
    Bulk lists all files in input and output paths at initialization and keeps them in memory.
    """
    
    def __init__(self, input_path: UPath, output_path: UPath):
        """
        Initialize the FileManager with input and output paths.
        
        Args:
            input_path: The input directory path
            output_path: The output directory path
        """
        self.input_path = input_path
        self.output_path = output_path
        
        # Cache for discovered files
        self.input_files: List[UPath] = []
        self.output_files: Dict[str, Set[UPath]] = {}  # Format -> set of files
        
        # Initialization will be done in the async init method
        # We can't scan in __init__ because it needs to be async
        
    async def initialize(self):
        """Asynchronously initialize the file manager by scanning directories."""
        # Initialize by scanning both directories
        await self._scan_input_files()
        await self._scan_output_files()
    
    
    async def _scan_input_files(self):
        """Scan and cache all audio files in the input path."""
        logger.info(f"Scanning input path for audio files: {self.input_path}")
        
        # Use a list to collect files from the async iterator
        all_files = []
        file_iter = FileUtils.riterdir(self.input_path)
        
        # Create a progress bar that updates as we discover files
        progress = tqdm(
            desc="Discovering files",
            unit="files",
            ncols=PROGRESS_BAR_WIDTH,
            leave=True
        )
        
        # Process files from the async iterator
        if self.input_path.protocol == 's3':
            # For S3, we need to handle async iteration
            async for file in file_iter:
                all_files.append(file)
                progress.update(1)
        else:
            # For non-S3, we have a synchronous iterator
            for file in file_iter:
                all_files.append(file)
                progress.update(1)
        
        # Close progress bar
        progress.total = len(all_files)
        progress.refresh()
        progress.close()
        
        # Filter for audio files
        self.input_files = [file for file in all_files if FileUtils.is_audio_file(file)]
        
        # Display summary
        print(f"Found {len(self.input_files)} audio files out of {len(all_files)} total files")
        
        if not self.input_files and self.input_path.is_dir():
            print("No audio files found.")
            logger.warning(
                f"{self.input_path} is a directory with no audio files."
            )
        
        logger.info(f"Found {len(self.input_files)} audio files to process")
    
    async def _scan_output_files(self):
        """Scan and cache all output files by format."""
        # Initialize output files dictionary with empty sets for each format
        self.output_files = {"json": set()}
        for fmt in DERIVATIVE_FORMATS:
            self.output_files[fmt] = set()
        
        logger.info(f"Scanning existing output files in {self.output_path}")
        
        # Skip if output directory doesn't exist yet
        if not self.output_path.exists():
            logger.info("Output directory does not exist yet, no existing files to scan")
            return
            
        try:
            # Use a list to collect files from the async iterator
            all_files = []
            file_iter = FileUtils.riterdir(self.output_path)
            
            # Create a progress bar that updates as we discover files
            progress = tqdm(
                desc="Scanning existing output files",
                unit="files",
                ncols=PROGRESS_BAR_WIDTH,
                leave=True
            )
            
            # Process files from the async iterator
            if self.output_path.protocol == 's3':
                # For S3, we need to handle async iteration
                async for file in file_iter:
                    all_files.append(file)
                    progress.update(1)
            else:
                # For non-S3, we have a synchronous iterator
                for file in file_iter:
                    all_files.append(file)
                    progress.update(1)
            
            # Close progress bar
            progress.total = len(all_files)
            progress.refresh()
            progress.close()
            
            # Process and categorize files after collecting them all
            for path in all_files:
                FileUtils.is_output_file(path, self.output_files)
                
        except Exception as e:
            logger.warning(f"Error scanning output directory {self.output_path}: {e}")
            all_files = []
        
        # Count total files found
        total_files = sum(len(files) for files in self.output_files.values())
        
        # Display summary
        print(f"Found {total_files} output files out of {len(all_files)} total files")
        
        logger.info(f"Found {total_files} existing output files in {self.output_path}")
    
    def get_audio_files(self) -> List[UPath]:
        """Return all discovered audio files."""
        return self.input_files
    
    def get_output_path(self, file_path: UPath, fmt: str) -> UPath:
        """
        Generate the output path for a given file and format, preserving relative path structure.
        
        Args:
            file_path: The input audio file path
            fmt: The format extension (json, txt, srt)
            
        Returns:
            The full output path for the given format
        """
        file_stem = file_path.stem
        
        # Handle relative paths differently based on protocol
        if file_path.protocol == self.input_path.protocol:
            # For same protocol, we can use relative_to
            try:
                rel_path = file_path.relative_to(self.input_path)
                output_path = self.output_path / rel_path.parent / f"{file_stem}.{fmt}"
            except ValueError:
                # If relative_to fails, use a flattened structure
                output_path = self.output_path / f"{file_stem}.{fmt}"
        else:
            # For different protocols, use a flattened structure
            output_path = self.output_path / f"{file_stem}.{fmt}"
        
        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        return output_path
    
    def check_file_needs_processing(
        self, 
        file_path: UPath, 
        derivatives: Optional[List[str]]
    ) -> Tuple[bool, bool, Set[str]]:
        """
        Check if a file needs transcription and/or derivative creation.
        
        Args:
            file_path: Path to the audio file to check
            derivatives: List of derivative formats to generate
            
        Returns:
            Tuple of (needs_transcription, needs_derivatives, missing_derivatives) where:
            - needs_transcription: True if JSON transcription needs to be generated
            - needs_derivatives: True if any derivative format needs to be generated
            - missing_derivatives: Set of derivative formats that need to be generated
        """
        # Get path to JSON transcription
        json_path = self.get_output_path(file_path, "json")
        needs_transcription = json_path not in self.output_files["json"]
        
        # Check which derivatives are missing
        missing_derivatives = set()
        if derivatives:
            for fmt in derivatives:
                output_path = self.get_output_path(file_path, fmt)
                if output_path not in self.output_files[fmt]:
                    missing_derivatives.add(fmt)
        
        needs_derivatives = len(missing_derivatives) > 0
        
        return needs_transcription, needs_derivatives, missing_derivatives
    
    def get_files_needing_processing(
        self, 
        derivatives: Optional[List[str]], 
        force: bool = False
    ) -> List[Tuple[UPath, bool, Set[str]]]:
        """
        Determine which files need processing and what formats are needed.
        
        Args:
            derivatives: List of derivative formats to generate
            force: If True, process all files regardless of existing outputs
            
        Returns:
            List of tuples (file_path, needs_transcription, missing_derivatives)
        """
        if force:
            # In force mode, process everything
            return [(file, True, set(derivatives or [])) for file in self.input_files]
        
        # Check each file to see what processing it needs
        files_to_process = []
        
        # Set up progress bar for checking files
        check_progress = tqdm(
            total=len(self.input_files),
            desc="Checking files to process",
            position=0,
            leave=True
        )
        
        for file_path in self.input_files:
            # Check if file needs processing and which formats are missing
            needs_transcription, needs_derivatives, missing_derivatives = self.check_file_needs_processing(
                file_path, derivatives
            )
            
            if needs_transcription or needs_derivatives:
                files_to_process.append((file_path, needs_transcription, missing_derivatives))
            
            check_progress.update(1)
            check_progress.refresh()  # Manually refresh the progress bar
        
        # Ensure the progress bar shows the correct total number of files checked
        check_progress.total = len(self.input_files)
        check_progress.n = len(self.input_files)
        check_progress.refresh()
        
        # Close progress bar
        check_progress.close()
        
        return files_to_process


class TranscriptionService:
    """Handles transcription of audio files using Whisper models."""
    
    def __init__(self, client=None):
        """Initialize the transcription service with an optional client."""
        self.client = client or openai_client
    
    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(Exception),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
        stop=tenacity.stop_after_attempt(5),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying {retry_state.attempt_number}/5 after exception: "
            f"{retry_state.outcome.exception() if retry_state.outcome else 'Unknown error'}"
        ),
    )
    async def transcribe_file(self, file_path: UPath, model: str) -> Dict:
        """Transcribe a single audio file using the specified Whisper model with retry logic."""
        logger.info(f"Transcribing {file_path} with model {model}")
        
        result = None
        with open(file_path, "rb") as audio_file:
            # Use the temp file directly for transcription
            response = await self.client.audio.transcriptions.create(
                file=audio_file, 
                model=model,
                response_format="verbose_json"
            )
            
            result = response.model_dump() if hasattr(response, "model_dump") else response
        
        return result


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


class FileOutputService:
    """Handles saving transcription data to files in various formats."""
    
    @staticmethod
    async def save_json_transcription(transcription_data: Dict, output_path: UPath) -> None:
        """Save transcription results as JSON."""
        # Format as JSON string
        content = json.dumps(transcription_data, indent=2, ensure_ascii=False)
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file using the appropriate method based on protocol
        if output_path.protocol == "file":
            # Use aiofiles for local files to get async IO benefits
            async with aiofiles.open(str(output_path), 'w', encoding='utf-8') as f:
                await f.write(content)
        else:
            # For cloud paths, use UPath's built-in methods
            output_path.write_text(content)
        
        logger.info(f"Saved JSON transcription to {output_path}")
    
    @staticmethod
    async def save_derivative(
        transcription_data: Dict,
        output_path: UPath,
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
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file using the appropriate method based on protocol
        if output_path.protocol == "file":
            # Use aiofiles for local files to get async IO benefits
            async with aiofiles.open(str(output_path), 'w', encoding='utf-8') as f:
                await f.write(content)
        else:
            # For cloud paths, use UPath's built-in methods
            output_path.write_text(content)
        
        logger.info(f"Saved {format_name.upper()} derivative to {output_path}")


class TranscriptionProcessor:
    """
    Orchestrates the process of transcribing files and generating derivative formats.
    """
    
    def __init__(
        self, 
        file_manager: FileManager,
        transcription_service: TranscriptionService = None,
        output_service: FileOutputService = None
    ):
        """Initialize the processor with needed services."""
        self.file_manager = file_manager
        self.transcription_service = transcription_service or TranscriptionService()
        self.output_service = output_service or FileOutputService()
    
    async def process_file(
        self,
        file_path: UPath,
        model: str = DEFAULT_MODEL,
        derivatives: Optional[List[Literal["txt", "srt"]]] = None
    ) -> None:
        """
        Process a single file: transcribe to JSON and optionally create derivative formats.
        
        Args:
            file_path: Path to the audio file to process
            model: Whisper model to use for transcription
            derivatives: Derivative formats to generate (txt, srt)
        """
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Set empty list if derivatives is None
            derivatives_to_process = derivatives or []
            
            # Get the JSON output path
            json_path = self.file_manager.get_output_path(file_path, "json")
                
            # Check if JSON file exists
            json_exists = json_path.exists()
            
            # Initialize results and tasks
            transcription_data = None
            save_tasks = []
            
            # STEP 1: Try to use existing JSON if possible
            if json_exists:
                logger.info(f"Found existing JSON file: {json_path}")
                # Load and parse the JSON file
                try:
                    # Read the file using the appropriate method based on protocol
                    if json_path.protocol == "file":
                        # Use aiofiles for local files to get async IO benefits
                        async with aiofiles.open(str(json_path), 'rb') as f:
                            json_content = await f.read()
                    else:
                        # For cloud paths, use UPath's built-in methods
                        json_content = json_path.read_bytes()
                    
                    json_text = json_content.decode('utf-8')
                    transcription_data = json.loads(json_text)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing existing JSON file {json_path}: {e}")
                    transcription_data = None  # Force re-transcription
            
            # STEP 2: If no valid JSON exists, perform transcription
            if not json_exists or transcription_data is None:
                logger.info(f"Transcribing: {file_path}")
                transcription_data = await self.transcription_service.transcribe_file(file_path, model)
                
                # Save JSON result
                save_tasks.append(self.output_service.save_json_transcription(
                    transcription_data, 
                    json_path
                ))
            
            # STEP 3: Create derivative formats if requested
            for fmt in derivatives_to_process:
                # Get the output path for this derivative format
                output_path = self.file_manager.get_output_path(file_path, fmt)
                
                # Check if this derivative already exists
                if not output_path.exists():
                    logger.info(f"Creating {fmt} derivative: {output_path}")
                    save_tasks.append(self.output_service.save_derivative(
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
    
    async def process_files(
        self,
        concurrency: int = 5,
        model: str = DEFAULT_MODEL,
        derivatives: Optional[List[Literal["txt", "srt"]]] = None,
        force: bool = False
    ) -> None:
        """
        Process multiple files concurrently, skipping files that already have outputs unless force=True.
        
        Args:
            concurrency: Number of concurrent transcription jobs
            model: Whisper model to use for transcription
            derivatives: Derivative formats to generate
            force: If True, process all files regardless of existing outputs
        """
        # Ensure derivatives is a list or None
        validated_derivatives = derivatives or []
        
        # Validate derivatives against supported list
        for fmt in validated_derivatives.copy():
            if fmt not in DERIVATIVE_FORMATS:
                logger.warning(f"Ignoring unsupported derivative format: {fmt}")
                validated_derivatives.remove(fmt)
        
        # Get files that need processing using the FileManager
        processing_info = self.file_manager.get_files_needing_processing(validated_derivatives, force)
        
        # Extract just the file paths for logging
        files_to_process = [file_info[0] for file_info in processing_info]
        
        # If nothing to process, we're done
        if not files_to_process:
            logger.info("No files need processing - all output files already exist")
            return
            
        logger.info(
            f"Processing {len(files_to_process)} of {len(self.file_manager.get_audio_files())} files "
            f"(skipping {len(self.file_manager.get_audio_files()) - len(files_to_process)} with existing outputs)"
        )
        
        # Step 3: Process files concurrently with semaphore to limit API calls
        semaphore = asyncio.Semaphore(concurrency)
        
        async def _process_with_semaphore(file_path, needs_transcription, missing_derivatives):
            async with semaphore:
                # If we don't need to transcribe or generate derivatives, skip this file
                if not needs_transcription and not missing_derivatives:
                    return
                    
                # Process the file with the required derivatives
                return await self.process_file(
                    file_path, 
                    model, 
                    list(missing_derivatives) if not needs_transcription else validated_derivatives
                )
        
        # Set up progress bar for processing
        progress = tqdm(
            total=len(files_to_process),
            desc="Transcribing files",
            position=0,
            smoothing=0.0,
            ncols=PROGRESS_BAR_WIDTH,
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
                progress.refresh()  # Manually refresh the progress bar
        
        # Create tasks for all files and process them concurrently
        tasks = [process_and_update(info) for info in processing_info]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Make sure progress bar shows the correct final count
        progress.total = len(files_to_process)
        progress.n = len(files_to_process)
        progress.refresh()
        progress.close()


@click.command()
@click.argument("input_path", required=True, type=str)
@click.argument("output_path", required=True, type=str)
@click.option(
    "--concurrency", "-c", default=5,
    help="Number of concurrent transcription requests"
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
    "--model", "-m", default=DEFAULT_MODEL,
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
def main(input_path, output_path, concurrency, verbose, log_file, model, derivative, force):
    """Bulk transcribe audio files using Whisper models.

    INPUT_PATH is the source directory (or s3:// URI).
    OUTPUT_PATH is the destination directory (or s3:// URI) for transcriptions.

    Supports local paths and S3 URIs (s3://bucket/path).

    By default, the tool runs in resumable mode and only processes files that 
    don't already have corresponding output files for all requested formats.
    Use --force to process all files regardless of existing outputs.

    The tool generates JSON transcriptions and optionally creates derivative 
    formats (txt or srt) from these transcriptions.

    All input and output directories are scanned recursively.

    Example usage:

        whisperbulk ./audio_files ./transcriptions -c 10
        whisperbulk s3://mybucket/audio s3://mybucket/transcriptions -m whisper-1 -d srt
        
    You can specify multiple derivative formats:
    
        whisperbulk ./audio_files ./transcriptions -d txt -d srt
        
    To reprocess all files regardless of existing outputs:
    
        whisperbulk ./audio_files ./transcriptions --force
    """
    # Convert input and output paths to UPath objects
    input_path_obj = UPath(input_path)
    output_path_obj = UPath(output_path) if output_path else None
    
    # Validate that input path is a directory
    if not input_path_obj.is_dir():
        print(f"Error: Input path '{input_path}' must be a directory.")
        logger.error(f"Input path '{input_path}' is not a directory")
        sys.exit(1)
    
    # Configure logging
    if log_file:
        # If user provided custom log file path
        log_path = UPath(log_file)
        if log_path.parent != UPath('.'):
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Reconfigure the logging to use the custom file
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=str(log_path),
            filemode='a'
        )
    elif verbose:
        # Just change the log level if using default log file
        logger.setLevel(logging.DEBUG)

    # Ensure output directory exists if it's local
    if output_path_obj and output_path_obj.protocol == "file":
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Use asyncio to run the entire pipeline
    async def run_pipeline():
        # Initialize the FileManager to bulk list all input and output files
        file_manager = FileManager(input_path_obj, output_path_obj)
        
        # Asynchronously initialize the file manager
        await file_manager.initialize()
        
        # Check if we found any audio files
        if not file_manager.get_audio_files():
            logger.error("No audio files found to process")
            sys.exit(1)
            
        logger.info(
            f"Found {len(file_manager.get_audio_files())} files to process "
            f"with concurrency {concurrency}"
        )
        
        # Process files - convert tuple to list if necessary
        derivatives_list = list(derivative) if isinstance(derivative, tuple) else derivative
        
        # Log resumable mode (unless force is enabled)
        if not force:
            logger.info("Running in resumable mode - will skip files that already have outputs")
        
        # Create processor and process the files
        processor = TranscriptionProcessor(file_manager)    
        await processor.process_files(
            concurrency, 
            model, 
            derivatives_list, 
            force
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
