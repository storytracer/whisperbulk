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
import aiofiles.tempfile
import srt
import datetime
import re
import os
import contextlib
from typing import List, Optional, Union, Any, Dict, Literal, Set, Tuple
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import click
import dotenv
from openai import AsyncOpenAI
import tenacity
from tqdm import tqdm
from aiobotocore.session import get_session

# Constants
AUDIO_EXTENSIONS = ("mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm")
DERIVATIVE_FORMATS = ("txt", "srt")
PROGRESS_BAR_WIDTH = 75
DEFAULT_MODEL = "Systran/faster-whisper-medium"

# Load environment variables from .env file
dotenv.load_dotenv()

# Setup logging
log_dir = Path(__file__).parent / "logs"
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

# Create a single aiobotocore session for the whole application
boto_session = get_session()


class FilePathType:
    """Represents a path that can be either local or S3."""
    
    def __init__(self, path_str: str):
        """Initialize from a string path that can be either local or s3://."""
        self.original_path = path_str
        parsed = urlparse(path_str)
        self.scheme = parsed.scheme

        if self.scheme == 's3':
            self.bucket = parsed.netloc
            # Remove leading slash for S3 keys
            self.path = parsed.path.lstrip('/')
            self.is_s3 = True
            self.local_path = None
        else:
            self.bucket = None
            self.path = path_str
            self.is_s3 = False
            self.local_path = Path(path_str)
    
    def __str__(self) -> str:
        """Return the string representation of the path."""
        return self.original_path
    
    def __truediv__(self, other) -> 'FilePathType':
        """Support path / component operations like Path objects."""
        if self.is_s3:
            if self.path and not self.path.endswith('/'):
                new_path = f"s3://{self.bucket}/{self.path}/{other}"
            else:
                new_path = f"s3://{self.bucket}/{self.path}{other}"
        else:
            new_path = str(self.local_path / other)
        
        return FilePathType(new_path)
    
    @property
    def parent(self) -> 'FilePathType':
        """Get the parent directory."""
        if self.is_s3:
            # Split the path and remove the last component
            components = self.path.rstrip('/').split('/')
            if len(components) <= 1:
                # Root of bucket
                return FilePathType(f"s3://{self.bucket}/")
            
            parent_path = '/'.join(components[:-1])
            return FilePathType(f"s3://{self.bucket}/{parent_path}/")
        else:
            return FilePathType(str(self.local_path.parent))
    
    @property
    def name(self) -> str:
        """Get the filename."""
        if self.is_s3:
            components = self.path.rstrip('/').split('/')
            return components[-1] if components else ""
        else:
            return self.local_path.name
    
    @property
    def stem(self) -> str:
        """Get the filename without extension."""
        name = self.name
        if '.' in name:
            return name.rsplit('.', 1)[0]
        return name
    
    @property
    def suffix(self) -> str:
        """Get the file extension with leading period."""
        if self.is_s3:
            if '.' in self.name:
                return '.' + self.name.rsplit('.', 1)[1]
            return ''
        else:
            return self.local_path.suffix
    
    async def exists(self) -> bool:
        """Check if the path exists."""
        if self.is_s3:
            try:
                async with boto_session.create_client('s3') as client:
                    if self.path.endswith('/'):
                        # For directories in S3, check if there are objects with this prefix
                        response = await client.list_objects_v2(
                            Bucket=self.bucket,
                            Prefix=self.path,
                            MaxKeys=1
                        )
                        return 'Contents' in response
                    else:
                        # For files, check if the object exists
                        try:
                            await client.head_object(Bucket=self.bucket, Key=self.path)
                            return True
                        except:
                            return False
            except Exception as e:
                logger.error(f"Error checking if S3 path exists: {e}")
                return False
        else:
            return self.local_path.exists()
    
    async def is_dir(self) -> bool:
        """Check if the path is a directory."""
        if self.is_s3:
            # S3 doesn't have directories per se, but we can check if path ends with /
            # or if there are objects with this prefix
            if self.path.endswith('/'):
                return True
                
            try:
                async with boto_session.create_client('s3') as client:
                    # Check if there are objects with this prefix + /
                    dir_check = f"{self.path}/"
                    response = await client.list_objects_v2(
                        Bucket=self.bucket,
                        Prefix=dir_check,
                        MaxKeys=1
                    )
                    return 'Contents' in response
            except Exception as e:
                logger.error(f"Error checking if S3 path is directory: {e}")
                return False
        else:
            return self.local_path.is_dir()
    
    async def mkdir(self, parents=False, exist_ok=False) -> None:
        """Create a directory. For S3, this is a no-op as S3 doesn't have directories."""
        if not self.is_s3:
            self.local_path.mkdir(parents=parents, exist_ok=exist_ok)
    
    async def read_bytes(self) -> bytes:
        """Read file contents as bytes."""
        if self.is_s3:
            async with boto_session.create_client('s3') as client:
                response = await client.get_object(Bucket=self.bucket, Key=self.path)
                async with response['Body'] as stream:
                    return await stream.read()
        else:
            async with aiofiles.open(str(self.local_path), 'rb') as f:
                return await f.read()
    
    async def read_text(self, encoding='utf-8') -> str:
        """Read file contents as text."""
        content = await self.read_bytes()
        return content.decode(encoding)
    
    async def write_bytes(self, data: bytes) -> None:
        """Write bytes to file."""
        if self.is_s3:
            async with boto_session.create_client('s3') as client:
                await client.put_object(Bucket=self.bucket, Key=self.path, Body=data)
        else:
            async with aiofiles.open(str(self.local_path), 'wb') as f:
                await f.write(data)
    
    async def write_text(self, data: str, encoding='utf-8') -> None:
        """Write text to file."""
        await self.write_bytes(data.encode(encoding))
    
    def relative_to(self, other: 'FilePathType') -> 'FilePathType':
        """Get path relative to another path."""
        if self.is_s3 and other.is_s3:
            if self.bucket != other.bucket:
                raise ValueError("Cannot get relative path between different S3 buckets")
            
            if not self.path.startswith(other.path):
                raise ValueError(f"Path {self.path} does not start with {other.path}")
            
            rel_path = self.path[len(other.path):].lstrip('/')
            return FilePathType(rel_path)
        elif not self.is_s3 and not other.is_s3:
            rel_path = self.local_path.relative_to(other.local_path)
            return FilePathType(str(rel_path))
        else:
            raise ValueError("Cannot get relative path between local and S3 paths")


class FileUtils:
    """Static utility methods for file operations."""
    
    @staticmethod
    def get_extension(path: FilePathType) -> str:
        """Get the file extension without the leading period."""
        return path.suffix.lower().lstrip('.')
    
    @staticmethod
    def is_audio_file(path: FilePathType) -> bool:
        """Check if a file is a supported audio format."""
        suffix = FileUtils.get_extension(path)
        return suffix in AUDIO_EXTENSIONS
    
    @staticmethod
    def is_output_file(path: FilePathType, output_files: Dict[str, Set[FilePathType]]) -> bool:
        """Check if a file is a supported output format and add it to the output files dictionary."""
        suffix = FileUtils.get_extension(path)
        if suffix == "json" or suffix in DERIVATIVE_FORMATS:
            output_files[suffix].add(path)
            return True
        return False
    
    @staticmethod
    async def list_s3_objects_paginated(bucket, prefix=""):
        """List all objects in an S3 bucket with the given prefix using aiobotocore.
        Returns results page by page for responsive UI updates.
        
        Args:
            bucket: The S3 bucket name
            prefix: The prefix to filter objects by
            
        Yields:
            Lists of S3 object keys, one list per page of results
        """
        async with boto_session.create_client('s3') as client:
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
    async def riterdir(path: FilePathType):
        """Recursively iterate through all files in a directory.
        
        Uses optimized path for S3 with simple parallel async aiobotocore calls.
        
        Args:
            path: The directory to recursively iterate through
            
        Yields:
            FilePathType objects for all files found recursively
        """
        if path.is_s3:
            # For S3, use aiobotocore to list objects
            bucket = path.bucket
            prefix = path.path
            if not prefix.endswith('/'):
                prefix += '/'
                
            async for page_keys in FileUtils.list_s3_objects_paginated(bucket, prefix):
                for key in page_keys:
                    yield FilePathType(f"s3://{bucket}/{key}")
        else:
            # For local paths, use os.walk but make it async-compatible
            for root, _, files in os.walk(str(path.local_path)):
                for file in files:
                    file_path = os.path.join(root, file)
                    yield FilePathType(file_path)
                # Give control back to the event loop occasionally
                await asyncio.sleep(0)


@asynccontextmanager
async def open_file(file_path: FilePathType, mode='rb'):
    """Context manager for opening files, handling both local and S3 paths."""
    if file_path.is_s3:
        if 'r' in mode:  # Reading
            # Create a temporary file using aiofiles.tempfile
            async with aiofiles.tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Download the S3 file to the temporary file
                async with boto_session.create_client('s3') as client:
                    response = await client.get_object(Bucket=file_path.bucket, Key=file_path.path)
                    async with response['Body'] as stream:
                        content = await stream.read()
                
                # Write the content to the temporary file
                async with aiofiles.open(temp_path, 'wb') as f:
                    await f.write(content)
                
                # Open the temporary file in the requested mode
                file = await aiofiles.open(temp_path, mode)
                try:
                    yield file
                finally:
                    await file.close()
                    # Clean up the temporary file
                    os.unlink(temp_path)
            except Exception as e:
                # Clean up in case of error
                os.unlink(temp_path)
                raise e
        else:  # Writing
            # Create a temporary file using aiofiles.tempfile
            async with aiofiles.tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Open the temporary file for writing
                file = await aiofiles.open(temp_path, mode)
                try:
                    yield file
                finally:
                    await file.close()
                
                # Read the file content
                async with aiofiles.open(temp_path, 'rb') as f:
                    file_content = await f.read()
                
                # Upload to S3
                async with boto_session.create_client('s3') as client:
                    await client.put_object(Bucket=file_path.bucket, Key=file_path.path, Body=file_content)
                
                # Clean up the temporary file
                os.unlink(temp_path)
            except Exception as e:
                # Clean up in case of error
                os.unlink(temp_path)
                raise e
    else:
        # For local files, use aiofiles directly
        file = await aiofiles.open(str(file_path.local_path), mode)
        try:
            yield file
        finally:
            await file.close()


class FileManager:
    """
    Manages file discovery and tracking for both input and output paths.
    Bulk lists all files in input and output paths at initialization and keeps them in memory.
    """
    
    def __init__(self, input_path: FilePathType, output_path: FilePathType):
        """
        Initialize the FileManager with input and output paths.
        
        Args:
            input_path: The input directory path
            output_path: The output directory path
        """
        self.input_path = input_path
        self.output_path = output_path
        
        # Cache for discovered files
        self.input_files: List[FilePathType] = []
        self.output_files: Dict[str, Set[FilePathType]] = {}  # Format -> set of files
        
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
        
        # Create a progress bar that updates as we discover files
        progress = tqdm(
            desc="Discovering files",
            unit="files",
            ncols=PROGRESS_BAR_WIDTH,
            leave=True
        )
        
        # Process files from the async iterator
        async for file in FileUtils.riterdir(self.input_path):
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
        
        if not self.input_files:
            input_is_dir = await self.input_path.is_dir()
            if input_is_dir:
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
        output_exists = await self.output_path.exists()
        if not output_exists:
            logger.info("Output directory does not exist yet, no existing files to scan")
            return
            
        # Use a list to collect files from the async iterator
        all_files = []
        
        # Create a progress bar that updates as we discover files
        progress = tqdm(
            desc="Scanning output files",
            unit="files",
            ncols=PROGRESS_BAR_WIDTH,
            leave=True
        )
        
        # Process files from the async iterator
        async for file in FileUtils.riterdir(self.output_path):
            all_files.append(file)
            progress.update(1)
        
        # Close progress bar
        progress.total = len(all_files)
        progress.refresh()
        progress.close()
        
        # Process and categorize files after collecting them all
        for path in all_files:
            FileUtils.is_output_file(path, self.output_files)
        
        # Count total files found
        total_files = sum(len(files) for files in self.output_files.values())
        
        # Display summary
        print(f"Found {total_files} output files out of {len(all_files)} total files")
        
        logger.info(f"Found {total_files} existing output files in {self.output_path}")
    
    def get_audio_files(self) -> List[FilePathType]:
        """Return all discovered audio files."""
        return self.input_files
    
    def get_output_files_by_format(self, format_name: str) -> List[FilePathType]:
        """Return all output files of a specific format."""
        return list(self.output_files.get(format_name, set()))
    
    def get_file_stem_mapping(self) -> Dict[str, Dict[str, Set[str]]]:
        """
        Create a mapping of file stems to formats and paths for efficient lookup.
        
        Returns:
            A dictionary mapping file stems to a dictionary of formats and their paths
        """
        mapping = {}
        
        # Group output files by stem and format
        for fmt, files in self.output_files.items():
            for file in files:
                stem = file.stem
                
                if stem not in mapping:
                    mapping[stem] = {}
                    
                if fmt not in mapping[stem]:
                    mapping[stem][fmt] = set()
                    
                mapping[stem][fmt].add(str(file))
                
        return mapping
    
    def get_output_path(self, file_path: FilePathType, fmt: str) -> FilePathType:
        """
        Generate the output path for a given file and format, preserving relative path structure.
        
        Args:
            file_path: The input audio file path
            fmt: The format extension (json, txt, srt)
            
        Returns:
            The full output path for the given format
        """
        # Get the file stem (filename without extension)
        file_stem = file_path.stem
        
        # Get relative path from input directory to the file
        rel_path = file_path.relative_to(self.input_path)
        # Remove the filename part, keeping only the directory structure
        rel_dir = rel_path.parent
            
        # Construct the output path preserving the directory structure
        output_path = self.output_path / rel_dir / f"{file_stem}.{fmt}"
        
        # Ensure the directory exists
        asyncio.create_task(output_path.parent.mkdir(parents=True, exist_ok=True))
        
        return output_path
    
    def check_transcription_needed(self, file_path: FilePathType, file_stem_mapping: Dict) -> bool:
        """
        Check if a file needs transcription.
        
        Args:
            file_path: Path to the audio file to check
            file_stem_mapping: Mapping of file stems to formats and paths
            
        Returns:
            True if transcription is needed, False otherwise
        """
        stem = file_path.stem
        return stem not in file_stem_mapping or "json" not in file_stem_mapping[stem]
    
    def get_missing_derivatives(
        self, 
        file_path: FilePathType, 
        derivatives: List[str],
        file_stem_mapping: Dict
    ) -> Set[str]:
        """
        Get the set of derivative formats that need to be generated for a file.
        
        Args:
            file_path: Path to the audio file to check
            derivatives: List of derivative formats to check
            file_stem_mapping: Mapping of file stems to formats and paths
            
        Returns:
            Set of derivative formats that need to be generated
        """
        if not derivatives:
            return set()
            
        stem = file_path.stem
        
        # If the file stem isn't in the mapping or has no derivatives, all are missing
        if stem not in file_stem_mapping:
            return set(derivatives)
            
        # Check which derivatives are missing
        return set(fmt for fmt in derivatives if fmt not in file_stem_mapping[stem])
    
    def get_files_needing_processing(
        self, 
        derivatives: Optional[List[str]], 
        force: bool = False
    ) -> List[Tuple[FilePathType, bool, Set[str]]]:
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
        
        # Create file stem mapping for efficient lookup
        file_stem_mapping = self.get_file_stem_mapping()
        
        # Check each file to see what processing it needs
        files_to_process = []
        
        # Set up progress bar for checking files
        check_progress = tqdm(
            total=len(self.input_files),
            desc="Checking files to process",
            position=0,
            ncols=PROGRESS_BAR_WIDTH,
            leave=True
        )
        
        for file_path in self.input_files:
            # Check if file needs transcription
            needs_transcription = self.check_transcription_needed(file_path, file_stem_mapping)
            
            # Check which derivatives are missing
            missing_derivatives = self.get_missing_derivatives(
                file_path, derivatives or [], file_stem_mapping
            )
            
            # If either transcription or derivatives are needed, add to list
            if needs_transcription or missing_derivatives:
                files_to_process.append((file_path, needs_transcription, missing_derivatives))
            
            check_progress.update(1)
        
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
        stop=tenacity.stop_after_attempt(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying {retry_state.attempt_number}/3 after exception: "
            f"{retry_state.outcome.exception() if retry_state.outcome else 'Unknown error'}"
        ),
    )
    async def transcribe_file(self, file_path: FilePathType, model: str) -> Dict:
        """Transcribe a single audio file using the specified Whisper model with retry logic."""
        logger.info(f"Transcribing {file_path} with model {model}")
        
        result = None
        
        # Use aiofiles.tempfile for temporary file handling
        async with aiofiles.tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Get file content - either from S3 or local file
            if file_path.is_s3:
                # Download S3 file
                async with boto_session.create_client('s3') as client:
                    response = await client.get_object(Bucket=file_path.bucket, Key=file_path.path)
                    async with response['Body'] as stream:
                        content = await stream.read()
                
                # Write to temporary file
                async with aiofiles.open(temp_path, 'wb') as f:
                    await f.write(content)
            else:
                # Copy local file to temporary file
                async with aiofiles.open(str(file_path.local_path), 'rb') as src:
                    content = await src.read()
                async with aiofiles.open(temp_path, 'wb') as dst:
                    await dst.write(content)
            
            # Use the temporary file for transcription
            # OpenAI API still requires a file object, which is synchronous
            with open(temp_path, 'rb') as audio_file:
                response = await self.client.audio.transcriptions.create(
                    file=audio_file, 
                    model=model,
                    response_format="verbose_json"
                )
                
                result = response.model_dump() if hasattr(response, "model_dump") else response
            
            # Clean up temporary file
            os.unlink(temp_path)
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
        
        return result


class TranscriptionConverter:
    """
    Handles conversion of JSON transcription data to various derivative formats.
    """
    
    @staticmethod
    def convert(
        transcription_data: Dict,
        format_name: Literal["txt", "srt"]
    ) -> str:
        """Convert JSON transcription data to the specified format."""
        if format_name == "txt":
            return TranscriptionConverter.to_text(transcription_data)
        elif format_name == "srt":
            return TranscriptionConverter.to_srt(transcription_data)
        else:
            raise ValueError(f"Unsupported derivative format: {format_name}")
    
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
    
    def __init__(self, converter: TranscriptionConverter = None):
        """Initialize with an optional converter."""
        self.converter = converter or TranscriptionConverter()
    
    @staticmethod
    async def save_json_transcription(transcription_data: Dict, output_path: FilePathType) -> None:
        """Save transcription results as JSON."""
        # Format as JSON string
        content = json.dumps(transcription_data, indent=2, ensure_ascii=False)
        
        # Ensure directory exists
        await output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the content
        await output_path.write_text(content)
        
        logger.info(f"Saved JSON transcription to {output_path}")
    
    async def save_derivative(
        self,
        transcription_data: Dict,
        output_path: FilePathType,
        format_name: Literal["txt", "srt"]
    ) -> None:
        """Save transcription results in a derivative format (txt or srt)."""
        # Convert to the requested format using the converter
        content = self.converter.convert(transcription_data, format_name)
        
        # Ensure directory exists
        await output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        await output_path.write_text(content)
        
        logger.info(f"Saved {format_name.upper()} derivative to {output_path}")


class TranscriptionProcessor:
    """
    Orchestrates the process of transcribing files and generating derivative formats.
    """
    
    def __init__(
        self, 
        file_manager: FileManager,
        transcription_service: TranscriptionService = None,
        output_service: FileOutputService = None,
        converter: TranscriptionConverter = None
    ):
        """Initialize the processor with needed services."""
        self.file_manager = file_manager
        self.transcription_service = transcription_service or TranscriptionService()
        self.converter = converter or TranscriptionConverter()
        self.output_service = output_service or FileOutputService(self.converter)
    
    async def process_file(
        self,
        file_path: FilePathType,
        model: str = DEFAULT_MODEL,
        derivatives: Optional[List[Literal["txt", "srt"]]] = None
    ) -> None:
        """
        Process a single file: transcribe to JSON and generate derivative formats.
        Always generates derivatives, overwriting existing ones if they exist.
        
        Args:
            file_path: Path to the audio file to process
            model: Whisper model to use for transcription
            derivatives: Derivative formats to generate (txt, srt)
        """
        logger.info(f"Processing file: {file_path}")
        
        # Set empty list if derivatives is None
        derivatives_to_process = derivatives or []
        
        # Get the JSON output path
        json_path = self.file_manager.get_output_path(file_path, "json")
            
        # Check if JSON file exists
        json_exists = await json_path.exists()
        
        # Initialize results and tasks
        transcription_data = None
        save_tasks = []
        
        # STEP 1: Try to use existing JSON if possible
        if json_exists:
            logger.info(f"Found existing JSON file: {json_path}")
            # Read the JSON file
            json_content = await json_path.read_bytes()
            json_text = json_content.decode('utf-8')
            transcription_data = json.loads(json_text)
        
        # STEP 2: If no valid JSON exists, perform transcription
        if not json_exists or transcription_data is None:
            logger.info(f"Transcribing: {file_path}")
            transcription_data = await self.transcription_service.transcribe_file(file_path, model)
            
            # Save JSON result
            save_tasks.append(self.output_service.save_json_transcription(
                transcription_data, 
                json_path
            ))
        
        # STEP 3: Create derivative formats if requested - ALWAYS generate them
        for fmt in derivatives_to_process:
            # Get the output path for this derivative format
            output_path = self.file_manager.get_output_path(file_path, fmt)
            
            # Always generate derivatives, overwriting existing ones
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
            await _process_with_semaphore(file_path, needs_transcription, missing_derivatives)
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
    # Convert input and output paths to FilePathType objects
    input_path_obj = FilePathType(input_path)
    output_path_obj = FilePathType(output_path) if output_path else None
    
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
            filename=str(log_path),
            filemode='a'
        )
    elif verbose:
        # Just change the log level if using default log file
        logger.setLevel(logging.DEBUG)

    # Use asyncio to run the entire pipeline
    async def run_pipeline():
        # Validate that input path is a directory
        input_is_dir = await input_path_obj.is_dir()
        if not input_is_dir:
            print(f"Error: Input path '{input_path}' must be a directory.")
            logger.error(f"Input path '{input_path}' is not a directory")
            sys.exit(1)
            
        # Ensure output directory exists if it's local
        if output_path_obj and not output_path_obj.is_s3:
            await output_path_obj.mkdir(parents=True, exist_ok=True)
            
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