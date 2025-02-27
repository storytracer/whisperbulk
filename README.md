# WhisperBulk

A CLI tool for bulk transcribing audio files using OpenAI's Whisper API.

## Features

- Transcribe multiple audio files concurrently
- Supports both local files and S3 storage
- Configurable concurrency level
- Automatic retries with exponential backoff
- Progress bar to track transcription status

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/whisperbulk.git
cd whisperbulk

# Install dependencies
pip install -e .
```

Or install directly:

```bash
pip install git+https://github.com/yourusername/whisperbulk.git
```

## Requirements

Required Python packages:
- click
- openai
- tenacity
- smart_open
- python-dotenv
- tqdm

## Usage

### Environment Setup

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
```

### Basic Usage

Transcribe all audio files in a directory:

```bash
whisperbulk ./audio_files ./transcriptions -r
```

Transcribe a single file:

```bash
whisperbulk file1.mp3 ./transcriptions
```

### S3 Support

Transcribe files from S3:

```bash
whisperbulk s3://mybucket/audio ./transcriptions -r
```

Save transcriptions to S3:

```bash
whisperbulk ./audio_files s3://mybucket/transcriptions -r
```

### Options

- `-c, --concurrency` - Number of concurrent transcription requests (default: 5)
- `-r, --recursive` - Recursively process directories
- `-v, --verbose` - Enable verbose logging
- `-f, --format` - Output format for transcriptions (default: txt, options: txt, json, srt). This option can be used multiple times to generate multiple output formats for each file.

### Examples with Multiple Formats

Generate both text and SRT format transcriptions:

```bash
whisperbulk ./audio_files ./transcriptions -f txt -f srt
```

Generate all supported formats:

```bash
whisperbulk ./audio_files ./transcriptions -f txt -f json -f srt
```

## License

MIT
