# WhisperX Local Interface

A local web interface for transcribing audio files using WhisperX or Lightning Whisper MLX, providing fast speech recognition with word-level timestamps and speaker diarization.

## Features

- Drop audio files to process with WhisperX or Lightning Whisper MLX
- View transcription with word-level timestamps
- Speaker diarization support (WhisperX backend only)
- Export transcriptions as text files
- Fast processing with GPU support (if available)
- Apple Silicon optimization with two backend options:
  - WhisperX (CPU mode with optimized compute types)
  - Lightning Whisper MLX (Optimized specifically for Apple Silicon)

## Requirements

- Python 3.10 or higher
- CUDA-compatible GPU (optional, for faster processing with WhisperX)
- Apple Silicon M-series chip (optional, for MLX backend)
- Hugging Face account for speaker diarization (obtain a token at [huggingface.co](https://huggingface.co/settings/tokens))

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/whisper-for-conversations.git
cd whisper-for-conversations
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Hugging Face token:

```
HF_TOKEN=your_hugging_face_token_here
```

## Model Weights Management

This repository **does not** include model weights, which will be downloaded automatically on first use.

- **WhisperX models**: Downloaded to `~/.cache/whisperx` directory
- **Lightning Whisper MLX models**: Downloaded to `~/.cache/huggingface` directory
- **Speaker diarization models**: Downloaded to the Hugging Face cache directory

If you want to pre-download models before running the application:

```bash
# For WhisperX models (replace MODEL_SIZE with tiny, base, small, medium, or large)
python -c "import whisperx; whisperx.load_model('MODEL_SIZE')"

# For Lightning Whisper MLX models
python -c "from lightning_whisper.models import WhisperCkpt; WhisperCkpt.from_pretrained('lightningwhisper/whisper-large-v3-mlx', device='auto')"
```

Note: Model weights can be several gigabytes in size, especially for larger models.

## Usage

1. Start the application:

```bash
python src/app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload an audio file by dropping it into the designated area

4. If using Apple Silicon, choose your backend:

   - WhisperX: For speaker diarization support
   - Lightning Whisper MLX: For much faster transcription (up to 10x faster)

5. Choose your compute type (see below for details)

6. Wait for the transcription to complete

7. View the transcription with timestamps and speaker labels

8. Export the transcription in TXT, SRT, or JSON format

## Backend Options

### WhisperX Backend

- **Benefits**: Speaker diarization, word-level timestamps
- **Limitations**: On Apple Silicon, runs in CPU mode only

### Lightning Whisper MLX Backend (Apple Silicon Only)

- **Benefits**:
  - Significantly faster transcription (up to 10x faster than WhisperX)
  - Optimized specifically for Apple Silicon
  - Uses Apple's MLX framework for efficient processing
  - Supports distilled models for even faster processing
  - Support for quantized models to optimize memory usage
- **Limitations**:
  - No speaker diarization support
  - Apple Silicon only (M1/M2/M3/M4)

## Compute Type Options

The application supports different compute types, which affect performance and compatibility:

### For WhisperX Backend

#### `float16`

- **Best for**: Apple Silicon (M1/M2/M3/M4) with sufficient RAM
- **Advantages**: Good balance of speed and accuracy
- **Disadvantages**: May not be supported on older non-Apple CPUs
- **Recommendation**: Default option for Apple Silicon (M1/M2/M3/M4)

#### `float32`

- **Best for**: Maximum compatibility across all devices
- **Advantages**: Works on virtually any modern CPU
- **Disadvantages**: Uses more memory and may be slower than float16/int8
- **Recommendation**: Use this if you need maximum precision

#### `int8`

- **Best for**: Low-memory environments or older hardware
- **Advantages**: Lowest memory usage, fastest on CPU
- **Disadvantages**: Slightly lower accuracy than float16/float32
- **Recommendation**: Use this if you're experiencing memory issues or running on older hardware

### For MLX Backend

#### `float16` (No Quantization)

- **Best for**: Higher accuracy with sufficient memory
- **Advantages**: Better accuracy, still fast on Apple Silicon
- **Recommendation**: Default option for most use cases

#### `int8` (8-bit Quantization)

- **Best for**: Low-memory environments
- **Advantages**: Faster processing, lower memory usage
- **Disadvantages**: Slightly lower accuracy
- **Recommendation**: Use for very large models or when memory is constrained

## Apple Silicon Optimization

This application offers two different approaches for optimizing performance on Apple Silicon:

1. **WhisperX Backend**: Uses CPU mode with optimized compute types, but doesn't leverage GPU acceleration.
2. **Lightning Whisper MLX Backend**: Fully optimized for Apple Silicon using Apple's MLX framework.
   - Much faster processing (10x faster than CPU-based alternatives)
   - Support for distilled models which are faster with minimal accuracy loss
   - Batch processing for higher throughput
   - Quantization options for memory efficiency

For the best experience on Apple Silicon:

- Use **Lightning Whisper MLX** with distilled models when speed is the priority
- Use **WhisperX** when speaker identification is required

## Batch Size (MLX Backend Only)

The MLX backend allows you to configure batch size for processing:

- Higher values process more audio in parallel (faster) but use more memory
- Recommended settings:
  - Large models: 8-10
  - Medium models: 12 (default)
  - Small/tiny models: 16+

## Configuration

Core settings can be adjusted through the web interface. Advanced configuration available in:

- `src/services/whisper_service.py` for WhisperX settings
- `src/services/mlx_whisper_service.py` for MLX Whisper settings

## License

MIT
