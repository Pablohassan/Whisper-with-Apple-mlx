import os
import gc
import torch
import whisperx
import platform

class WhisperService:
    """Service for transcribing audio files using WhisperX"""
    
    def __init__(self, hf_token=None, model_name="large-v2", compute_type="float16"):
        """
        Initialize the WhisperService
        
        Args:
            hf_token (str): Hugging Face token for speaker diarization
            model_name (str): Whisper model size (tiny, base, small, medium, large-v1, large-v2, large-v3)
            compute_type (str): Compute type (float16, int8, float32) - use int8 for CPU or lower GPU memory
        """
        self.hf_token = hf_token
        self.model_name = model_name
        
        # Detect if running on Apple Silicon
        self.is_apple_silicon = (platform.processor() == 'arm' and platform.system() == 'Darwin')
        
        # Set device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.device_type = "NVIDIA GPU"
            self.acceleration = "GPU Accelerated"
        else:
            # For Apple Silicon and other CPU devices, we use CPU
            # faster-whisper doesn't support MPS yet
            self.device = "cpu"
            if self.is_apple_silicon:
                self.device_type = "Apple Silicon"
                self.acceleration = "CPU Only (faster-whisper doesn't support GPU)"
            else:
                self.device_type = "CPU"
                self.acceleration = "CPU Only"
        
        # Determine compute_type based on device and hardware
        if self.device == "cpu":
            if self.is_apple_silicon:
                # For Apple Silicon, allow the specified compute_type
                self.compute_type = compute_type
                print(f"Running on Apple Silicon (CPU mode) with compute_type: {self.compute_type}")
            elif compute_type == "float16":
                # For non-Apple CPUs, float16 isn't efficient, use int8
                self.compute_type = "int8"
                print(f"Switching to int8 compute type for CPU (float16 is not efficient on regular CPUs)")
            else:
                self.compute_type = compute_type
        else:
            self.compute_type = compute_type
        
        # Print device info
        print(f"Using device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        elif self.is_apple_silicon:
            print(f"Running on Apple Silicon (CPU mode) with compute_type: {self.compute_type}")
            print(f"Note: MPS is detected but not used as faster-whisper doesn't support it yet")
        else:
            print(f"Running on CPU with compute_type: {self.compute_type}")
    
    @property
    def device_info(self):
        """Return information about the device and acceleration being used"""
        info = {
            "device": self.device_type,
            "acceleration": self.acceleration,
            "compute_type": self.compute_type,
            "model": self.model_name,
            "backend": "WhisperX"
        }
        
        if self.device == "cuda":
            info["gpu_model"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        
        return info
    
    def transcribe(self, audio_file, language=None, batch_size=16, max_speakers=None):
        """
        Transcribe an audio file using WhisperX
        
        Args:
            audio_file (str): Path to the audio file
            language (str, optional): Language code (e.g., 'en', 'fr', 'de')
            batch_size (int): Batch size for processing
            max_speakers (int, optional): Maximum number of speakers for diarization
            
        Returns:
            dict: Transcription result with segments and word-level timestamps
        """
        try:
            # 1. Load audio
            audio = whisperx.load_audio(audio_file)
            
            # 2. Load ASR model and transcribe
            model = whisperx.load_model(
                self.model_name, 
                self.device, 
                compute_type=self.compute_type,
                language=language
            )
            
            # Transcribe with batched processing
            result = model.transcribe(
                audio, 
                batch_size=batch_size,
                language=language
            )
            
            detected_language = result["language"]
            print(f"Detected language: {detected_language}")
            
            # Clear GPU memory
            if self.device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
                del model
            
            # 3. Load alignment model and align
            model_a, metadata = whisperx.load_align_model(
                language_code=detected_language, 
                device=self.device
            )
            
            # Align whisper output
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            
            # Clear GPU memory
            if self.device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
                del model_a
            
            # 4. Speaker diarization (if token provided and max_speakers specified)
            if self.hf_token and max_speakers:
                try:
                    diarize_model = whisperx.DiarizationPipeline(
                        use_auth_token=self.hf_token,
                        device=self.device
                    )
                    
                    # Get speaker labels
                    diarize_segments = diarize_model(
                        audio,
                        min_speakers=1,
                        max_speakers=max_speakers
                    )
                    
                    # Assign word-level speakers
                    result = whisperx.assign_word_speakers(
                        diarize_segments,
                        result
                    )
                    
                    # Clear GPU memory
                    if self.device == "cuda":
                        gc.collect()
                        torch.cuda.empty_cache()
                        del diarize_model
                        
                except Exception as e:
                    print(f"Warning: Speaker diarization failed - {str(e)}")
                    # Continue without diarization
            
            return result
        
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            raise e 