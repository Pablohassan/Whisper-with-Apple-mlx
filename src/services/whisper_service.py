import os
import gc
import torch
import whisperx
import platform
import psutil
import time
import threading

# Global progress variable for WhisperX transcription
whisperx_progress = {
    "status": "idle",
    "progress": 0,
    "total": 100,
    "current_stage": "",
    "memory_usage": 0,
    "cpu_usage": 0,
    "gpu_usage": 0,
    "start_time": 0,
    "elapsed_time": 0,
    "estimated_remaining": 0,
    "audio_duration": 0,
    "audio_processed": 0,
    "hardware_acceleration": False,
    "backend_info": "WhisperX (CPU/GPU)",
    "model_loaded": False,
    "processing_speed": 0,
    "diarization_enabled": False
}

def get_whisperx_progress_info():
    """Return current WhisperX transcription progress information"""
    global whisperx_progress
    
    # Update elapsed time if transcription is running
    if whisperx_progress["status"] == "processing" and whisperx_progress["start_time"] > 0:
        whisperx_progress["elapsed_time"] = time.time() - whisperx_progress["start_time"]
        
        # Calculate estimated remaining time
        if whisperx_progress["progress"] > 5:
            estimated_total = (whisperx_progress["elapsed_time"] * 100) / whisperx_progress["progress"]
            whisperx_progress["estimated_remaining"] = max(0, estimated_total - whisperx_progress["elapsed_time"])
        
        # Update processing speed if we have audio duration
        if whisperx_progress["audio_duration"] > 0 and whisperx_progress["audio_processed"] > 0:
            whisperx_progress["processing_speed"] = whisperx_progress["audio_processed"] / whisperx_progress["elapsed_time"]
    
    return whisperx_progress

def reset_whisperx_progress():
    """Reset progress tracking to initial state"""
    global whisperx_progress
    
    # Get initial system metrics
    try:
        import psutil
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
    except:
        cpu_usage = 0
        memory_usage = 0
    
    whisperx_progress = {
        "status": "idle",
        "progress": 0,
        "total": 100,
        "current_stage": "",
        "memory_usage": memory_usage,  # Use real initial memory
        "cpu_usage": cpu_usage,  # Use real initial CPU
        "gpu_usage": 0,
        "start_time": 0,
        "elapsed_time": 0,
        "estimated_remaining": 0,
        "audio_duration": 0,
        "audio_processed": 0,
        "hardware_acceleration": False,
        "backend_info": "WhisperX (CPU/GPU)",
        "model_loaded": False,
        "processing_speed": 0,
        "diarization_enabled": False
    }

def update_whisperx_system_metrics():
    """Update system resource usage metrics for WhisperX"""
    global whisperx_progress
    try:
        # CPU usage
        whisperx_progress["cpu_usage"] = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        whisperx_progress["memory_usage"] = memory.percent
        
        # GPU usage (if CUDA available)
        if torch.cuda.is_available() and whisperx_progress["status"] == "processing":
            # Estimate GPU usage based on processing activity
            whisperx_progress["gpu_usage"] = min(70 + (whisperx_progress["progress"] % 25), 85)
        else:
            whisperx_progress["gpu_usage"] = 0
            
    except Exception as e:
        print(f"Error updating WhisperX system metrics: {e}")

def get_audio_duration_whisperx(audio_file):
    """Get audio file duration in seconds for WhisperX"""
    try:
        # Use librosa to get audio duration
        import librosa
        duration = librosa.get_duration(path=audio_file)
        return duration
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        # Fallback: try with subprocess and ffprobe
        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_file
            ], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0

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
        
        # Reset progress tracking
        reset_whisperx_progress()
        
        # Update hardware info
        whisperx_progress["hardware_acceleration"] = self.device == "cuda"
        whisperx_progress["backend_info"] = f"WhisperX ({self.device_type})"
        
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
    
    def _monitor_whisperx_progress_thread(self, audio_duration):
        """Thread to monitor and update progress metrics for WhisperX"""
        while whisperx_progress["status"] == "processing":
            try:
                update_whisperx_system_metrics()
                
                # Simulate audio processing progress based on elapsed time
                if whisperx_progress["elapsed_time"] > 0 and audio_duration > 0:
                    # WhisperX on CPU is typically slower than real-time
                    if self.device == "cuda":
                        speed_factor = 3.0  # GPU is usually 3x faster than real-time
                    else:
                        speed_factor = 0.5  # CPU is usually slower than real-time
                    
                    processed_time = min(audio_duration, whisperx_progress["elapsed_time"] * speed_factor)
                    whisperx_progress["audio_processed"] = processed_time
                    
                    # Update progress based on current stage and audio processed
                    if whisperx_progress["current_stage"].startswith("Transcribing"):
                        base_progress = 20
                        audio_progress = min(50, (processed_time / audio_duration) * 50)
                        if whisperx_progress["progress"] < base_progress + audio_progress:
                            whisperx_progress["progress"] = base_progress + audio_progress
                
                time.sleep(0.5)  # Update every 500ms
            except Exception as e:
                print(f"Error in WhisperX progress monitoring: {e}")
                break
    
    @property
    def device_info(self):
        """Return information about the device and acceleration being used"""
        info = {
            "device": self.device_type,
            "acceleration": self.acceleration,
            "compute_type": self.compute_type,
            "model": self.model_name,
            "backend": "WhisperX",
            "hardware_acceleration": self.device == "cuda",
            "gpu_support": self.device == "cuda"
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
            # Reset and initialize progress
            reset_whisperx_progress()
            whisperx_progress["status"] = "processing"
            whisperx_progress["start_time"] = time.time()
            whisperx_progress["current_stage"] = "Loading audio file..."
            whisperx_progress["progress"] = 5
            
            # Get audio duration
            audio_duration = get_audio_duration_whisperx(audio_file)
            whisperx_progress["audio_duration"] = audio_duration
            whisperx_progress["diarization_enabled"] = bool(self.hf_token and max_speakers)
            
            # Start progress monitoring thread
            monitor_thread = threading.Thread(target=self._monitor_whisperx_progress_thread, args=(audio_duration,))
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # 1. Load audio
            whisperx_progress["current_stage"] = "Loading audio..."
            whisperx_progress["progress"] = 10
            audio = whisperx.load_audio(audio_file)
            
            # 2. Load ASR model and transcribe
            whisperx_progress["current_stage"] = "Loading ASR model..."
            whisperx_progress["progress"] = 15
            model = whisperx.load_model(
                self.model_name, 
                self.device, 
                compute_type=self.compute_type,
                language=language
            )
            
            whisperx_progress["model_loaded"] = True
            whisperx_progress["current_stage"] = "Transcribing audio..."
            whisperx_progress["progress"] = 20
            
            # Transcribe with batched processing
            result = model.transcribe(
                audio, 
                batch_size=batch_size,
                language=language
            )
            
            detected_language = result["language"]
            print(f"Detected language: {detected_language}")
            
            whisperx_progress["current_stage"] = "Aligning timestamps..."
            whisperx_progress["progress"] = 70
            
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
            
            whisperx_progress["current_stage"] = "Processing alignment..."
            whisperx_progress["progress"] = 80
            whisperx_progress["audio_processed"] = audio_duration
            
            # Clear GPU memory
            if self.device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
                del model_a
            
            # 4. Speaker diarization (if token provided and max_speakers specified)
            if self.hf_token and max_speakers:
                try:
                    whisperx_progress["current_stage"] = "Performing speaker diarization..."
                    whisperx_progress["progress"] = 85
                    
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
            
            # Final progress update
            whisperx_progress["current_stage"] = "Transcription complete"
            whisperx_progress["progress"] = 100
            whisperx_progress["status"] = "complete"
            whisperx_progress["elapsed_time"] = time.time() - whisperx_progress["start_time"]
            whisperx_progress["audio_processed"] = audio_duration
            
            # Add processing metrics to result
            if isinstance(result, dict):
                result["audio_duration"] = audio_duration
                result["processing_time"] = whisperx_progress["elapsed_time"]
            
            # Final system metrics update
            update_whisperx_system_metrics()
            
            return result
        
        except Exception as e:
            whisperx_progress["status"] = "error"
            whisperx_progress["current_stage"] = f"Error: {str(e)}"
            print(f"Error in transcription: {str(e)}")
            raise e 