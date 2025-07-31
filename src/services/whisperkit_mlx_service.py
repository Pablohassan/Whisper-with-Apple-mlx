import os
import gc
import subprocess
import json
import tempfile
import platform
import time
import threading
import psutil
import librosa

# Global progress variable for WhisperKit MLX transcription
whisperkit_progress = {
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
    "hardware_acceleration": True,
    "backend_info": "WhisperKit (CoreML + Metal)",
    "model_loaded": False,
    "processing_speed": 0  # Real-time factor (1.0 = real-time speed)
}

def get_whisperkit_progress_info():
    """Return current WhisperKit transcription progress information"""
    global whisperkit_progress
    
    # Update elapsed time if transcription is running
    if whisperkit_progress["status"] == "processing" and whisperkit_progress["start_time"] > 0:
        whisperkit_progress["elapsed_time"] = time.time() - whisperkit_progress["start_time"]
        
        # Calculate estimated remaining time
        if whisperkit_progress["progress"] > 5:
            estimated_total = (whisperkit_progress["elapsed_time"] * 100) / whisperkit_progress["progress"]
            whisperkit_progress["estimated_remaining"] = max(0, estimated_total - whisperkit_progress["elapsed_time"])
        
        # Update processing speed if we have audio duration
        if whisperkit_progress["audio_duration"] > 0 and whisperkit_progress["audio_processed"] > 0:
            whisperkit_progress["processing_speed"] = whisperkit_progress["audio_processed"] / whisperkit_progress["elapsed_time"]
    
    return whisperkit_progress

def reset_whisperkit_progress():
    """Reset progress tracking to initial state"""
    global whisperkit_progress
    
    # Get initial system metrics
    try:
        import psutil
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
    except:
        cpu_usage = 0
        memory_usage = 0
    
    whisperkit_progress = {
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
        "hardware_acceleration": True,
        "backend_info": "WhisperKit (CoreML + Metal)",
        "model_loaded": False,
        "processing_speed": 0
    }

def update_system_metrics():
    """Update system resource usage metrics"""
    global whisperkit_progress
    try:
        # CPU usage
        whisperkit_progress["cpu_usage"] = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        whisperkit_progress["memory_usage"] = memory.percent
        
        # For Apple Silicon, we can't easily get GPU usage, so we estimate based on activity
        if whisperkit_progress["status"] == "processing":
            # Estimate GPU usage based on processing activity (simulated)
            whisperkit_progress["gpu_usage"] = min(80 + (whisperkit_progress["progress"] % 20), 95)
        else:
            whisperkit_progress["gpu_usage"] = 0
            
    except Exception as e:
        print(f"Error updating system metrics: {e}")

def get_audio_duration(audio_file):
    """Get audio file duration in seconds"""
    try:
        # Use librosa to get audio duration
        duration = librosa.get_duration(path=audio_file)
        return duration
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        # Fallback: try with subprocess and ffprobe
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', audio_file
            ], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except:
            pass
        return 0

class WhisperKitMLXService:
    """Service for transcribing audio files using WhisperKit by Argmax (Apple Silicon optimized)"""
    
    def __init__(self, hf_token=None, model_name="large-v3", compute_type="float16", batch_size=12, default_language=None):
        """
        Initialize the WhisperKitMLXService
        
        Args:
            hf_token (str): Hugging Face token (not used by WhisperKit)
            model_name (str): Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            compute_type (str): Compute type (for compatibility, not used by WhisperKit)
            batch_size (int): Batch size (for compatibility, not used by WhisperKit)
            default_language (str): Default language to use if none specified (e.g., 'fr' for French)
        """
        self.hf_token = hf_token  # Not used but kept for compatibility
        self.model_name = model_name
        self.compute_type = compute_type  # Not used by WhisperKit but kept for compatibility
        self.batch_size = batch_size  # Not used by WhisperKit but kept for compatibility
        self.default_language = default_language  # New: default language option
        
        # Check if running on Apple Silicon
        self.is_apple_silicon = (platform.processor() == 'arm' and platform.system() == 'Darwin')
        
        if not self.is_apple_silicon:
            raise RuntimeError("WhisperKit by Argmax requires Apple Silicon (M1/M2/M3/M4)")
        
        # Check if whisperkit-cli is installed
        self._check_whisperkit_cli()
        
        # Store device info
        self.device_type = "Apple Silicon"
        self.acceleration = "GPU Accelerated (WhisperKit/CoreML)"
        
        # Reset progress tracking
        reset_whisperkit_progress()
        
        # Update progress with hardware info
        whisperkit_progress["hardware_acceleration"] = True
        whisperkit_progress["backend_info"] = f"WhisperKit (CoreML + Metal) - {model_name}"
        
        print(f"Using WhisperKit by Argmax (Apple Silicon Optimized)")
        print(f"Model: {self.model_name}")
        print(f"Framework: CoreML with Metal Performance Shaders")
    
    def _check_whisperkit_cli(self):
        """Check if whisperkit-cli is installed and accessible"""
        try:
            result = subprocess.run(['whisperkit-cli', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError("whisperkit-cli not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(
                "whisperkit-cli not found. Please install it with: brew install whisperkit-cli"
            )
    
    @property
    def device_info(self):
        """Return information about the device and acceleration being used"""
        return {
            "device": self.device_type,
            "acceleration": self.acceleration,
            "model": self.model_name,
            "backend": "WhisperKit (Argmax)",
            "framework": "CoreML + Metal",
            "optimization": "Apple Silicon Native",
            "hardware_acceleration": True,
            "gpu_support": True
        }
    
    def _monitor_progress_thread(self, audio_duration):
        """Thread to monitor and update progress metrics"""
        while whisperkit_progress["status"] == "processing":
            try:
                update_system_metrics()
                
                # Simulate audio processing progress based on elapsed time and estimated duration
                if whisperkit_progress["elapsed_time"] > 0 and audio_duration > 0:
                    # Estimate processed audio time (this is a rough estimation)
                    estimated_speed = audio_duration / max(60, audio_duration * 0.1)  # Rough speed estimate
                    processed_time = min(audio_duration, whisperkit_progress["elapsed_time"] * estimated_speed)
                    whisperkit_progress["audio_processed"] = processed_time
                    
                    # Update progress based on audio processed
                    audio_progress = min(80, (processed_time / audio_duration) * 80)
                    if whisperkit_progress["progress"] < audio_progress:
                        whisperkit_progress["progress"] = audio_progress
                
                time.sleep(0.5)  # Update every 500ms
            except Exception as e:
                print(f"Error in progress monitoring: {e}")
                break
    
    def _transcribe_thread(self, audio_file, language, result_container):
        """Internal thread function to handle transcription with progress updates"""
        try:
            # Update progress state
            global whisperkit_progress
            whisperkit_progress["status"] = "processing"
            whisperkit_progress["start_time"] = time.time()
            whisperkit_progress["current_stage"] = "Initializing WhisperKit"
            whisperkit_progress["progress"] = 5
            
            # Get audio duration
            audio_duration = get_audio_duration(audio_file)
            whisperkit_progress["audio_duration"] = audio_duration
            
            # Start progress monitoring thread
            monitor_thread = threading.Thread(target=self._monitor_progress_thread, args=(audio_duration,))
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Prepare whisperkit-cli command
            cmd = ['whisperkit-cli', 'transcribe']
            
            # Add model if specified (WhisperKit will auto-download if needed)
            if self.model_name:
                cmd.extend(['--model', self.model_name])
            
            # Add audio file
            cmd.extend(['--audio-path', audio_file])
            
            # Add language if specified or use default
            effective_language = language or self.default_language
            if effective_language:
                cmd.extend(['--language', effective_language])
                print(f"DEBUG: Language specified: {effective_language}")
            else:
                print("DEBUG: No language specified, using auto-detection")
            
            # Add word timestamps for better segment information
            cmd.append('--word-timestamps')
            
            # Use verbose mode for better error reporting
            cmd.append('--verbose')
            
            # Set task to transcribe (default, but explicit)
            cmd.extend(['--task', 'transcribe'])
            
            whisperkit_progress["current_stage"] = "Loading model..."
            whisperkit_progress["progress"] = 15
            
            print(f"DEBUG: Running command: {' '.join(cmd)}")
            
            # Update stage
            whisperkit_progress["current_stage"] = "Transcribing audio..."
            whisperkit_progress["model_loaded"] = True
            whisperkit_progress["progress"] = 25
            
            # Run the transcription
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            whisperkit_progress["current_stage"] = "Processing results..."
            whisperkit_progress["progress"] = 90
            whisperkit_progress["audio_processed"] = audio_duration
            
            if result.returncode != 0:
                error_msg = f"WhisperKit CLI failed: {result.stderr}"
                print(f"ERROR: {error_msg}")
                print(f"STDOUT: {result.stdout}")
                raise Exception(error_msg)
            
            # WhisperKit CLI outputs plain text, not JSON
            # Parse the output to create segments
            transcription_text = result.stdout.strip()
            
            # Create a basic segment structure
            if transcription_text:
                normalized_result = {
                    "text": transcription_text,
                    "segments": [{
                        "id": 0,
                        "start": 0.0,
                        "end": audio_duration if audio_duration > 0 else 10.0,
                        "text": transcription_text
                    }],
                    "warning": "Speaker diarization not supported with WhisperKit backend. Limited segment timing info from CLI.",
                    "audio_duration": audio_duration,
                    "processing_time": whisperkit_progress["elapsed_time"]
                }
            else:
                normalized_result = {
                    "text": "No transcription result",
                    "segments": [],
                    "error": "Empty transcription result"
                }
            
            # Final progress update
            whisperkit_progress["progress"] = 100
            whisperkit_progress["current_stage"] = "Transcription complete"
            whisperkit_progress["status"] = "complete"
            whisperkit_progress["elapsed_time"] = time.time() - whisperkit_progress["start_time"]
            whisperkit_progress["audio_processed"] = audio_duration
            
            # Final system metrics update
            update_system_metrics()
            
            # Store result
            result_container["result"] = normalized_result
            result_container["success"] = True
            
        except Exception as e:
            whisperkit_progress["status"] = "error"
            whisperkit_progress["current_stage"] = f"Error: {str(e)}"
            
            print(f"Error in WhisperKitMLXService.transcribe: {str(e)}")
            import traceback
            traceback.print_exc()
            result_container["error"] = str(e)
            result_container["success"] = False
    
    def _normalize_whisperkit_result(self, result):
        """Normalize WhisperKit result to match expected format"""
        try:
            # WhisperKit may return different formats, normalize them
            if isinstance(result, dict):
                # If it already has the expected structure
                if "text" in result and "segments" in result:
                    # Add warning about no diarization
                    if "warning" not in result:
                        result["warning"] = "Speaker diarization not supported with WhisperKit backend"
                    return result
                
                # If it's a simple structure, extract text
                elif "text" in result:
                    segments = []
                    if "segments" in result and isinstance(result["segments"], list):
                        segments = result["segments"]
                    else:
                        # Create a simple segment
                        segments = [{
                            "id": 0,
                            "start": 0,
                            "end": 10,
                            "text": result["text"]
                        }]
                    
                    return {
                        "text": result["text"],
                        "segments": segments,
                        "warning": "Speaker diarization not supported with WhisperKit backend"
                    }
            
            # Fallback for unknown formats
            text_content = str(result) if result else "Transcription failed"
            return {
                "text": text_content,
                "segments": [{
                    "id": 0,
                    "start": 0,
                    "end": 10,
                    "text": text_content
                }],
                "warning": "Speaker diarization not supported with WhisperKit backend"
            }
            
        except Exception as e:
            print(f"Error normalizing WhisperKit result: {str(e)}")
            return {
                "text": "Error processing transcription result",
                "segments": [{
                    "id": 0,
                    "start": 0,
                    "end": 10,
                    "text": "Error processing transcription result"
                }],
                "error": str(e)
            }
    
    def transcribe(self, audio_file, language=None, batch_size=None, max_speakers=None):
        """
        Transcribe an audio file using WhisperKit by Argmax
        
        Args:
            audio_file (str): Path to the audio file
            language (str, optional): Language code (e.g., 'en', 'fr', 'de')
            batch_size (int, optional): Not used by WhisperKit (kept for compatibility)
            max_speakers (int, optional): Not supported by WhisperKit
            
        Returns:
            dict: Transcription result with segments
        """
        # Reset progress tracking
        reset_whisperkit_progress()
        
        # Validate input file
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Create a container to hold the result from the thread
        result_container = {"result": None, "success": False, "error": None}
        
        # Start transcription in a separate thread to allow progress reporting
        transcription_thread = threading.Thread(
            target=self._transcribe_thread,
            args=(audio_file, language, result_container)
        )
        transcription_thread.start()
        transcription_thread.join()  # Wait for completion
        
        # Check if transcription was successful
        if not result_container["success"]:
            raise Exception(result_container["error"])
        
        return result_container["result"] 