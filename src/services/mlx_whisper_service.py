import os
import gc
import torch
import psutil
import platform
from lightning_whisper_mlx import LightningWhisperMLX
import math
import time
import threading
import json

# Global progress variable for MLX transcription
progress_info = {
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
    "backend_info": "Lightning Whisper MLX",
    "model_loaded": False,
    "processing_speed": 0,
    "batch_processing": False
}

def get_progress_info():
    """Return current transcription progress information"""
    global progress_info
    
    # Update elapsed time if transcription is running
    if progress_info["status"] == "processing" and progress_info["start_time"] > 0:
        progress_info["elapsed_time"] = time.time() - progress_info["start_time"]
        
        # Calculate estimated remaining time
        if progress_info["progress"] > 5:
            estimated_total = (progress_info["elapsed_time"] * 100) / progress_info["progress"]
            progress_info["estimated_remaining"] = max(0, estimated_total - progress_info["elapsed_time"])
        
        # Update processing speed if we have audio duration
        if progress_info["audio_duration"] > 0 and progress_info["audio_processed"] > 0:
            progress_info["processing_speed"] = progress_info["audio_processed"] / progress_info["elapsed_time"]
    
    return progress_info

def reset_progress():
    """Reset progress tracking to initial state"""
    global progress_info
    
    # Get initial system metrics
    try:
        import psutil
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
    except:
        cpu_usage = 0
        memory_usage = 0
    
    progress_info = {
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
        "backend_info": "Lightning Whisper MLX",
        "model_loaded": False,
        "processing_speed": 0,
        "batch_processing": False
    }

def get_gpu_memory_usage():
    """Get approximate GPU memory usage for Apple Silicon (via RAM since Metal/MLX uses unified memory)"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    except Exception as e:
        print(f"Error getting memory usage: {str(e)}")
        return 0

def update_system_metrics():
    """Update system resource usage metrics"""
    global progress_info
    try:
        # CPU usage
        progress_info["cpu_usage"] = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        progress_info["memory_usage"] = memory.percent
        
        # For MLX, we can't easily get GPU usage, but estimate based on activity
        if progress_info["status"] == "processing":
            # Estimate GPU usage based on processing activity
            base_usage = 70 if progress_info["batch_processing"] else 50
            progress_info["gpu_usage"] = min(base_usage + (progress_info["progress"] % 30), 90)
        else:
            progress_info["gpu_usage"] = 0
            
    except Exception as e:
        print(f"Error updating system metrics: {e}")

def get_audio_duration(audio_file):
    """Get audio file duration in seconds"""
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

class MLXWhisperService:
    """Service for transcribing audio files using Lightning Whisper MLX (Apple Silicon optimized)"""
    
    def __init__(self, hf_token=None, model_name="medium", compute_type="float16", batch_size=12):
        """
        Initialize the MLXWhisperService
        
        Args:
            hf_token (str): Hugging Face token for speaker diarization
            model_name (str): Whisper model size (tiny, small, base, medium, distil-medium.en, large, large-v2, distil-large-v2, large-v3, distil-large-v3)
            compute_type (str): Compute type (None, "4bit", "8bit") - quantization level
            batch_size (int): Batch size for processing (higher is faster but uses more memory)
        """
        self.hf_token = hf_token
        
        # Map model names from WhisperX to lightning-whisper-mlx format
        # This ensures compatibility with existing UI/settings
        model_mapping = {
            "tiny": "tiny",
            "base": "base", 
            "small": "small",
            "medium": "medium",
            "large-v1": "large",
            "large-v2": "large-v2",
            "large-v3": "large-v3"
        }
        
        # Use mapped model name or default to the original if not in mapping
        self.model_name = model_mapping.get(model_name, model_name)
        
        # Map compute_type to quantization options
        # float16 -> None (no quantization)
        # int8 -> "8bit"
        # float32 -> None (no quantization)
        quant_mapping = {
            "float16": None,
            "int8": "8bit",
            "float32": None
        }
        self.quant = quant_mapping.get(compute_type, None)
        
        # Set batch size (default to 12 as recommended by the library)
        self.batch_size = batch_size
        
        # Store device info
        self.device_type = "Apple Silicon"
        self.acceleration = "GPU Accelerated (MLX)"
        
        # Reset progress tracking
        reset_progress()
        
        # Update progress - model loading
        global progress_info
        progress_info["status"] = "loading_model"
        progress_info["current_stage"] = f"Loading MLX model: {self.model_name}"
        
        # Print device info
        print(f"Using Lightning Whisper MLX (Apple Silicon Optimized)")
        print(f"Model: {self.model_name}")
        print(f"Quantization: {self.quant}")
        print(f"Batch size: {self.batch_size}")
        
        # Initialize the model
        self.whisper = LightningWhisperMLX(
            model=self.model_name,
            batch_size=self.batch_size,
            quant=self.quant
        )
        
        # Update initial memory usage
        progress_info["memory_usage"] = get_gpu_memory_usage()
        progress_info["status"] = "ready"
    
    @property
    def device_info(self):
        """Return information about the device and acceleration being used"""
        # Get current memory usage
        memory_usage = get_gpu_memory_usage()
        
        return {
            "device": self.device_type,
            "acceleration": self.acceleration,
            "model": self.model_name,
            "quantization": self.quant if self.quant else "None",
            "batch_size": self.batch_size,
            "backend": "MLX",
            "memory_usage": f"{memory_usage:.1f} MB"
        }
    
    def _monitor_progress_thread(self, audio_duration):
        """Thread to monitor and update progress metrics"""
        while progress_info["status"] == "processing":
            try:
                update_system_metrics()
                
                # Update batch processing indicator
                progress_info["batch_processing"] = self.batch_size > 1
                
                # Simulate audio processing progress based on elapsed time and estimated duration
                if progress_info["elapsed_time"] > 0 and audio_duration > 0:
                    # MLX is typically faster than real-time, estimate based on model size
                    speed_factor = {
                        "tiny": 8.0, "base": 6.0, "small": 4.0,
                        "medium": 3.0, "large": 2.0, "large-v2": 1.8, "large-v3": 1.5
                    }.get(self.model_name, 3.0)
                    
                    # Adjust for quantization (8bit is faster)
                    if self.quant == "8bit":
                        speed_factor *= 1.5
                    
                    processed_time = min(audio_duration, progress_info["elapsed_time"] * speed_factor)
                    progress_info["audio_processed"] = processed_time
                    
                    # Update progress based on audio processed
                    audio_progress = min(75, (processed_time / audio_duration) * 75)
                    if progress_info["progress"] < audio_progress + 10:
                        progress_info["progress"] = audio_progress + 10
                
                time.sleep(0.3)  # Update every 300ms for MLX (faster updates)
            except Exception as e:
                print(f"Error in MLX progress monitoring: {e}")
                break

    def _transcribe_thread(self, audio_file, language, batch_size, result_container):
        """Internal thread function to handle transcription with progress updates"""
        try:
            # Update progress state
            global progress_info
            progress_info["status"] = "processing"
            progress_info["start_time"] = time.time()
            progress_info["current_stage"] = "Loading audio file"
            progress_info["progress"] = 5
            
            # Get audio duration
            audio_duration = get_audio_duration(audio_file)
            progress_info["audio_duration"] = audio_duration
            progress_info["hardware_acceleration"] = True
            progress_info["backend_info"] = f"Lightning Whisper MLX - {self.model_name} ({self.quant or 'float16'})"
            
            # Start progress monitoring thread
            monitor_thread = threading.Thread(target=self._monitor_progress_thread, args=(audio_duration,))
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Use provided batch_size if specified, otherwise use default
            actual_batch_size = batch_size if batch_size is not None else self.batch_size
            
            time.sleep(0.5)  # Simulate audio loading time
            progress_info["current_stage"] = "Loading model..."
            progress_info["progress"] = 10
            progress_info["model_loaded"] = True
            
            progress_info["current_stage"] = "Transcribing audio..."
            progress_info["progress"] = 15
            
            print("DEBUG: Starting transcription")
            # Transcribe audio
            result = self.whisper.transcribe(
                audio_path=audio_file,
                language=language
                # batch_size is not supported in the transcribe method
                # batch_size can only be set during initialization
            )
            print("DEBUG: Transcription completed")
            
            # Update memory usage
            progress_info["memory_usage"] = get_gpu_memory_usage()
            progress_info["progress"] = 80
            progress_info["current_stage"] = "Post-processing transcription..."
            progress_info["audio_processed"] = audio_duration
            
            # Debug output
            print(f"MLX result type: {type(result)}")
            if isinstance(result, dict):
                print(f"Result keys: {list(result.keys())}")
                
                # Special handling for dictionary results with 'segments' field
                if 'segments' in result:
                    print(f"Segments type: {type(result['segments'])}")
                    if result['segments'] and len(result['segments']) > 0:
                        print(f"First segment type: {type(result['segments'][0])}")
                        print(f"First segment: {result['segments'][0]}")
            
            print("DEBUG: Beginning format normalization")
            # Check the response format and normalize
            # Depends on the Lightning Whisper MLX version and might change
            # Handle different potential formats
            normalized_result = {}
            
            try:
                # Pre-process the result to fix common issues
                processed_result = self._preprocess_mlx_result(result)
                
                # Now proceed with normal normalization
                print("DEBUG: Using pre-processed result")
                normalized_result = processed_result
                
                # Add processing metrics
                if isinstance(normalized_result, dict):
                    normalized_result["audio_duration"] = audio_duration
                    normalized_result["processing_time"] = progress_info["elapsed_time"]
                
                # Ensure normalized_result has a proper structure
                print(f"Normalized result type: {type(normalized_result)}")
                print(f"Normalized result keys: {list(normalized_result.keys()) if isinstance(normalized_result, dict) else 'Not a dict'}")
                
            except Exception as format_error:
                print(f"Error during format normalization: {str(format_error)}")
                import traceback
                traceback.print_exc()
                # Create a safe fallback structure
                normalized_result = {
                    "text": str(result) if result else "Transcription failed",
                    "segments": [{"id": 0, "start": 0, "end": 10, "text": "Error processing result", "words": []}]
                }
            
            # Note: Lightning Whisper MLX doesn't have speaker diarization built-in
            # We could add a message to the output that diarization isn't supported
            if isinstance(normalized_result, dict) and "warning" not in normalized_result:
                normalized_result["warning"] = "Speaker diarization not supported with Lightning Whisper MLX backend"
            
            # Final progress update
            progress_info["progress"] = 100
            progress_info["current_stage"] = "Transcription complete"
            progress_info["status"] = "complete"
            progress_info["elapsed_time"] = time.time() - progress_info["start_time"]
            progress_info["audio_processed"] = audio_duration
            
            # Final system metrics update
            update_system_metrics()
            
            # Store result
            result_container["result"] = normalized_result
            result_container["success"] = True
            
        except Exception as e:
            progress_info["status"] = "error"
            progress_info["current_stage"] = f"Error: {str(e)}"
            
            print(f"Error in MLXWhisperService.transcribe: {str(e)}")
            import traceback
            traceback.print_exc()
            result_container["error"] = str(e)
            result_container["success"] = False
            
    def _preprocess_mlx_result(self, result):
        """Pre-process MLX result to handle various formats and fix common issues"""
        print("DEBUG: In _preprocess_mlx_result")
        
        # If result is None or empty, return a simple structure
        if result is None:
            return {"text": "", "segments": []}
            
        # If result is a string
        if isinstance(result, str):
            return {"text": result, "segments": [{"id": 0, "start": 0, "end": 10, "text": result, "words": []}]}
            
        # If result is a dictionary
        if isinstance(result, dict):
            print("DEBUG: Processing dict result")
            # Create a copy to avoid modifying the original
            processed = result.copy()
            
            # Ensure 'text' exists
            if 'text' not in processed:
                processed['text'] = ""
                
            # Process segments if they exist
            if 'segments' in processed:
                print(f"DEBUG: Processing segments, type: {type(processed['segments'])}")
                
                # Ensure segments is a list
                if not isinstance(processed['segments'], list):
                    processed['segments'] = []
                else:
                    # Check if segments are in list format [start, end, text]
                    segments = processed['segments']
                    if segments and isinstance(segments[0], list):
                        print("DEBUG: Converting list-format segments to dictionaries")
                        formatted_segments = []
                        
                        for i, seg in enumerate(segments):
                            try:
                                # Verify we have a proper segment
                                if len(seg) >= 3:
                                    start_time = float(seg[0])
                                    end_time = float(seg[1])
                                    text = str(seg[2])
                                    
                                    # Convert times from milliseconds to seconds if needed
                                    start_sec = start_time / 1000 if start_time > 100 else start_time
                                    end_sec = end_time / 1000 if end_time > 100 else end_time
                                    
                                    # Create segment dict
                                    formatted_segment = {
                                        'id': i,
                                        'start': start_sec,
                                        'end': end_sec,
                                        'text': text,
                                        'words': []
                                    }
                                    
                                    # Add word-level timestamps
                                    words = text.split()
                                    if words:
                                        segment_duration = end_sec - start_sec
                                        word_duration = segment_duration / len(words)
                                        
                                        word_start = start_sec
                                        for word in words:
                                            word_end = word_start + word_duration
                                            formatted_segment['words'].append({
                                                'start': word_start,
                                                'end': word_end,
                                                'word': word
                                            })
                                            word_start = word_end
                                    
                                    formatted_segments.append(formatted_segment)
                            except Exception as e:
                                print(f"DEBUG: Error processing segment {i}: {str(e)}")
                        
                        # Replace segments with our formatted version
                        processed['segments'] = formatted_segments
                    else:
                        # Process existing dictionary segments
                        print("DEBUG: Processing dictionary-based segments")
                        for i, segment in enumerate(segments):
                            # Ensure each segment is a dictionary
                            if not isinstance(segment, dict):
                                print(f"DEBUG: Segment {i} is not a dict: {type(segment)}")
                                segments[i] = {"id": i, "start": 0, "end": 1, "text": str(segment), "words": []}
                            else:
                                # Ensure required fields exist
                                if 'id' not in segment:
                                    segment['id'] = i
                                if 'start' not in segment:
                                    segment['start'] = i * 2.0
                                if 'end' not in segment:
                                    segment['end'] = (i + 1) * 2.0
                                if 'text' not in segment:
                                    segment['text'] = ""
                                if 'words' not in segment:
                                    segment['words'] = []
            else:
                # If no segments exist, create a basic segment from the text
                if 'text' in processed and processed['text']:
                    processed['segments'] = [{
                        'id': 0,
                        'start': 0,
                        'end': len(processed['text'].split()) * 0.3,  # Rough estimate
                        'text': processed['text'],
                        'words': []
                    }]
                else:
                    processed['segments'] = []
                    
            return processed
            
        # If result is a list (rare, but possible)
        if isinstance(result, list):
            print("DEBUG: Result is a list")
            segments = []
            full_text = ""
            
            for i, item in enumerate(result):
                if isinstance(item, dict) and 'text' in item:
                    segment_text = item['text']
                elif isinstance(item, list) and len(item) >= 3:
                    segment_text = item[2]
                else:
                    segment_text = str(item)
                    
                full_text += segment_text + " "
                
                start = i * 2.0
                end = start + len(segment_text.split()) * 0.3
                
                segments.append({
                    'id': i,
                    'start': start,
                    'end': end,
                    'text': segment_text,
                    'words': []
                })
                
            return {
                "text": full_text.strip(),
                "segments": segments
            }
            
        # Fallback for unknown formats
        return {
            "text": str(result),
            "segments": [{
                "id": 0, 
                "start": 0, 
                "end": 10, 
                "text": str(result),
                "words": []
            }]
        }
    
    def transcribe(self, audio_file, language=None, batch_size=None, max_speakers=None):
        """
        Transcribe an audio file using Lightning Whisper MLX
        
        Args:
            audio_file (str): Path to the audio file
            language (str, optional): Language code (e.g., 'en', 'fr', 'de')
            batch_size (int, optional): Override batch size for this transcription
            max_speakers (int, optional): Maximum number of speakers to detect
            
        Returns:
            dict: Transcription result with segments
        """
        # Reset progress tracking
        reset_progress()
        
        # Create a container to hold the result from the thread
        result_container = {"result": None, "success": False, "error": None}
        
        # Start transcription in a separate thread to allow progress reporting
        transcription_thread = threading.Thread(
            target=self._transcribe_thread,
            args=(audio_file, language, batch_size, result_container)
        )
        transcription_thread.start()
        transcription_thread.join()  # Wait for completion
        
        # Check if transcription was successful
        if not result_container["success"]:
            raise Exception(result_container["error"])
        
        return result_container["result"] 