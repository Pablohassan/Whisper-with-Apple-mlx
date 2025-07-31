import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
import uuid
import platform
from services.whisper_service import WhisperService, get_whisperx_progress_info, reset_whisperx_progress
from services.mlx_whisper_service import MLXWhisperService, get_progress_info, reset_progress
from services.whisperkit_mlx_service import WhisperKitMLXService, get_whisperkit_progress_info, reset_whisperkit_progress

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder="../public", template_folder="../src/components/templates")

# Detect if running on Apple Silicon
is_apple_silicon = (platform.processor() == 'arm' and platform.system() == 'Darwin')

# Initialize metrics for all backends at startup
reset_whisperx_progress()
reset_progress()
reset_whisperkit_progress()

print("Initialized metrics for all backends")

# Initialize WhisperService
# The compute_type will be determined by the service based on the device
whisper_service = WhisperService(
    hf_token=os.getenv("HF_TOKEN"),
    model_name="large-v3",
    # Let the service determine the appropriate compute_type
    # Based on device (CPU, CUDA, MPS)
)

# Ensure upload directory exists
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    # Pass information about Apple Silicon to the template
    return render_template('index.html', is_apple_silicon=is_apple_silicon)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Generate unique filename
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process with selected backend
        try:
            # Get parameters from form
            language = request.form.get('language', None)
            max_speakers = int(request.form.get('max_speakers', 2))
            compute_type = request.form.get('compute_type', whisper_service.compute_type)
            backend = request.form.get('backend', 'whisperkit')  # Default to WhisperKit on Apple Silicon
            model_name = request.form.get('model_name', 'large-v3')
            batch_size = int(request.form.get('batch_size', 12)) if request.form.get('batch_size') else 12
            output_name = request.form.get('output_name', 'transcript').strip()
            
            # Initialize service based on selected backend
            if backend == 'whisperkit' and is_apple_silicon:
                # Use WhisperKit by Argmax for Apple Silicon
                # Configure with French as default language for better consistency
                temp_service = WhisperKitMLXService(
                    hf_token=os.getenv("HF_TOKEN"),
                    model_name=model_name,
                    compute_type=compute_type,
                    batch_size=batch_size,
                    default_language='fr'  # Force French by default for better consistency
                )
            elif backend == 'mlx' and is_apple_silicon:
                # Use Lightning Whisper MLX for Apple Silicon
                temp_service = MLXWhisperService(
                    hf_token=os.getenv("HF_TOKEN"),
                    model_name=model_name,
                    compute_type=compute_type,
                    batch_size=batch_size
                )
            else:
                # Use WhisperX as fallback
                temp_service = WhisperService(
                    hf_token=os.getenv("HF_TOKEN"),
                    model_name=model_name,
                    compute_type=compute_type
                )
            
            result = temp_service.transcribe(
                audio_file=filepath,
                language=language,
                max_speakers=max_speakers if backend not in ['mlx', 'whisperkit'] else None
            )
            
            # Return result
            return jsonify({
                "success": True,
                "filename": filename,
                "result": result,
                "backend": backend,
                "device_info": temp_service.device_info,
                "output_name": output_name
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_transcription(filename):
    # Get the transcription content from the request
    content = request.args.get('content')
    
    if not content:
        return jsonify({"error": "No content provided"}), 400
    
    # Generate a unique filename for the transcription
    basename = os.path.splitext(filename)[0]
    transcription_filename = f"{basename}_transcription.txt"
    transcription_path = os.path.join(UPLOAD_FOLDER, transcription_filename)
    
    # Write content to file
    with open(transcription_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return send_from_directory(
        UPLOAD_FOLDER, 
        transcription_filename, 
        as_attachment=True
    )

@app.route('/progress', methods=['GET'])
def get_progress():
    """Return current transcription progress"""
    # Try to get progress from different backends
    backend = request.args.get('backend', 'whisperx')
    
    # Force update of system metrics before returning
    try:
        import psutil
        
        if backend == 'whisperkit':
            from services.whisperkit_mlx_service import update_system_metrics as update_whisperkit_metrics
            update_whisperkit_metrics()
            progress_data = get_whisperkit_progress_info()
        elif backend == 'mlx':
            from services.mlx_whisper_service import update_system_metrics as update_mlx_metrics
            update_mlx_metrics()
            progress_data = get_progress_info()
        else:  # whisperx or fallback
            from services.whisper_service import update_whisperx_system_metrics
            update_whisperx_system_metrics()
            progress_data = get_whisperx_progress_info()
    except Exception as e:
        print(f"Error updating metrics: {e}")
        # Fallback to basic progress data
        if backend == 'whisperkit':
            progress_data = get_whisperkit_progress_info()
        elif backend == 'mlx':
            progress_data = get_progress_info()
        else:
            progress_data = get_whisperx_progress_info()
    
    return jsonify(progress_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 