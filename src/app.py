import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
import uuid
import platform
from services.whisper_service import WhisperService
from services.mlx_whisper_service import MLXWhisperService, get_progress_info

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder="../public", template_folder="../src/components/templates")

# Detect if running on Apple Silicon
is_apple_silicon = (platform.processor() == 'arm' and platform.system() == 'Darwin')

# Initialize WhisperService
# The compute_type will be determined by the service based on the device
whisper_service = WhisperService(
    hf_token=os.getenv("HF_TOKEN"),
    model_name="large-v2",
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
            backend = request.form.get('backend', 'whisperx')
            model_name = request.form.get('model_name', 'large-v2')
            batch_size = int(request.form.get('batch_size', 12)) if request.form.get('batch_size') else 12
            
            # Initialize service based on selected backend
            if backend == 'mlx' and is_apple_silicon:
                # Use MLX Whisper for Apple Silicon
                temp_service = MLXWhisperService(
                    hf_token=os.getenv("HF_TOKEN"),
                    model_name=model_name,
                    compute_type=compute_type,
                    batch_size=batch_size
                )
            else:
                # Use WhisperX as default or fallback
                temp_service = WhisperService(
                    hf_token=os.getenv("HF_TOKEN"),
                    model_name=model_name,
                    compute_type=compute_type
                )
            
            result = temp_service.transcribe(
                audio_file=filepath,
                language=language,
                max_speakers=max_speakers if backend != 'mlx' else None
            )
            
            # Return result
            return jsonify({
                "success": True,
                "filename": filename,
                "result": result,
                "backend": backend,
                "device_info": temp_service.device_info
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
    progress_data = get_progress_info()
    return jsonify(progress_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') 