import os
import datetime

def ensure_dir(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (str): Path to the directory
        
    Returns:
        str: Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def generate_filename(original_filename=None, extension=".txt"):
    """
    Generate a unique filename based on the current timestamp and optionally the original filename.
    
    Args:
        original_filename (str, optional): Original filename
        extension (str): File extension to use
        
    Returns:
        str: Generated filename
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if original_filename:
        # Extract the base name without extension
        basename = os.path.splitext(os.path.basename(original_filename))[0]
        # Sanitize the filename
        basename = ''.join(c for c in basename if c.isalnum() or c in '._- ')
        return f"{basename}_{timestamp}{extension}"
    
    return f"transcription_{timestamp}{extension}"

def format_time(seconds):
    """
    Format time in seconds to HH:MM:SS.ms format.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    else:
        return f"{minutes:02d}:{seconds:06.3f}"

def format_transcript_for_export(segments, include_speakers=True):
    """
    Format transcription segments for export as text.
    
    Args:
        segments (list): List of transcription segments
        include_speakers (bool): Whether to include speaker information
        
    Returns:
        str: Formatted transcription text
    """
    text = ""
    
    for segment in segments:
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        
        # Add timestamp
        text += f"[{start_time} - {end_time}]"
        
        # Add speaker if available and requested
        if include_speakers and "speaker" in segment and segment["speaker"]:
            text += f" Speaker {segment['speaker']}:"
        
        # Add text
        text += f" {segment['text']}\n\n"
    
    return text 