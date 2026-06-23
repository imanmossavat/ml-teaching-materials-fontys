"""
Local Text-to-Speech Engine using Kokoro TTS.
Install the dependencies below to run this script. Then run the script to generate a WAV file from the provided text block.

Dependencies:
    brew install espeak-ng soundfile
    pip install torch kokoro soundfile numpy

You can use the LLM prompt to pre-process your markdown or academic document into a raw text block optimized for TTS reading.

LLM Pre-Processing Prompt:
    "Convert the following markdown or academic document into a raw text block optimized for Text-to-Speech reading. 
    1. Transform all mathematical equations, variables, and LaTeX expressions into fully spelled-out, naturally spoken English words. 
    2. Convert markdown headers into spoken cues (e.g., '# Title' becomes 'Document Title: [Title].', '## Heading' becomes 'Section Heading: [Heading].'). 
    3. Completely strip out all markdown syntax like asterisks, brackets, citation numbers, and reference blocks. 
    4. Provide only the final clean text block without any markdown code wrappers or introductory conversational comments."

Way of Working:
    1. Paste the LLM-optimized text into the orchestrator block at the bottom of this script.
    2. Run the script. It automatically detects and leverages CUDA, Apple Silicon MPS, or CPU.
    3. A unique timestamped folder is created inside a 'data/' directory to store your generated WAV files without overwriting previous sessions.
"""

import os
import torch
import soundfile as sf
import numpy as np
from kokoro import KPipeline
from datetime import datetime


# ==========================================
# 1. FILE SYSTEM / UTILITY CONCERN
# ==========================================

def resolve_io_paths(input_filepath=None, output_filename=None):
    """
    Decoupled path resolution logic.
    Determines where the input file lives, and builds a relative 
    'data/YYYYMMDD_HHMMSS/' workspace directly adjacent to it.
    """
    # Find where the script is located to anchor default files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_input_path = os.path.dirname(os.path.abspath(input_filepath))
    
    # 1. Check if the input text file exists
    if os.path.exists(absolute_input_path):
        # Base the output data directory in the exact same folder as the text file
        base_dir = absolute_input_path
    else:
        # Fallback path if the text file doesn't exist yet
        base_dir = script_dir
        
    # 2. Build the structured timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_data_dir = os.path.join(base_dir, "data", timestamp)
    
    os.makedirs(target_data_dir, exist_ok=True)
    absolute_output_path = os.path.join(target_data_dir, output_filename)
    
    return absolute_input_path, absolute_output_path



def generate_output_path(filename="output.wav"):
    """
    Handles folder structural logic independently.
    Creates a 'data/YYYYMMDD_HHMMSS/' workspace relative to the script directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(data_dir, timestamp)
    
    os.makedirs(timestamped_dir, exist_ok=True)
    return os.path.join(timestamped_dir, filename)

# ==========================================
# 2. AUDIO GENERATION & HARDWARE CONCERN
# ==========================================
def read_text_on_mac(text_to_speak, output_file=None, lang_code='a', voice='af_bella', sample_rate=24000, speed=1.0):
    """
    Processes speech rendering. Fallback patterns safely decoupled from compilation initialization.
    """
    print("Initializing Kokoro TTS...")

    # Fix the signature compilation bug: evaluate path generation lazily at runtime execution
    if output_file is None:
        output_file = generate_output_path()

    # Dynamically detect hardware backends: Nvidia CUDA, Apple Silicon MPS, or standard CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using Hardware Acceleration Backend: {device.upper()}")

    # Initialize the English flavor using the resolved active hardware backend target
    pipeline = KPipeline(lang_code=lang_code, device=device)
    
    print("Generating studio-quality narration...")
    generator = pipeline(text_to_speak, voice=voice, speed=speed)
    
    audio_segments = []
    for i, (graphemes, phonemes, audio_chunk) in enumerate(generator):
        if audio_chunk is not None and len(audio_chunk) > 0:
            # Crucial Cross-Platform Safety Fix: Pull arrays safely out of CUDA VRAM to prevent file system crashes
            if hasattr(audio_chunk, "cpu"):
                audio_chunk = audio_chunk.cpu().numpy()
            audio_segments.append(audio_chunk)
            print(f"Processed sentence block #{i+1}")
            
    if not audio_segments:
        print("Error: No audio could be synthesized.")
        return

    # Combine all individual spoken lines into a single uniform audio timeline array
    final_audio = np.concatenate(audio_segments)
    
    # Save as an uncompressed high-fidelity studio WAV file matching configured metrics
    sf.write(output_file, final_audio, sample_rate)
    print(f"\n✨ Done! Audio saved to: {os.path.abspath(output_file)}")

# ==========================================
# 3. COORDINATION ORCHESTRATOR
# ==========================================
if __name__ == "__main__":
    # Example text with stripped academic formatting for fluent listening
    scientific_document_excerpt = (
        "The standard deviation, denoted by the Greek letter sigma, is calculated "
        "as the square root of the variance. Specifically, it equals the square root "
        "of one over N, multiplied by the sum of x sub i minus mu squared."
    )
    
    # Resolve the destination location target dynamically relative to runtime execution
    target_destination = generate_output_path("scientific_reading.wav")
    
    # Fire processing code
    read_text_on_mac(scientific_document_excerpt, output_file=target_destination)
