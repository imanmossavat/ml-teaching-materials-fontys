import os
import torch
import soundfile as sf
import numpy as np
from kokoro import KPipeline
from datetime import datetime

# Diagnostic Modules
import logging
import sys
import warnings
import traceback
import inspect
import functools
import types

# ==========================================
# 1. FILE SYSTEM / UTILITY CONCERN
# ==========================================
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
