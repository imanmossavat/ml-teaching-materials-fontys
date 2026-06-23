"""
Local Text-to-Speech Engine using Kokoro TTS.

Dependencies:
    brew install espeak-ng soundfile
    pip install torch kokoro soundfile numpy

LLM Pre-Processing Prompt:
    "Convert the following markdown or academic document into a raw text block optimized for Text-to-Speech reading.
    1. Transform all mathematical equations, variables, and LaTeX expressions into fully spelled-out, naturally spoken English words.
    2. Convert markdown headers into spoken cues (e.g., '# Title' becomes 'Document Title: [Title].', '## Heading' becomes 'Section Heading: [Heading].').
    3. Completely strip out all markdown syntax like asterisks, brackets, citation numbers, and reference blocks.
    4. Provide only the final clean text block without any markdown code wrappers or introductory conversational comments."

Way of Working:
    1. Pass any file path directly to the paths resolver.
    2. The script finds where that file lives and creates a relative 'data/YYYYMMDD_HHMMSS/' workspace directly adjacent to it.
    3. If the input path does not exist on your file system, it falls back to the script's root directory.
"""

import os
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline


# ==========================================================
# GLOBAL PIPELINE CACHE
# ==========================================================
_PIPELINE_CACHE = {}


# ==========================================================
# 1. FILE SYSTEM / UTILITY CONCERN
# ==========================================================
def get_script_dir():
    """
    Robust script location resolver.
    Works in normal scripts, IPython, and Jupyter.
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

def resolve_io_paths(input_filepath=None, output_filename=None):
    """
    The supplied path is treated as the intended input location.

    If the file exists:
        - workspace is created beside it

    If the file does not exist:
        - the future file location is still respected
        - workspace is created beside where the file will live
    """

    script_dir = get_script_dir()

    if output_filename is None:
        output_filename = "scientific_reading.wav"

    if input_filepath is None:
        input_filepath = os.path.join(
            script_dir,
            "input.txt"
        )

    absolute_input_path = os.path.abspath(
        input_filepath
    )

    base_dir = os.path.dirname(
        absolute_input_path
    )

    os.makedirs(
        base_dir,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    target_data_dir = os.path.join(
        base_dir,
        "data",
        timestamp
    )

    os.makedirs(
        target_data_dir,
        exist_ok=True
    )

    absolute_output_path = os.path.join(
        target_data_dir,
        output_filename
    )

    return absolute_input_path, absolute_output_path

def get_speech_content(input_path):
    """
    Reads text from the supplied path.

    If the file does not exist, creates a starter template.
    """

    if not os.path.exists(input_path):

        parent_dir = os.path.dirname(input_path)

        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        default_content = (
            "Document Title: Quantum Superposition Essentials.\n"
            "Section Heading: One point One, Core Principles.\n"
            "Superposition is a fundamental principle of quantum mechanics. "
            "It states that physical systems exist in multiple states "
            "simultaneously until they are actively measured. "
            "The unified state is written as psi equals alpha times ket zero, "
            "plus beta times ket one."
        )

        with open(input_path, "w", encoding="utf-8") as f:
            f.write(default_content)

        print(f"Created a sample input file at: {input_path}")


        return default_content

    with open(input_path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ==========================================================
# 2. HARDWARE / PIPELINE CONCERN
# ==========================================================
def get_best_device():
    """
    Returns the best available backend.
    """

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_pipeline(lang_code, device):
    """
    Cached Kokoro pipeline loader.
    Prevents reloading models repeatedly.
    """

    key = (lang_code, device)

    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = KPipeline(
            lang_code=lang_code,
            device=device
        )

    return _PIPELINE_CACHE[key]


# ==========================================================
# 3. AUDIO GENERATION CONCERN
# ==========================================================
def read_text_on_mac(
    text_to_speak,
    output_file,
    lang_code="a",
    voice="af_bella",
    sample_rate=24000,
    speed=1.0,
    normalize_output=False
):
    """
    Generates narration using Kokoro TTS.

    normalize_output=False:
        Stream audio directly to disk (recommended for large documents).

    normalize_output=True:
        Hold audio in memory and normalize entire file before saving.
    """

    if not text_to_speak:
        print("Error: Input text target is empty.")
        return

    print("Initializing Kokoro TTS...")

    device = get_best_device()

    print(
        f"Using Hardware Acceleration Backend: "
        f"{device.upper()}"
    )

    try:
        pipeline = get_pipeline(
            lang_code=lang_code,
            device=device
        )
    except Exception as e:
        print(f"Failed to initialize Kokoro pipeline: {e}")
        return

    print("Generating studio-quality narration...")

    try:
        generator = pipeline(
            text_to_speak,
            voice=voice,
            speed=speed
        )
    except Exception as e:
        print(f"Speech generation failed: {e}")
        return

    total_samples = 0
    processed_blocks = 0

    # ------------------------------------------------------
    # STREAMING MODE (recommended)
    # ------------------------------------------------------
    if not normalize_output:

        try:
            with sf.SoundFile(
                output_file,
                mode="w",
                samplerate=sample_rate,
                channels=1
            ) as outfile:

                for i, (_, _, audio_chunk) in enumerate(generator):

                    if audio_chunk is None or len(audio_chunk) == 0:
                        continue

                    if hasattr(audio_chunk, "cpu"):
                        audio_chunk = audio_chunk.cpu().numpy()

                    peak = np.max(np.abs(audio_chunk))

                    if peak > 1.0:
                        audio_chunk = audio_chunk / peak

                    outfile.write(audio_chunk)

                    total_samples += len(audio_chunk)
                    processed_blocks += 1

                    print(
                        f"Processed sentence block #{i + 1}"
                    )

        except Exception as e:
            print(f"Audio write failed: {e}")
            return

    # ------------------------------------------------------
    # FULL NORMALIZATION MODE
    # ------------------------------------------------------
    else:

        audio_segments = []

        try:
            for i, (_, _, audio_chunk) in enumerate(generator):

                if audio_chunk is None or len(audio_chunk) == 0:
                    continue

                if hasattr(audio_chunk, "cpu"):
                    audio_chunk = audio_chunk.cpu().numpy()

                audio_segments.append(audio_chunk)

                total_samples += len(audio_chunk)
                processed_blocks += 1

                print(
                    f"Processed sentence block #{i + 1}"
                )

        except Exception as e:
            print(f"Generation failed: {e}")
            return

        if not audio_segments:
            print("Error: No audio could be synthesized.")
            return

        final_audio = np.concatenate(audio_segments)

        peak = np.max(np.abs(final_audio))

        if peak > 0:
            final_audio = final_audio / peak * 0.98

        try:
            sf.write(
                output_file,
                final_audio,
                sample_rate
            )
        except Exception as e:
            print(f"Audio save failed: {e}")
            return

    duration_seconds = (
        total_samples / sample_rate
        if sample_rate > 0
        else 0
    )

    print()
    print("===================================")
    print("Narration Complete")
    print("===================================")
    print(f"Blocks processed : {processed_blocks}")
    print(f"Duration         : {duration_seconds:.1f} seconds")
    print(f"Audio saved to   : {output_file}")
    print("===================================")


# ==========================================================
# 4. COORDINATION ORCHESTRATOR
# ==========================================================
if __name__ == "__main__":

    # Change this to any text file on your machine.
    my_test_file = None

    resolved_input, resolved_output = resolve_io_paths(
        input_filepath=my_test_file,
        output_filename="scientific_reading.wav"
    )

    document_content = get_speech_content(
        resolved_input
    )

    read_text_on_mac(
        text_to_speak=document_content,
        output_file=resolved_output,
        voice="af_bella",
        speed=1.0,
        normalize_output=False
    )

