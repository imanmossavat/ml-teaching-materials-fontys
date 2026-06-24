import os
import soundfile as sf
import sounddevice as sd
import numpy as np
from datetime import datetime

import torch

from TTS.tts.models.xtts import (
    XttsAudioConfig,
    XttsArgs,
)

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    BaseDatasetConfig,
    XttsArgs
])
from TTS.api import TTS   # 👈 XTTS v2 comes from Coqui TTS


# ==========================================
# 1. FILE SYSTEM & AUDIO CAPTURE
# ==========================================

def generate_output_path(filename="cloned_output.wav"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(data_dir, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)
    return os.path.join(timestamped_dir, filename)


def record_from_microphone(duration=7, sample_rate=24000):
    temp_ref_path = generate_output_path("live_microphone_ref.wav")

    print("\n" + "="*50)
    print(f"READY TO RECORD: {duration} seconds")
    print("PROMPT:")
    print("  'The quick brown fox jumps over the lazy dog.'")
    print("="*50)

    input("\n🎤 Press ENTER to start...")
    print("🔴 RECORDING...")

    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    print("🛑 DONE RECORDING")

    sf.write(temp_ref_path, audio_data, sample_rate)
    return temp_ref_path


# ==========================================
# 2. XTTS v2 VOICE CLONING CORE
# ==========================================

def read_text_with_xtts(
    text_to_generate,
    reference_audio_path,
    output_file=None
):
    print("\nInitializing XTTS v2 Pipeline...")

    if output_file is None:
        output_file = generate_output_path()

    # device is handled internally by XTTS (important difference!)
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    print("Generating cloned speech...")

    wav = tts.tts(
        text=text_to_generate,
        speaker_wav=reference_audio_path,
        language="en"
    )

    sf.write(output_file, wav, 24000)

    print(f"\n✨ Saved to: {os.path.abspath(output_file)}")


# ==========================================
# 3. ORCHESTRATOR
# ==========================================

if __name__ == "__main__":
    text_block = (
        "This is a live engineering test. "
        "The cloned voice is generated from a microphone sample."
    )

    microphone_sample_path = record_from_microphone(duration=6, sample_rate=24000)

    target_destination = generate_output_path("xtts_cloned_reading.wav")

    read_text_with_xtts(
        text_to_generate=text_block,
        reference_audio_path=microphone_sample_path,
        output_file=target_destination
    )

#%%
from TTS.api import TTS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

wav = tts.tts(
    text="Hello, this is a test.",
    speaker_wav="your_audio.wav",
    language="en"
)
# %%
