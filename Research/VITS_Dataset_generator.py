'''
dataset_generation.py
Purpose:
- Batch-transcribe WAV clips in data/HGclips using OpenAI Whisper (local) and write a dataset file.
- Normalizes transcripts by expanding numbers to words and common contractions to full forms.
- Outputs lines in the format: "clip_XXX.wav | normalized text" into data/HGclips/dataset.txt.

Warnings:
- Whisper runs locally and can be slow on CPU; a CUDA-enabled environment is recommended.
- The "base" model is used by default for speed; switch to "medium"/"large" for better accuracy (requires more VRAM).
- This script expects WAV clips in data/HGclips and writes dataset.txt in the same folder.
- Text normalization here is simple and may need expansion for production-quality corpora.
'''

import os
import re
from pathlib import Path

import whisper
import inflect  # For number conversion


# Load Whisper model (runs locally). Options: "tiny", "base", "small", "medium", "large"
model = whisper.load_model("base")  # change to "medium"/"large" for better accuracy (slower, more VRAM)

# Initialize inflect engine for number conversion
p = inflect.engine()


def normalize_text(text: str) -> str:
    """
    Normalize text for TTS training:
    - Convert standalone numbers (e.g., 12 -> twelve).
    - Expand common contractions (e.g., "don't" -> "do not").
    """
    # Convert standalone numbers to words
    text = re.sub(r'\b\d+\b', lambda x: p.number_to_words(x.group()), text)

    # Handle common contractions (case-sensitive basic pass)
    contractions = {
        "aren't": "are not",
        "can't": "cannot",
        "don't": "do not",
        "it's": "it is",
        "you're": "you are",
        "we're": "we are",
        "they're": "they are",
        "isn't": "is not",
        "wasn't": "was not",
        "won't": "will not",
    }
    for contraction, full_form in contractions.items():
        text = text.replace(contraction, full_form)

    return text.strip()


# ---------------------------
# Relative paths (portable)
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
CLIP_FOLDER = BASE_DIR / "data" / "HGclips"
TRANSCRIPT_FILE = CLIP_FOLDER / "dataset.txt"

# Ensure output folder exists
CLIP_FOLDER.mkdir(parents=True, exist_ok=True)

# Get all audio clips (WAV only, sorted)
audio_clips = sorted([f for f in os.listdir(CLIP_FOLDER) if f.lower().endswith(".wav")])

# Open dataset file and process
with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as transcript:
    for clip in audio_clips:
        clip_path = CLIP_FOLDER / clip

        # Transcribe clip using Whisper
        result = model.transcribe(str(clip_path))
        text = (result.get("text") or "").strip()

        # Normalize text
        normalized_text = normalize_text(text)

        # Save transcript in dataset format (filename | text)
        transcript.write(f"{clip} | {normalized_text}\n")

        print(f"Processed {clip}: {normalized_text}")

print(f"Transcription complete! Dataset saved as '{TRANSCRIPT_FILE.name}'.")
