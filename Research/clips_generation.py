'''
clips_generation.py
Purpose:
- Detect silence boundaries in a long audio file using FFmpeg's silencedetect filter.
- Build segments between consecutive silence_end timestamps.
- Enforce a minimum clip length by merging forward if a segment is shorter than 5 seconds.
- Export each segment to WAV (16-bit PCM) using ffmpeg-python with the same time bounds.

Warnings:
- Requires FFmpeg installed and available on PATH (ffmpeg/ffprobe must be callable).
- Silence detection is heuristic; tune noise (dB) and duration (sec) for your material.
- This script preserves your original logic (FFmpeg CLI for detection, ffmpeg-python for export).
- Paths are relative to the repository root: input at data/hungergames.wav, outputs to data/HGclips/.
'''

import os
import ffmpeg
import subprocess
from pathlib import Path

# Define input and output paths (relative to project root)
BASE_DIR = Path(__file__).resolve().parent
input_audio = BASE_DIR / "data" / "hungergames.wav"
output_folder = BASE_DIR / "data" / "HGclips"

# Ensure output folder exists
output_folder.mkdir(parents=True, exist_ok=True)

# Step 1: Detect silence and extract timestamps (FFmpeg CLI)
cmd = f'ffmpeg -i "{input_audio}" -af silencedetect=noise=-30dB:d=0.5 -f null -'
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

# Extract silence timestamps from FFmpeg output
silence_starts = []
for line in result.stderr.split("\n"):
    if "silence_end" in line:
        try:
            silence_starts.append(float(line.split("silence_end: ")[1].split(" | ")[0]))
        except Exception:
            # If parsing fails for any line, skip it
            pass

# Step 2: Split the audio at silent parts and ensure minimum length (merge short segments forward)
i = 0
clips = []

while i < len(silence_starts) - 1:
    start = silence_starts[i]
    end = silence_starts[i + 1]
    clip_length = end - start

    # If the clip is too short (<5 sec), merge with the next one
    while clip_length < 5 and i < len(silence_starts) - 2:
        i += 1
        end = silence_starts[i + 1]
        clip_length = end - start  # Update clip length after merging

    clips.append((start, end))
    i += 1  # Move to the next clip

# Step 3: Save the merged clips (ffmpeg-python)
for idx, (start, end) in enumerate(clips):
    output_file = output_folder / f"clip_{idx:03d}.wav"

    (
        ffmpeg
        .input(str(input_audio), ss=start, to=end)
        .output(str(output_file), format="wav", acodec="pcm_s16le")
        .run(overwrite_output=True)
    )

print("Audio splitting at silence points complete with minimum clip length!")
