'''
app.py
Purpose:
- Flask web app for text-to-speech with three options:
  1) VITS (demo pretrained voice)
  2) XTTS zero-shot (default narrator clip)
  3) Custom upload (user-provided 10–30s reference clip), auto-converted to WAV
- Supports long-text chunking and chunk merge via pydub/ffmpeg.
- Cleans old outputs and removes custom upload after synthesis.

Warnings:
- Requires ffmpeg installed and on PATH for pydub conversions and WAV export.
- Uses GPU if available for XTTS; automatically falls back to CPU when CUDA is unavailable.
- Do not commit data/Outputs or uploaded clips to Git; keep them ignored.
- Large model files are cached outside the repo by the TTS library automatically.
'''

from flask import Flask, request, jsonify, render_template
from TTS.utils.synthesizer import Synthesizer
from TTS.api import TTS
from pydub import AudioSegment
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import time
import glob
import re

# --------------------------------------------------------------------------------------
# Paths and configuration (relative to project root)
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "Outputs"          # generated audio files
SAMPLE_CLIP_DIR = DATA_DIR / "tmp"         # temporary custom uploads
TM_CLIP_DIR = DATA_DIR / "TM_Clip"         # default narrator/reference clip for XTTS

# Ensure folders exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_CLIP_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Flask app using explicit template/static folders
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))

# Allowed extensions for upload
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac', 'wma', 'webm', 'opus'}

# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_old_files(retention_seconds: int = 3600) -> None:
    """Delete audio_*.wav in static folder older than retention_seconds."""
    current_time = time.time()
    audio_files = glob.glob(str(STATIC_DIR / "audio_*.wav"))
    for file in audio_files:
        try:
            if os.path.exists(file) and (current_time - os.path.getmtime(file)) > retention_seconds:
                os.remove(file)
                print(f"🗑️ Deleted old file: {file}")
        except Exception as e:
            print(f"⚠️ Could not delete {file}: {e}")

def cleanup_sample_clips() -> None:
    """Delete all files in temp custom upload directory."""
    try:
        files = glob.glob(str(SAMPLE_CLIP_DIR / "*"))
        for f in files:
            try:
                os.remove(f)
                print(f"🗑️ Deleted sample clip: {Path(f).name}")
            except Exception as e:
                print(f"⚠️ Could not delete {f}: {e}")
    except Exception as e:
        print(f"⚠️ Error cleaning sample clips: {e}")

def split_text_into_chunks(text: str, max_chars: int = 1000):
    """Split text into manageable chunks by paragraphs, then sentences."""
    paragraphs = text.split('\n\n')
    chunks, current = [], ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if len(current) + len(sentence) > max_chars and current:
                    chunks.append(current.strip())
                    current = sentence + " "
                else:
                    current += sentence + " "
        else:
            if len(current) + len(para) > max_chars and current:
                chunks.append(current.strip())
                current = para + "\n\n"
            else:
                current += para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]

# --------------------------------------------------------------------------------------
# (Optional) Torch diagnostics
# --------------------------------------------------------------------------------------
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"⚠️ Torch diagnostics failed: {e}")

# --------------------------------------------------------------------------------------
# Load models (with CUDA→CPU fallback)
# --------------------------------------------------------------------------------------
print("🔄 Loading default VITS model...")
vits_tts = None
try:
    # Try GPU hint; if unsupported, Coqui may still run CPU.
    vits_tts = TTS("tts_models/en/ljspeech/vits", gpu=True)
    print("✅ VITS model loaded successfully!")
except Exception as e:
    print(f"⚠️ VITS load failed with gpu=True: {e}")
    print("   ↳ Retrying VITS on CPU...")
    try:
        vits_tts = TTS("tts_models/en/ljspeech/vits", gpu=False)
        print("✅ VITS model loaded on CPU.")
    except Exception as e2:
        print(f"❌ VITS model failed on CPU as well: {e2}")
        vits_tts = None

print("🔄 Loading XTTS zero-shot voice cloning model (CUDA→CPU fallback)...")
xtts_tts = None
device_used = None
try:
    xtts_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
    device_used = "CUDA"
except Exception as e:
    print(f"⚠️ XTTS CUDA failed: {e}")
    print("   ↳ Retrying XTTS on CPU...")
    try:
        xtts_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        device_used = "CPU"
    except Exception as e2:
        print(f"❌ XTTS failed on CPU as well: {e2}")
        xtts_tts = None
        device_used = None

# Default narrator/reference clip for XTTS (TM option)
# Expect a file like: data/TM_Clip/clip_001.wav (you can change the name)
default_tm_clip = TM_CLIP_DIR / "TM_clip.wav"
if not default_tm_clip.exists():
    print("⚠️ Warning: Default TM voice reference clip not found in data/TM_Clip/")
    NARRATOR_VOICE_CLIP = None
else:
    NARRATOR_VOICE_CLIP = str(default_tm_clip)

if xtts_tts and device_used:
    name = default_tm_clip.name if NARRATOR_VOICE_CLIP else "N/A"
    print(f"✅ XTTS loaded on {device_used}! Default voice: {name}")

print("\n" + "="*70)
print("✅ All models loaded and ready!")
print("="*70)
print("Available voices:")
if vits_tts:
    print("  ✓ VITS (default pre-trained)")
if xtts_tts:
    print(f"  ✓ XTTS (zero-shot voice cloning) [{device_used}]")
print("  ✓ Custom (upload your own voice)")
print("="*70 + "\n")

# Quick health summary and endpoint
print("Health summary:")
print(f" - VITS available: {bool(vits_tts)}")
print(f" - XTTS available: {bool(xtts_tts)}")
print(f" - Default TM clip found: {bool(NARRATOR_VOICE_CLIP)}")
print("="*70 + "\n")

@app.route("/health")
def health():
    return jsonify({
        "vits_available": bool(vits_tts),
        "xtts_available": bool(xtts_tts),
        "tm_clip_found": bool(NARRATOR_VOICE_CLIP),
        "xtts_device": device_used or "N/A"
    }), 200

# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_voice", methods=["POST"])
def upload_voice():
    """
    Handle custom voice file upload with auto-conversion to WAV (22,050 Hz).
    Returns a JSON { filename } to be passed back to /synthesize when voice='custom'.
    """
    if 'voice_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['voice_file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    try:
        # Clean old custom clips
        cleanup_sample_clips()

        # Save to temp with original extension
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        original_ext = filename.rsplit('.', 1)[1].lower()

        temp_filename = f"temp_upload_{timestamp}.{original_ext}"
        temp_filepath = SAMPLE_CLIP_DIR / temp_filename
        file.save(str(temp_filepath))
        print(f"📁 Saved temporary file: {temp_filename}")

        # Convert to standardized WAV 22,050 Hz
        final_filename = f"custom_voice_{timestamp}.wav"
        final_filepath = SAMPLE_CLIP_DIR / final_filename

        if original_ext == 'wav':
            temp_filepath.replace(final_filepath)
            print(f"✅ Using WAV directly: {final_filename}")
        else:
            from pydub import AudioSegment
            print(f"🔄 Converting {original_ext.upper()} to WAV...")
            audio = AudioSegment.from_file(str(temp_filepath), format=original_ext)
            audio.export(str(final_filepath), format="wav", parameters=["-ar", "22050"])
            temp_filepath.unlink(missing_ok=True)
            print(f"✅ Converted to WAV: {final_filename}")

        return jsonify({
            "message": "File uploaded and converted successfully",
            "filename": final_filename,
            "original_format": original_ext
        })

    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/synthesize", methods=["POST"])
def synthesize():
    """
    Synthesize audio using selected voice.
    - voice='vits' uses pretrained VITS
    - voice='xtts' uses default narrator clip (data/TM_Clip/clip_001.wav)
    - voice='custom' uses the uploaded/converted file from data/tmp/
    """
    data = request.json
    text = data.get("text", "")
    voice = data.get("voice", "vits")
    custom_voice_file = data.get("custom_voice_file", None)

    if not text:
        return jsonify({"error": "No text provided"}), 400

    cleanup_old_files()

    timestamp = int(time.time())
    output_path = STATIC_DIR / f"audio_{timestamp}.wav"

    char_count = len(text)
    word_count = len(text.split())

    print("\n" + "="*70)
    print(f"📝 Text: {char_count} chars, {word_count} words")
    print(f"   Voice: {voice.upper()}")
    print("="*70)

    # Defensive availability checks
    if voice == "vits" and not vits_tts:
        return jsonify({"error": "VITS model not available. Check /health and console logs. Ensure internet on first run and restart the app."}), 503
    if voice == "xtts" and not xtts_tts:
        return jsonify({"error": "XTTS model not available. Check /health and console logs. Ensure internet on first run and restart the app."}), 503
    if voice == "xtts" and not NARRATOR_VOICE_CLIP:
        return jsonify({"error": "Default TM clip missing (data/TM_Clip/clip_001.wav). Add a reference file or use Custom upload."}), 400
    if voice == "custom" and not xtts_tts:
        return jsonify({"error": "XTTS model not available for Custom voice. Check /health and console logs."}), 503

    try:
        voice_reference = None
        if voice == "custom":
            if not custom_voice_file:
                return jsonify({"error": "No custom voice file provided"}), 400
            voice_reference_path = SAMPLE_CLIP_DIR / custom_voice_file
            if not voice_reference_path.exists():
                return jsonify({"error": "Custom voice file not found"}), 404
            voice_reference = str(voice_reference_path)
            print(f"🎤 Using custom voice: {voice_reference_path.name}")

        # Chunking for long text
        if char_count > 1000:
            chunks = split_text_into_chunks(text, max_chars=1000)
            print(f"\n📊 Split into {len(chunks)} chunks")

            temp_files = []
            for i, chunk in enumerate(chunks, 1):
                chunk_file = STATIC_DIR / f"temp_chunk_{timestamp}_{i}.wav"
                print(f"\n🎤 Chunk {i}/{len(chunks)} ({(i/len(chunks))*100:.0f}%)")
                t0 = time.time()

                if voice == "vits":
                    vits_tts.tts_to_file(text=chunk, file_path=str(chunk_file))
                elif voice == "xtts":
                    xtts_tts.tts_to_file(text=chunk, speaker_wav=NARRATOR_VOICE_CLIP, language="en", file_path=str(chunk_file))
                elif voice == "custom":
                    xtts_tts.tts_to_file(text=chunk, speaker_wav=voice_reference, language="en", file_path=str(chunk_file))

                print(f"   ✅ Done in {time.time() - t0:.1f}s")
                temp_files.append(chunk_file)

            # Merge chunks
            print(f"\n🔗 Combining {len(temp_files)} chunks...")
            combined = AudioSegment.empty()
            for temp_file in temp_files:
                combined += AudioSegment.from_wav(str(temp_file))
                try:
                    temp_file.unlink(missing_ok=True)
                except Exception:
                    pass
            combined.export(str(output_path), format="wav")
            print("✅ Combined")

        else:
            print("\n🎤 Processing single audio")
            t0 = time.time()

            if voice == "vits":
                vits_tts.tts_to_file(text=text, file_path=str(output_path))
            elif voice == "xtts":
                xtts_tts.tts_to_file(text=text, speaker_wav=NARRATOR_VOICE_CLIP, language="en", file_path=str(output_path))
            elif voice == "custom":
                xtts_tts.tts_to_file(text=text, speaker_wav=voice_reference, language="en", file_path=str(output_path))

            print(f"   ✅ Done in {time.time() - t0:.1f}s")

        final_audio = AudioSegment.from_wav(str(output_path))
        audio_duration = len(final_audio) / 1000.0
        print(f"\n✅ Generated: {audio_duration:.1f}s audio")

        if voice == "custom":
            cleanup_sample_clips()
            print("🗑️ Cleaned up sample")

        print("="*70 + "\n")

        # Return path relative to Flask static/ for the frontend
        rel_file = f"{output_path.relative_to(BASE_DIR).as_posix()}"
        return jsonify({
            "message": "Success",
            "file": rel_file,
            "duration": f"{audio_duration:.1f}s",
            "chunks": len(chunks) if char_count > 1000 else 1
        })

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("="*70 + "\n")
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🌐 Server: http://localhost:5000")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
