# **AI Audiobook Voice Generator**

A Flask-based web application for converting text to natural, expressive audiobook narration using **XTTS zero-shot voice cloning**. Upload a 10–30 second reference clip and generate hours of audio in that voice—**no training required**.

***

## **📖 Table of Contents**
- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [Features](#features)
- [The Journey: From Training to Zero-Shot](#the-journey-from-training-to-zero-shot)
- [Technology Stack](#technology-stack)
- [Usage](#usage)
- [Future Vision: Emotion-Aware Narration](#future-vision-emotion-aware-narration)
- [Research & Lessons Learned](#research--lessons-learned)

***

## **Overview**

This project delivers a production-ready **local web app** that synthesizes high-quality audiobook narration with **three voice options**:

1. **VITS** – Pre-trained demo voice (fast, no setup)
2. **XTTS (TM)** – Zero-shot cloning using a default narrator reference clip
3. **Custom Upload** – Upload any 10–30s voice sample; XTTS clones it instantly

**Key Capabilities:**
- **Zero training overhead** – XTTS clones voices from short samples without model fine-tuning
- **Long-text chunking** – Handles full chapters; auto-splits, synthesizes, and merges seamlessly
- **Multi-format upload** – Accepts WAV, MP3, M4A, FLAC, OGG, AAC, WMA, WebM, Opus (auto-converts to WAV)
- **GPU-accelerated** – CUDA-enabled PyTorch for fast synthesis; graceful CPU fallback
- **Clean UX** – Circular voice buttons, real-time upload status, audio player, and progress logs

***

Perfect! Here's the updated **Installation & Setup** section for your README.md that uses the `install_torch_cuda.py` script and removes the TM_Clip step:

---

## **Installation & Setup**

### **Prerequisites**
- **Python 3.11** installed ([python.org](https://www.python.org/downloads/))
- **Git** installed ([git-scm.com](https://git-scm.com/downloads))
- **NVIDIA GPU** (optional; CPU fallback supported but slower)
- **NVIDIA Drivers** (if using GPU; latest version recommended from [nvidia.com](https://www.nvidia.com/Download/index.aspx))
- **FFmpeg** installed and on PATH ([download guide](https://ffmpeg.org/download.html))

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/build-by-Ayush/AI-AudioBook-Voice-Generator.git
cd AI-AudioBook-Voice-Generator
```

### **Step 2: Create Virtual Environment**
```bash
python -m venv .venv
```

**Activate the environment:**

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

You should see `(.venv)` appear in your terminal prompt.

### **Step 3: Install PyTorch with CUDA Support (Automatic)**

Run the included helper script that **automatically detects your GPU** and installs the correct PyTorch build:

```bash
python install_torch_cuda.py
```

**What this does:**
- Detects your NVIDIA GPU and driver version
- Recommends CUDA 12.1 or CUDA 11.8 PyTorch wheels (or CPU-only if no GPU)
- Asks for confirmation before installing
- Verifies CUDA is working after installation

**If you don't have a GPU or prefer CPU:**
- The script will detect this and install CPU-only PyTorch
- Or manually run: `pip install torch torchaudio`

### **Step 4: Install Remaining Dependencies**
```bash
pip install -r requirements.txt
```

This installs:
- Flask (web server)
- TTS (Coqui TTS library with XTTS/VITS models)
- transformers (required for XTTS)
- pydub (audio processing)
- Other utilities

### **Step 5: Verify FFmpeg**
Ensure FFmpeg is installed and accessible:

```bash
ffmpeg -version
```

**Expected output:** Version info like `ffmpeg version 6.0...`

**If not found:**
- Windows: [Download FFmpeg](https://ffmpeg.org/download.html), extract, and add `bin/` folder to PATH
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg` (Debian/Ubuntu)

### **Step 6: Run the App**
```bash
python app.py
```

**Open your browser:** [http://localhost:5000](http://localhost:5000)

***

### **Quick Health Check**
After starting the app, visit:
```
http://localhost:5000/health
```

**Expected JSON response:**
```json
{
  "vits_available": true,
  "xtts_available": true,
  "xtts_device": "CUDA",
  "tm_clip_found": true
}
```

**What the fields mean:**
- `vits_available`: VITS model loaded ✅
- `xtts_available`: XTTS model loaded ✅
- `xtts_device`: Using GPU ("CUDA") or CPU
- `tm_clip_found`: Default reference clip exists in `data/TM_Clip/`

***

## **Features**

✅ **Three Synthesis Modes:**
- VITS: Baseline pre-trained English voice
- XTTS (TM): Clone a default narrator (e.g., from `data/TM_Clip/clip_001.wav`)
- Custom: Upload your own reference → instant voice clone

✅ **Smart Text Processing:**
- Auto-chunks long passages (1000-char segments) to prevent timeouts
- Sentence/paragraph-aware splitting for natural pacing
- Merges chunks into a single WAV via pydub + FFmpeg

✅ **Robust File Handling:**
- Accepts 9 audio formats; standardizes to 22,050 Hz WAV for XTTS
- Auto-cleanup of old outputs and temp uploads

***

## **The Journey: From Training to Zero-Shot**

### **Original Goal**
Train a **custom voice model** (VITS or Tacotron2) fine-tuned on a specific narrator's audiobook dataset (~10 hours, 4,115 clips) to achieve studio-quality, personalized narration.

### **What We Tried**

#### **Phase 1: VITS Training from Scratch (Unintended)**
- **Approach:** Configured VITS training script; ran 18+ hours (~50k steps, ~160 epochs)
- **Outcome:** No intelligible speech—only screeching/noise. Loss decreased but audio remained unintelligible.
- **Root Cause:** Code initialized a fresh model (`Vits.init_from_config`) instead of loading pre-trained weights. VITS requires weeks of training on large datasets (100s of hours) to converge from scratch.
- **Obstacles:**
  - 4 GB VRAM (RTX 3050 Ti) → OOM errors; fixed with tiny batches and `PYTORCH_CUDA_ALLOC_CONF`
  - Windows multiprocessing errors → resolved with `if __name__ == "__main__"`
  - File lock issues (WinError 32) during checkpoint cleanup

#### **Phase 2: VITS Fine-Tuning (Attempted Transfer Learning)**
- **Approach:** Loaded LJSpeech pre-trained VITS checkpoint; resumed training with `load_state_dict(strict=False)`
- **Outcome:** Audio remained screeching/garbled even after +1000 fine-tuning steps
- **Root Cause:** Mismatch between pre-trained model's tokenizer/text cleaners and our dataset format. Partial weight loading left critical layers (encoder/decoder) randomly initialized.
- **Decision:** High risk of sinking weeks without convergence; pivoted to zero-shot.

#### **Phase 3: Tacotron2 Experiment**
- **Approach:** Train simpler attention-based model (Tacotron2) from scratch on full dataset
- **Outcome:** At ~1,100 steps, decoder hit max step limit (10,000); outputs were screeching or ultra-short clips
- **Root Cause:** Alignment failure—Tacotron2's attention mechanism never learned to map text to audio frames. Classic symptom of insufficient batch size, dataset noise, or missing phoneme preprocessing.
- **Decision:** Evidence pointed to low success probability on 4 GB VRAM + heterogeneous audiobook data.

### **Breakthrough: XTTS Zero-Shot**
- **Discovery:** XTTS v2 (pre-trained multilingual model) already produced **studio-quality clones** from a single 10–30s reference clip—**no training required**.
- **Integration:** Added XTTS to Flask app with chunking, progress logs, and multi-format upload support.
- **Result:** Production-ready in days vs. weeks of uncertain training. High quality, fast synthesis, minimal user dependencies.

### **Why Training Failed (Core Reasons)**
1. **Compute Limits:** 4 GB VRAM restricts effective batch sizes; modern TTS models (VITS, Tacotron2) need 16–40 GB for stable alignment and convergence.
2. **Data Engineering Gap:** Audiobook narration has variable prosody. Training requires:
   - Studio-grade recordings with flawless transcriptions
   - Strict preprocessing (silence trimming, loudness normalization, phonemization)
   - Controlled phoneme/word coverage
3. **Library Sensitivity:** Version mismatches (deprecated PyTorch autocast, transformers API changes), Windows file locks, and tokenizer/config frictions compounded debugging costs.
4. **Alignment Complexity:** Tacotron2 requires early attention alignment. When stop-prediction and attention fail (common with long/noisy clips and small batches), the decoder runs to max steps and produces garbage.

### **What is the professional approach**
- **Data:** Studio recordings; forced phoneme alignment; loudness/silence normalization at scale; transcript QA
- **Training:** Multi-GPU (A100 class, 40–80 GB VRAM); pretrained checkpoints with matched cleaners/tokenizers; layer freezing + gradual unfreezing; curriculum learning (short→long); weeks of continuous runs
- **Infrastructure:** MLOps pipelines, automated eval, alignment diagnostics, auto-stop on failure
- **Economics:** Often choose zero-shot/commercial APIs unless custom differentiation justifies training cost

***

## **Technology Stack**

### **Core**
- **Python 3.11** – Primary language
- **Flask** – Web server (routes, API, file serving)
- **Coqui TTS** – XTTS/VITS model loader and synthesis API
- **PyTorch + CUDA** – GPU runtime for model inference
- **pydub + FFmpeg** – Audio chunk merging and format conversion
- **pathlib/os/glob** – Cross-platform file operations

### **Frontend**
- **HTML/CSS/JavaScript** – Responsive UI with circular voice buttons, upload status banners, and audio player

### **Models**
- **XTTS v2** (`tts_models/multilingual/multi-dataset/xtts_v2`) – Zero-shot voice cloning (primary)
- **VITS** (`tts_models/en/ljspeech/vits`) – Baseline demo voice


***

## **Usage**

### **Quick Start**
1. **Select a voice:**
   - **VITS** – Demo voice (no setup)
   - **XTTS (TM)** – Uses default narrator (requires `data/TM_Clip/clip_001.wav`)
   - **Custom** – Upload your own 10–30s sample (any format)

2. **Enter text** in the text box (supports full chapters)

3. **Click Convert** → Wait for synthesis (progress in console)

4. **Play audio** from the embedded player or download the WAV

### **Custom Upload Workflow**
1. Click **"Custom"** button
2. **Browse** → select a clean 10–30s voice sample (MP3/WAV/M4A/etc.)
3. Wait for **green "Upload successful!"** banner
4. Enter text and click **Convert**
5. Audio generated in uploaded voice; original clip auto-deleted after synthesis

### **Health Check**
Visit [http://localhost:5000/health](http://localhost:5000/health) to see:
```json
{
  "vits_available": true,
  "xtts_available": true,
  "xtts_device": "CUDA",
  "tm_clip_found": true
}
```
**Note:** Models (XTTS/VITS) are **not** in the repo. They download automatically on first run and cache under `%USERPROFILE%\.local\share\tts\` (Windows) or `~/.local/share/tts/` (Linux/macOS).

***

## **Future Vision: Emotion-Aware Narration**

### **The Next Leap: Sentiment-Driven Multi-Tone Synthesis**

#### **Concept**
Use **sentiment analysis** (NLP) to detect emotion in text chunks (angry, sad, neutral, happy, excited), then synthesize each chunk with a corresponding **pre-recorded tone sample** for richer, more expressive narration.

#### **How It Would Work**
1. **Text Preprocessing:**
   - Split long text into sentences/paragraphs
   - Run sentiment classifier (e.g., BERT-based emotion detection or simpler VADER)
   - Tag each chunk: `angry`, `sad`, `neutral`, `happy`, etc.

2. **Tone Library:**
   - Record 5–10 short (10–20s) clips of the narrator speaking in different emotional tones:
     - `angry_tone.wav` – Intense, raised volume
     - `sad_tone.wav` – Slower, softer, melancholic
     - `neutral_tone.wav` – Calm, conversational
     - `happy_tone.wav` – Upbeat, energetic
     - `excited_tone.wav` – Fast-paced, dynamic

3. **Emotion-Matched Synthesis:**
   - For each chunk tagged as "angry" → use `angry_tone.wav` as XTTS `speaker_wav`
   - For "sad" → use `sad_tone.wav`
   - Merge all generated chunks into final audio

4. **Result:**
   - Narration automatically adapts prosody/energy to the story's emotional arc
   - Far more **expressive** than monotone synthesis
   - No manual intervention—**fully automated** after initial tone library creation

***

## **Research & Lessons Learned**

### **Key Takeaways**
1. **Zero-shot models (XTTS) win for voice cloning** – Training custom models is only justified when differentiation requires it (unique architecture, domain-specific data, or latency constraints).
2. **4 GB VRAM is insufficient** for modern TTS training (VITS, Tacotron2, FastSpeech). 16–40 GB recommended.
3. **Data engineering >> model tuning** – Clean transcripts, silence trimming, loudness normalization, and phoneme alignment are 80% of success.
4. **Alignment is critical** – Tacotron2/VITS fail silently when attention doesn't form early; stop training if no alignment by ~2k steps.

### **What Worked**
✅ XTTS zero-shot voice cloning  
✅ Flask + chunking + pydub for long-text synthesis  
✅ Multi-format upload with auto-WAV conversion  
✅ CUDA → CPU fallback for portability  

### **What Didn't Work (and Why)**
❌ **VITS training from scratch** – Needs weeks + 100s of hours of data  
❌ **VITS fine-tuning with mismatched tokenizer** – Partial load left layers random  
❌ **Tacotron2 on 4 GB VRAM** – Alignment failure due to small batches + dataset noise  
