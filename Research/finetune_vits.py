'''
finetune_vits.py
Purpose:
- Fine-tune a pre-trained VITS (LJSpeech) model on a custom single-speaker dataset.
- Uses a custom formatter for two-column metadata lines: "filename.wav|text".
- Loads the pre-trained checkpoint via TrainerArgs(restore_path=...) and continues training.

Warnings:
- This script is experimental and resource intensive; quality is not guaranteed on small VRAM.
- Requires CUDA-enabled PyTorch and coqui-ai/TTS compatible versions.
- Expects dataset files under data/model_input/ with a metadata file named Dataset_clips.txt.
- Outputs checkpoints and logs under data/model_output/finetune-run-YYYYMMDD_HHMMSS.
- Ensure sample_rate and text processing config match the pre-trained model to reduce instability.
'''

import os
from datetime import datetime
from pathlib import Path

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig, VitsAudioConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.manage import ModelManager


def custom_formatter(root_path, meta_file, **kwargs):
    """
    Custom formatter for lines formatted as: 'filename.wav|text'
    Resolves audio files under root_path and builds items usable by the TTS loader.
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "narrator"

    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")
            if len(parts) < 2:
                continue

            wav_file = os.path.join(root_path, parts[0].strip())
            text = parts[1].strip()

            if not os.path.exists(wav_file):
                continue

            items.append({
                "text": text,
                "audio_file": wav_file,
                "speaker_name": speaker_name,
                "root_path": root_path
            })

    return items


if __name__ == "__main__":
    # CUDA memory config
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    # ---------------------------
    # Relative paths (portable)
    # ---------------------------
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    CLIPS_PATH = DATA_DIR / "model_input"          # folder containing audio clips and Dataset_clips.txt
    BASE_OUTPUT = DATA_DIR / "model_output"        # folder to store finetune runs
    META_FILE = "Dataset_clips.txt"

    RUN_OUTPUT = BASE_OUTPUT / ("finetune-run-" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    RUN_OUTPUT.mkdir(parents=True, exist_ok=True)

    # 🔥 DOWNLOAD PRE-TRAINED MODEL
    print("📥 Downloading pre-trained VITS model (LJSpeech)...")
    manager = ModelManager()
    model_path, config_path, _ = manager.download_model("tts_models/en/ljspeech/vits")
    print(f"✅ Model downloaded to: {model_path}")

    # Dataset config
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",          # using custom formatter below
        meta_file_train=META_FILE,
        path=str(CLIPS_PATH),
    )

    # Characters config (as in your original code)
    characters = CharactersConfig(
        pad="_",
        eos="~",
        bos="^",
        blank="<BLNK>",
        characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        punctuations=" !\"'(),-./:;?[]",
        phonemes=""
    )

    # Main VITS config (kept same as your code)
    config = VitsConfig(
        text_cleaner=None,
        use_phonemes=False,

        # Small batch for stability on low VRAM
        batch_size=2,
        eval_batch_size=2,

        mixed_precision=True,
        num_loader_workers=0,
        num_eval_loader_workers=0,

        # Fine-tuning settings
        epochs=20,
        run_eval=True,
        test_delay_epochs=3,

        # Audio must match pre-trained model
        audio=VitsAudioConfig(
            sample_rate=22050,
            win_length=1024,
            hop_length=256,
            num_mels=80,
        ),

        output_path=str(RUN_OUTPUT),
        save_step=250,
        print_step=10,

        datasets=[dataset_config],
        characters=characters,

        # 🔥 Critical: lower LR for fine-tuning
        lr=0.00005,
    )

    # Build processor and tokenizer
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    vocab_size = len(tokenizer.characters.vocab)
    print(f"✓ Vocabulary size: {vocab_size}")

    # Load samples using custom formatter
    print("\n📂 Loading dataset...")
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        formatter=custom_formatter
    )

    print(f"✓ Training samples: {len(train_samples)}")
    print(f"✓ Validation samples: {len(eval_samples)}")

    print("\n🏗️ Building model...")
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    print(f"✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # 🔥 Fine-tune from pre-trained weights via Trainer restore_path
    print("\n🚀 Starting fine-tuning from pre-trained checkpoint...")
    print(f"📦 Pre-trained model: {model_path}\n")

    trainer = Trainer(
        TrainerArgs(restore_path=model_path),
        config,
        str(RUN_OUTPUT),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples
    )

    trainer.fit()
