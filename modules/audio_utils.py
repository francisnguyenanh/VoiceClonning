"""
audio_utils.py — Reference audio preprocessing.

Steps:
  1. Convert any format to mono WAV at REFERENCE_SAMPLE_RATE
  2. Noise reduction (noisereduce)
  3. Normalize amplitude to -3 dBFS
  4. Trim leading/trailing silence
"""
from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import noisereduce as nr
import soundfile as sf

from config import REFERENCE_SAMPLE_RATE

logger = logging.getLogger(__name__)


def _normalize(audio: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    """Peak-normalize audio to target_dbfs."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    target_amp = 10 ** (target_dbfs / 20.0)
    return audio * (target_amp / peak)


def preprocess_reference(input_path: str | Path, output_path: str | Path) -> Path:
    """
    Load, clean and save a reference audio file for voice cloning.

    Args:
        input_path: Path to the uploaded audio file (any format librosa supports).
        output_path: Destination path for the clean WAV file.

    Returns:
        Path to the saved output WAV file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Preprocessing reference audio: %s", input_path)

    # 1. Load & resample to mono at target sample rate
    audio, sr = librosa.load(str(input_path), sr=REFERENCE_SAMPLE_RATE, mono=True)

    # 2. Noise reduction — use first 0.5 s as noise profile if long enough
    profile_samples = int(0.5 * sr)
    noise_clip = audio[:profile_samples] if len(audio) > profile_samples else audio
    audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip, stationary=False)

    # 3. Normalize
    audio = _normalize(audio)

    # 4. Trim silence
    audio, _ = librosa.effects.trim(audio, top_db=30)

    # 5. Save
    sf.write(str(output_path), audio, sr, subtype="PCM_16")
    logger.info("Saved preprocessed reference: %s (%.1f s)", output_path, len(audio) / sr)

    return output_path


def float32_to_int16(audio: np.ndarray) -> bytes:
    """Convert float32 numpy array [-1, 1] to int16 bytes for streaming."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def int16_bytes_to_float32(data: bytes) -> np.ndarray:
    """Convert raw int16 bytes from browser to float32 numpy array."""
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return arr / 32768.0
