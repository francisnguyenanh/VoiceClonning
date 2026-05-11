"""
voice_converter.py — Real-time voice conversion using FreeVC (via Coqui TTS).

Pipeline (per audio chunk from browser):
  browser mic (WebAudio API, PCM int16)
    → webrtcvad VAD (skip silence)
    → FreeVC convert (source → target speaker)
    → PCM int16 bytes
    → SocketIO back to browser

FreeVC on CPU has ~0.5-2 s latency. We use a ring-buffer to accumulate
enough audio for FreeVC context (~1-2 s), then convert and stream output.
"""
from __future__ import annotations

import logging
import threading
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from config import (
    VOICES_DIR,
    VC_MODEL_NAME,
    REALTIME_SAMPLE_RATE,
)
from modules.audio_utils import int16_bytes_to_float32, float32_to_int16

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_instance: Optional[VoiceConverter] = None


def get_voice_converter() -> "VoiceConverter":
    global _instance
    if _instance is None:
        with _LOCK:
            if _instance is None:
                _instance = VoiceConverter()
    return _instance


class VoiceConverter:
    """Thin wrapper around Coqui TTS FreeVC24 for voice conversion."""

    # Accumulate ~1.5 s of audio before converting (FreeVC needs context)
    BUFFER_DURATION_S = 1.5

    def __init__(self) -> None:
        self._tts = None
        self._loaded = False
        self._load_lock = threading.Lock()

    def load(self) -> None:
        with self._load_lock:
            if self._loaded:
                return
            import os
            os.environ["COQUI_TOS_AGREED"] = "1"  # agree to non-commercial CPML
            logger.info("Loading VC model: %s", VC_MODEL_NAME)
            from TTS.api import TTS
            self._tts = TTS(model_name=VC_MODEL_NAME, progress_bar=True, gpu=False)
            self._loaded = True
            logger.info("VC model ready.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def convert_chunk(
        self,
        audio_bytes: bytes,
        profile_id: str,
        sample_rate: int = REALTIME_SAMPLE_RATE,
    ) -> bytes:
        """
        Convert audio_bytes (raw int16 PCM) to the voice of profile_id.

        Returns raw int16 PCM bytes at sample_rate.
        """
        if not self._loaded:
            self.load()

        target_wav = VOICES_DIR / profile_id / "reference.wav"
        if not target_wav.exists():
            raise FileNotFoundError(f"Voice profile '{profile_id}' not found")

        # Decode incoming bytes to float32
        source_audio = int16_bytes_to_float32(audio_bytes)

        # Write source audio to a temp file (FreeVC expects file paths)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src_tmp:
            sf.write(src_tmp.name, source_audio, sample_rate, subtype="PCM_16")
            src_path = src_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_tmp:
            out_path = out_tmp.name

        try:
            self._tts.voice_conversion_to_file(
                source_wav=src_path,
                target_wav=str(target_wav),
                file_path=out_path,
            )
            converted, _ = sf.read(out_path, dtype="float32")
        finally:
            Path(src_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)

        return float32_to_int16(converted)
