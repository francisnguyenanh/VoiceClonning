"""
tts_engine.py — Singleton wrapper around Coqui TTS XTTS v2.

XTTS v2 is a zero-shot multilingual TTS model. It reads a reference WAV
to extract the speaker's voice characteristics and generates speech in
the target language using those characteristics — no fine-tuning needed.

Supported languages: en, vi, ja, zh-cn (and 13 more).
Model downloads (~2 GB) automatically on first use to ~/.local/share/tts/.
"""
from __future__ import annotations

import logging
import re
import threading
import uuid
from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf

from config import (
    OUTPUTS_DIR,
    VOICES_DIR,
    TTS_MODEL_NAME,
    TTS_CHUNK_SIZE,
    REFERENCE_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_instance: TTSEngine | None = None


def get_tts_engine() -> "TTSEngine":
    """Return the shared TTSEngine singleton, creating it if necessary."""
    global _instance
    if _instance is None:
        with _LOCK:
            if _instance is None:
                _instance = TTSEngine()
    return _instance


class TTSEngine:
    """Thin wrapper around Coqui TTS for XTTS v2 inference."""

    def __init__(self) -> None:
        self._tts = None
        self._loaded = False
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the XTTS v2 model (blocking, ~10-30 s on first cold start)."""
        with self._load_lock:
            if self._loaded:
                return
            import os
            os.environ["COQUI_TOS_AGREED"] = "1"  # agree to non-commercial CPML
            logger.info("Loading TTS model: %s", TTS_MODEL_NAME)
            from TTS.api import TTS  # imported here to avoid slow startup
            self._tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=True, gpu=False)
            self._loaded = True
            logger.info("TTS model ready.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(
        self,
        text: str,
        profile_id: str,
        language: str = "vi",
        session_id: str | None = None,
    ) -> Path:
        """
        Generate speech for *text* using the voice profile *profile_id*.

        Returns the path to the output WAV file.
        """
        self._ensure_loaded()

        reference_wav = VOICES_DIR / profile_id / "reference.wav"
        if not reference_wav.exists():
            raise FileNotFoundError(f"Voice profile '{profile_id}' has no reference.wav")

        out_dir = OUTPUTS_DIR / (session_id or str(uuid.uuid4()))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "output.wav"

        lang = self._normalise_lang(language)

        # Split long text into chunks to prevent OOM on CPU
        chunks = self._split_text(text, TTS_CHUNK_SIZE)
        logger.info(
            "Generating TTS: profile=%s lang=%s chunks=%d", profile_id, lang, len(chunks)
        )

        if len(chunks) == 1:
            self._tts.tts_to_file(
                text=chunks[0],
                speaker_wav=str(reference_wav),
                language=lang,
                file_path=str(out_path),
            )
        else:
            segments: list[np.ndarray] = []
            sr = REFERENCE_SAMPLE_RATE
            for i, chunk in enumerate(chunks):
                logger.info("  chunk %d/%d", i + 1, len(chunks))
                wav = self._tts.tts(
                    text=chunk,
                    speaker_wav=str(reference_wav),
                    language=lang,
                )
                segments.append(np.array(wav, dtype=np.float32))

            # 0.3 s silence between chunks
            silence = np.zeros(int(sr * 0.3), dtype=np.float32)
            combined = np.concatenate(
                [seg for pair in zip(segments, [silence] * len(segments)) for seg in pair]
            )
            sf.write(str(out_path), combined, sr, subtype="PCM_16")

        logger.info("Output saved: %s", out_path)
        return out_path

    def generate_streaming(
        self,
        text: str,
        profile_id: str,
        language: str = "vi",
    ) -> Generator[bytes, None, None]:
        """
        Yield raw WAV bytes chunk-by-chunk using XTTS streaming API.
        Suitable for Server-Sent Events or chunked HTTP responses.
        """
        self._ensure_loaded()

        reference_wav = VOICES_DIR / profile_id / "reference.wav"
        if not reference_wav.exists():
            raise FileNotFoundError(f"Voice profile '{profile_id}' not found")

        lang = self._normalise_lang(language)
        chunks = self._split_text(text, TTS_CHUNK_SIZE)

        import io
        import wave

        for chunk in chunks:
            wav_list = self._tts.tts(
                text=chunk,
                speaker_wav=str(reference_wav),
                language=lang,
            )
            audio = np.array(wav_list, dtype=np.float32)
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(REFERENCE_SAMPLE_RATE)
                wf.writeframes(pcm)
            yield buf.getvalue()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    @staticmethod
    def _normalise_lang(lang: str) -> str:
        mapping = {"zh": "zh-cn", "zh_cn": "zh-cn", "zh_tw": "zh-cn"}
        return mapping.get(lang, lang)

    @staticmethod
    def _split_text(text: str, max_chars: int) -> list[str]:
        """
        Split text into chunks of at most max_chars, preferring sentence
        boundaries. Works for CJK (uses 。！？) and Latin punctuation.
        """
        # Sentence-ending characters (Latin + CJK)
        sentence_end = re.compile(r"(?<=[.!?。！？])\s*")
        sentences = sentence_end.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= max_chars:
                current = (current + " " + sent).strip() if current else sent
            else:
                if current:
                    chunks.append(current)
                # If a single sentence exceeds max_chars, hard-split it
                while len(sent) > max_chars:
                    chunks.append(sent[:max_chars])
                    sent = sent[max_chars:]
                current = sent
        if current:
            chunks.append(current)

        return chunks or [text]
