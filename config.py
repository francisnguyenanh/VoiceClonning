import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

VOICES_DIR = BASE_DIR / "voices"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

for d in (VOICES_DIR, UPLOADS_DIR, OUTPUTS_DIR):
    d.mkdir(exist_ok=True)

# XTTS v2 model identifier (auto-downloaded to ~/.local/share/tts/ on first run)
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Real-time voice conversion model (FreeVC via Coqui TTS)
VC_MODEL_NAME = "voice_conversion_models/multilingual/vctk/freevc24"

# Audio settings for real-time pipeline
REALTIME_SAMPLE_RATE = 16000       # Hz — mic input (webrtcvad requires 8/16/32 kHz)
REALTIME_CHUNK_DURATION_MS = 480   # ms per chunk sent from browser
REALTIME_FRAME_MS = 30             # ms per VAD frame (10/20/30 ms only)
VAD_AGGRESSIVENESS = 2             # 0-3 (3 = most aggressive filtering)

# Reference audio for voice profiles (pre-processed)
REFERENCE_SAMPLE_RATE = 22050

# TTS chunk size (characters) — split long text to avoid OOM on CPU
TTS_CHUNK_SIZE = 200

# Supported document MIME / extensions
ALLOWED_DOC_EXTENSIONS = {".txt", ".docx", ".pptx", ".xlsx", ".pdf"}

# Allowed audio upload extensions for voice profiles
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}

# Max upload sizes
MAX_AUDIO_UPLOAD_MB = 50
MAX_DOC_UPLOAD_MB = 25

# Map language display names → XTTS language codes
LANGUAGE_MAP = {
    "vi": "vi",
    "en": "en",
    "ja": "ja",
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
}

SUPPORTED_LANGUAGES = [
    {"code": "vi",    "label": "Tiếng Việt"},
    {"code": "en",    "label": "English"},
    {"code": "ja",    "label": "日本語"},
    {"code": "zh-cn", "label": "中文"},
]

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")
