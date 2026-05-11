"""
app.py — Flask application entry point.

Routes:
  /                      → Dashboard (index)
  /profiles              → Voice profile management
  /tts                   → TTS from text / document
  /realtime              → Real-time voice conversion

  API:
  POST /api/profiles               → Upload reference audio, create profile
  GET  /api/profiles               → List all profiles
  DELETE /api/profiles/<id>        → Delete profile
  GET  /api/profiles/<id>/audio    → Stream reference audio
  POST /api/tts/from-text          → Generate TTS from text input
  GET  /api/tts/stream/<session>   → Stream TTS audio (chunked)
  POST /api/tts/from-file          → Upload document, extract + generate TTS
  GET  /api/tts/download/<session> → Download output WAV

SocketIO events:
  audio_chunk  (client → server)  → real-time audio bytes for VC
  converted    (server → client)  → converted audio bytes
  vc_status    (server → client)  → status messages
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

import config
from config import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_DOC_EXTENSIONS,
    MAX_AUDIO_UPLOAD_MB,
    MAX_DOC_UPLOAD_MB,
    OUTPUTS_DIR,
    UPLOADS_DIR,
    VOICES_DIR,
    SUPPORTED_LANGUAGES,
)
from modules.audio_utils import preprocess_reference
from modules.document_parser import parse_document
from modules.tts_engine import get_tts_engine
from modules.voice_converter import get_voice_converter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = max(MAX_AUDIO_UPLOAD_MB, MAX_DOC_UPLOAD_MB) * 1024 * 1024

socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*", max_http_buffer_size=2 * 1024 * 1024)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _allowed_file(filename: str, allowed: set[str]) -> bool:
    return Path(filename).suffix.lower() in allowed


def _load_profile_meta(profile_id: str) -> dict | None:
    meta_path = VOICES_DIR / profile_id / "metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _all_profiles() -> list[dict]:
    profiles = []
    for d in sorted(VOICES_DIR.iterdir()):
        if d.is_dir():
            meta = _load_profile_meta(d.name)
            if meta:
                profiles.append(meta)
    return profiles


def _cleanup_old_outputs(max_age_s: int = 3600) -> None:
    """Delete output directories older than max_age_s seconds."""
    now = time.time()
    for d in OUTPUTS_DIR.iterdir():
        if d.is_dir() and (now - d.stat().st_mtime) > max_age_s:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    profiles = _all_profiles()
    tts_ready = get_tts_engine().is_loaded
    return render_template("index.html", profiles=profiles, tts_ready=tts_ready)


@app.route("/profiles")
def profiles_page():
    profiles = _all_profiles()
    return render_template("profiles.html", profiles=profiles, languages=SUPPORTED_LANGUAGES)


@app.route("/tts")
def tts_page():
    profiles = _all_profiles()
    return render_template("tts.html", profiles=profiles, languages=SUPPORTED_LANGUAGES)


@app.route("/realtime")
def realtime_page():
    profiles = _all_profiles()
    return render_template("realtime.html", profiles=profiles)


# ---------------------------------------------------------------------------
# API — Voice Profiles
# ---------------------------------------------------------------------------

@app.route("/api/profiles", methods=["GET"])
def api_list_profiles():
    return jsonify(_all_profiles())


@app.route("/api/profiles", methods=["POST"])
def api_create_profile():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    name = request.form.get("name", "").strip()
    language = request.form.get("language", "vi")

    if not name:
        return jsonify({"error": "Profile name is required"}), 400
    if not _allowed_file(audio_file.filename, ALLOWED_AUDIO_EXTENSIONS):
        return jsonify({"error": "Unsupported audio format"}), 400

    profile_id = str(uuid.uuid4())
    profile_dir = VOICES_DIR / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Save raw upload
    raw_filename = secure_filename(audio_file.filename)
    raw_path = UPLOADS_DIR / f"ref_{profile_id}_{raw_filename}"
    audio_file.save(str(raw_path))

    try:
        ref_path = profile_dir / "reference.wav"
        preprocess_reference(raw_path, ref_path)
    except Exception as exc:
        shutil.rmtree(profile_dir, ignore_errors=True)
        raw_path.unlink(missing_ok=True)
        logger.exception("Failed to preprocess reference audio")
        return jsonify({"error": f"Audio preprocessing failed: {exc}"}), 500
    finally:
        raw_path.unlink(missing_ok=True)

    meta = {
        "id": profile_id,
        "name": name,
        "language": language,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(profile_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Created voice profile: %s (%s)", name, profile_id)
    return jsonify(meta), 201


@app.route("/api/profiles/<profile_id>", methods=["DELETE"])
def api_delete_profile(profile_id: str):
    profile_dir = VOICES_DIR / profile_id
    if not profile_dir.exists():
        return jsonify({"error": "Profile not found"}), 404
    shutil.rmtree(profile_dir)
    return jsonify({"message": "Deleted"}), 200


@app.route("/api/profiles/<profile_id>/audio")
def api_profile_audio(profile_id: str):
    ref_path = VOICES_DIR / profile_id / "reference.wav"
    if not ref_path.exists():
        return jsonify({"error": "Audio not found"}), 404
    return send_file(str(ref_path), mimetype="audio/wav")


# ---------------------------------------------------------------------------
# API — TTS
# ---------------------------------------------------------------------------

@app.route("/api/tts/from-text", methods=["POST"])
def api_tts_from_text():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    profile_id = data.get("profile_id", "")
    language = data.get("language", "vi")

    if not text:
        return jsonify({"error": "Text is required"}), 400
    if not profile_id or not (VOICES_DIR / profile_id / "reference.wav").exists():
        return jsonify({"error": "Invalid voice profile"}), 400

    session_id = str(uuid.uuid4())
    engine = get_tts_engine()

    try:
        out_path = engine.generate(text, profile_id, language, session_id)
    except Exception as exc:
        logger.exception("TTS generation failed")
        return jsonify({"error": str(exc)}), 500

    _cleanup_old_outputs()
    return jsonify({"session_id": session_id, "url": f"/api/tts/download/{session_id}"})


@app.route("/api/tts/stream/<session_id>")
def api_tts_stream(session_id: str):
    """Server-Sent Events stream: emit progress events during TTS generation."""
    profile_id = request.args.get("profile_id", "")
    language = request.args.get("language", "vi")
    text = request.args.get("text", "")

    if not all([text, profile_id]):
        return jsonify({"error": "Missing parameters"}), 400

    engine = get_tts_engine()

    def generate():
        try:
            yield "event: start\ndata: {}\n\n"
            for chunk_bytes in engine.generate_streaming(text, profile_id, language):
                import base64
                b64 = base64.b64encode(chunk_bytes).decode()
                yield f"event: chunk\ndata: {json.dumps({'audio': b64})}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/tts/download/<session_id>")
def api_tts_download(session_id: str):
    out_path = OUTPUTS_DIR / session_id / "output.wav"
    if not out_path.exists():
        return jsonify({"error": "Audio not found"}), 404
    return send_file(str(out_path), mimetype="audio/wav", as_attachment=True, download_name="output.wav")


@app.route("/api/tts/from-file", methods=["POST"])
def api_tts_from_file():
    if "document" not in request.files:
        return jsonify({"error": "No document provided"}), 400

    doc_file = request.files["document"]
    profile_id = request.form.get("profile_id", "")
    language = request.form.get("language", "vi")

    if not _allowed_file(doc_file.filename, ALLOWED_DOC_EXTENSIONS):
        return jsonify({"error": "Unsupported document format"}), 400
    if not profile_id or not (VOICES_DIR / profile_id / "reference.wav").exists():
        return jsonify({"error": "Invalid voice profile"}), 400

    filename = secure_filename(doc_file.filename)
    upload_path = UPLOADS_DIR / f"{uuid.uuid4()}_{filename}"
    doc_file.save(str(upload_path))

    try:
        text = parse_document(upload_path)
    except Exception as exc:
        logger.exception("Document parsing failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        upload_path.unlink(missing_ok=True)

    if not text.strip():
        return jsonify({"error": "No text found in document"}), 400

    # Return extracted text first so user can preview / edit
    session_id = str(uuid.uuid4())
    return jsonify({
        "session_id": session_id,
        "text": text,
        "char_count": len(text),
    })


@app.route("/api/tts/generate-from-session", methods=["POST"])
def api_generate_from_session():
    """Generate TTS after user reviews extracted document text."""
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    profile_id = data.get("profile_id", "")
    language = data.get("language", "vi")
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not text:
        return jsonify({"error": "Text is required"}), 400
    if not profile_id or not (VOICES_DIR / profile_id / "reference.wav").exists():
        return jsonify({"error": "Invalid voice profile"}), 400

    engine = get_tts_engine()
    try:
        out_path = engine.generate(text, profile_id, language, session_id)
    except Exception as exc:
        logger.exception("TTS generation failed")
        return jsonify({"error": str(exc)}), 500

    _cleanup_old_outputs()
    return jsonify({"session_id": session_id, "url": f"/api/tts/download/{session_id}"})


# ---------------------------------------------------------------------------
# SocketIO — Real-time Voice Conversion
# ---------------------------------------------------------------------------

# Per-sid buffer to accumulate audio until we have enough for FreeVC
_buffers: dict[str, bytes] = {}
_buffer_lock = threading.Lock()

BUFFER_THRESHOLD_BYTES = int(config.REALTIME_SAMPLE_RATE * 1.5 * 2)  # 1.5 s × 2 bytes/sample


@socketio.on("connect")
def on_connect():
    sid = request.sid
    with _buffer_lock:
        _buffers[sid] = b""
    emit("vc_status", {"status": "connected"})
    logger.info("Client connected: %s", sid)


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    with _buffer_lock:
        _buffers.pop(sid, None)
    logger.info("Client disconnected: %s", sid)


@socketio.on("audio_chunk")
def on_audio_chunk(data):
    """
    Receive raw int16 PCM audio bytes from the browser.
    data = {"audio": <bytes>, "profile_id": <str>}
    """
    sid = request.sid
    profile_id = data.get("profile_id", "")
    audio_bytes = data.get("audio", b"")

    if not profile_id or not audio_bytes:
        return

    with _buffer_lock:
        _buffers[sid] = _buffers.get(sid, b"") + audio_bytes
        if len(_buffers[sid]) < BUFFER_THRESHOLD_BYTES:
            return
        chunk = _buffers[sid]
        _buffers[sid] = b""

    # Convert in a background thread so we don't block the SocketIO event loop
    def _convert():
        try:
            converter = get_voice_converter()
            converted = converter.convert_chunk(chunk, profile_id)
            socketio.emit("converted", {"audio": converted}, to=sid)
        except Exception as exc:
            logger.exception("Voice conversion failed")
            socketio.emit("vc_status", {"status": "error", "message": str(exc)}, to=sid)

    threading.Thread(target=_convert, daemon=True).start()


@socketio.on("load_models")
def on_load_models():
    """Client requests model loading (called on realtime page open)."""
    emit("vc_status", {"status": "loading"})

    def _load():
        try:
            get_tts_engine().load()
            get_voice_converter().load()
            socketio.emit("vc_status", {"status": "ready"}, to=request.sid)
        except Exception as exc:
            socketio.emit("vc_status", {"status": "error", "message": str(exc)}, to=request.sid)

    threading.Thread(target=_load, daemon=True).start()


# ---------------------------------------------------------------------------
# Pre-load TTS model on startup (background thread)
# ---------------------------------------------------------------------------

def _preload_models():
    logger.info("Pre-loading TTS model in background...")
    try:
        get_tts_engine().load()
        logger.info("TTS model pre-loaded.")
    except Exception:
        logger.exception("Model pre-load failed — will retry on first request")


threading.Thread(target=_preload_models, daemon=True).start()


if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5050, debug=False)
