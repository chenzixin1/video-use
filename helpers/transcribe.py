"""Transcribe a video with a selectable ASR provider.

Default provider: ElevenLabs Scribe.
Optional provider: Volcengine / ByteDance BigModel ASR standard HTTP.

Extracts mono 16kHz audio via ffmpeg, calls the provider, normalizes the
response to the Scribe-like word list used downstream, and writes the full
response to <edit_dir>/transcripts/<video_stem>.json.

Cached: if the output file already exists, the upload is skipped.

Usage:
    python helpers/transcribe.py <video_path>
    python helpers/transcribe.py <video_path> --provider volcengine
    python helpers/transcribe.py <video_path> --edit-dir /custom/edit
    python helpers/transcribe.py <video_path> --language en
    python helpers/transcribe.py <video_path> --num-speakers 2
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import requests


SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"
VOLC_STANDARD_SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
VOLC_STANDARD_QUERY_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
VOLC_STANDARD_RESOURCE_ID = "volc.bigasr.auc"

Provider = str


def _load_dotenv_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for candidate in [Path(__file__).resolve().parent.parent / ".env", Path(".env")]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def _config_value(values: dict[str, str], *names: str) -> str:
    for name in names:
        v = os.environ.get(name) or values.get(name)
        if v:
            return v
    return ""


def load_api_key() -> str:
    values = _load_dotenv_values()
    v = _config_value(values, "ELEVENLABS_API_KEY")
    if not v:
        sys.exit("ELEVENLABS_API_KEY not found in .env or environment")
    return v


def _load_config_py(path: str) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        return {}
    spec = importlib.util.spec_from_file_location("_volcengine_config", p)
    if not spec or not spec.loader:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    out: dict[str, str] = {}
    for name in [
        "ACCESS_KEY",
        "APP_KEY",
        "SUBMIT_URL",
        "QUERY_URL",
        "DEFAULT_LANGUAGE",
    ]:
        value = getattr(module, name, None)
        if value:
            out[name] = str(value)
    return out


def load_volcengine_config() -> dict[str, str]:
    values = _load_dotenv_values()
    config_py = _config_value(values, "VOLCENGINE_CONFIG_PY", "MODEL_SPEECH_CONFIG_PY")
    py_values = _load_config_py(config_py)

    def get(*names: str) -> str:
        return _config_value(values, *names) or next((py_values[n] for n in names if n in py_values), "")

    api_key = get("MODEL_SPEECH_API_KEY", "VOLCENGINE_API_KEY", "VOLCANO_ACCESS_KEY", "ACCESS_KEY")
    app_id = get("MODEL_SPEECH_APP_ID", "VOLCENGINE_APP_ID", "VOLCENGINE_APP_KEY", "APP_KEY")
    if not api_key:
        sys.exit("Volcengine API key not found. Set MODEL_SPEECH_API_KEY or VOLCENGINE_API_KEY.")
    return {
        "api_key": api_key,
        "app_id": app_id,
        "submit_url": get("MODEL_SPEECH_ASR_STANDARD_SUBMIT_URL", "VOLCENGINE_SUBMIT_URL", "SUBMIT_URL") or VOLC_STANDARD_SUBMIT_URL,
        "query_url": get("MODEL_SPEECH_ASR_STANDARD_QUERY_URL", "VOLCENGINE_QUERY_URL", "QUERY_URL") or VOLC_STANDARD_QUERY_URL,
        "resource_id": get("MODEL_SPEECH_ASR_STANDARD_RESOURCE_ID", "VOLCENGINE_RESOURCE_ID") or VOLC_STANDARD_RESOURCE_ID,
    }


def extract_audio(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def extract_audio_mp3(video_path: Path, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-codec:a", "libmp3lame", "-b:a", "64k",
        str(dest),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def call_scribe(
    audio_path: Path,
    api_key: str,
    language: str | None = None,
    num_speakers: int | None = None,
) -> dict:
    data: dict[str, str] = {
        "model_id": "scribe_v1",
        "diarize": "true",
        "tag_audio_events": "true",
        "timestamps_granularity": "word",
    }
    if language:
        data["language_code"] = language
    if num_speakers:
        data["num_speakers"] = str(num_speakers)

    with open(audio_path, "rb") as f:
        resp = requests.post(
            SCRIBE_URL,
            headers={"xi-api-key": api_key},
            files={"file": (audio_path.name, f, "audio/wav")},
            data=data,
            timeout=1800,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"Scribe returned {resp.status_code}: {resp.text[:500]}")

    return resp.json()


def _volcengine_headers(cfg: dict[str, str], request_id: str, sequence: str | None = None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "X-Api-Resource-Id": cfg["resource_id"],
        "X-Api-Request-Id": request_id,
    }
    if sequence is not None:
        headers["X-Api-Sequence"] = sequence
    if cfg.get("app_id"):
        headers["X-Api-App-Key"] = cfg["app_id"]
        headers["X-Api-Access-Key"] = cfg["api_key"]
    else:
        headers["X-Api-Key"] = cfg["api_key"]
    return headers


def _file_to_base64(audio_path: Path) -> str:
    return base64.b64encode(audio_path.read_bytes()).decode("utf-8")


def call_volcengine_standard(
    audio_path: Path,
    cfg: dict[str, str],
    language: str | None = None,
    poll_interval: float = 3.0,
    poll_max_time: float = 10800.0,
) -> dict:
    task_id = str(uuid.uuid4())
    audio_payload: dict[str, str] = {"data": _file_to_base64(audio_path)}
    if language:
        audio_payload["language"] = language

    payload = {
        "user": {"uid": cfg.get("app_id") or "video_use"},
        "audio": audio_payload,
        "request": {
            "model_name": "bigmodel",
            "enable_itn": True,
            "enable_punc": True,
            "enable_speaker_info": True,
            "enable_channel_split": False,
            "enable_ddc": False,
            "show_utterances": True,
            "vad_segment": True,
            "sensitive_words_filter": "",
        },
    }

    submit = requests.post(
        cfg["submit_url"],
        data=json.dumps(payload),
        headers=_volcengine_headers(cfg, task_id, sequence="-1"),
        timeout=60,
    )
    status = submit.headers.get("X-Api-Status-Code", "")
    if submit.status_code != 200 or status != "20000000":
        message = submit.headers.get("X-Api-Message", submit.text[:500])
        raise RuntimeError(f"Volcengine submit failed: {submit.status_code} {status} {message}".strip())

    logid = submit.headers.get("X-Tt-Logid", "")
    start = time.time()
    while time.time() - start <= poll_max_time:
        headers = _volcengine_headers(cfg, task_id)
        if logid:
            headers["X-Tt-Logid"] = logid
        query = requests.post(cfg["query_url"], data=json.dumps({}), headers=headers, timeout=30)
        q_status = query.headers.get("X-Api-Status-Code", "")
        if query.status_code in {429, 503} or q_status in {"20000001", "20000002", ""}:
            time.sleep(poll_interval)
            continue
        if query.status_code != 200:
            raise RuntimeError(f"Volcengine query failed: {query.status_code} {query.text[:500]}")
        if q_status == "20000000":
            body = query.json() if query.text.strip() else {}
            if body.get("result", {}).get("text") or body.get("result", {}).get("utterances"):
                return body
            time.sleep(poll_interval)
            continue
        message = query.headers.get("X-Api-Message", query.text[:500])
        raise RuntimeError(f"Volcengine query failed: {q_status} {message}".strip())

    raise RuntimeError(f"Volcengine transcription timed out after {poll_max_time:.0f}s")


def _speaker_id(utterance: dict) -> str | None:
    speaker = utterance.get("speaker")
    if speaker in [None, ""]:
        speaker = utterance.get("additions", {}).get("speaker")
    if speaker in [None, ""]:
        return None
    s = str(speaker)
    return s if s.startswith("speaker_") else f"speaker_{s}"


def normalize_volcengine_result(raw: dict, resource_id: str | None = None) -> dict:
    result = raw.get("result", {})
    utterances = result.get("utterances", []) or []
    words: list[dict] = []

    for utt in utterances:
        speaker_id = _speaker_id(utt)
        for w in utt.get("words", []) or []:
            text = (w.get("text") or "").strip()
            start_ms = w.get("start_time")
            end_ms = w.get("end_time")
            if not text or start_ms is None or end_ms is None:
                continue
            if float(start_ms) < 0 or float(end_ms) < 0:
                continue
            item = {
                "text": text,
                "type": "word",
                "start": float(start_ms) / 1000.0,
                "end": float(end_ms) / 1000.0,
            }
            if speaker_id:
                item["speaker_id"] = speaker_id
            if "confidence" in w:
                item["confidence"] = w.get("confidence")
            words.append(item)

        if not utt.get("words"):
            text = (utt.get("text") or "").strip()
            start_ms = utt.get("start_time")
            end_ms = utt.get("end_time")
            if text and start_ms is not None and end_ms is not None:
                item = {
                    "text": text,
                    "type": "word",
                    "start": float(start_ms) / 1000.0,
                    "end": float(end_ms) / 1000.0,
                }
                if speaker_id:
                    item["speaker_id"] = speaker_id
                words.append(item)

    words.sort(key=lambda item: (item.get("start", 0.0), item.get("end", 0.0)))
    return {
        "text": result.get("text", ""),
        "words": words,
        "metadata": {
            "provider": "volcengine",
            "resource_id": resource_id,
            "audio_duration_ms": raw.get("audio_info", {}).get("duration"),
        },
        "provider_raw": raw,
    }


def transcribe_one(
    video: Path,
    edit_dir: Path,
    api_key: str | None = None,
    provider: Provider = "elevenlabs",
    language: str | None = None,
    num_speakers: int | None = None,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Transcribe a single video. Returns path to transcript JSON.

    Cached: returns existing path immediately if the transcript already exists.
    """
    transcripts_dir = edit_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    out_path = transcripts_dir / f"{video.stem}.json"

    if out_path.exists() and not force:
        if verbose:
            print(f"cached: {out_path.name}")
        return out_path

    if verbose:
        print(f"  extracting audio from {video.name}", flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        if provider == "volcengine":
            audio = Path(tmp) / f"{video.stem}.mp3"
            extract_audio_mp3(video, audio)
        else:
            audio = Path(tmp) / f"{video.stem}.wav"
            extract_audio(video, audio)
        size_mb = audio.stat().st_size / (1024 * 1024)
        if verbose:
            print(f"  uploading {audio.name} ({size_mb:.1f} MB)", flush=True)
        if provider == "elevenlabs":
            payload = call_scribe(audio, api_key or load_api_key(), language, num_speakers)
        elif provider == "volcengine":
            cfg = load_volcengine_config()
            raw = call_volcengine_standard(audio, cfg, language=language)
            payload = normalize_volcengine_result(raw, resource_id=cfg["resource_id"])
        else:
            raise RuntimeError(f"unknown transcription provider: {provider}")

    out_path.write_text(json.dumps(payload, indent=2))
    dt = time.time() - t0

    if verbose:
        kb = out_path.stat().st_size / 1024
        print(f"  saved: {out_path.name} ({kb:.1f} KB) in {dt:.1f}s")
        if isinstance(payload, dict) and "words" in payload:
            print(f"    words: {len(payload['words'])}")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a video")
    ap.add_argument("video", type=Path, help="Path to video file")
    ap.add_argument(
        "--provider",
        choices=["elevenlabs", "volcengine"],
        default=os.environ.get("TRANSCRIBE_PROVIDER", "elevenlabs"),
        help="ASR provider (default: TRANSCRIBE_PROVIDER or elevenlabs)",
    )
    ap.add_argument(
        "--edit-dir",
        type=Path,
        default=None,
        help="Edit output directory (default: <video_parent>/edit)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional ISO language code (e.g., 'en'). Omit to auto-detect.",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional number of speakers when known. Improves diarization accuracy.",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite an existing cached transcript")
    args = ap.parse_args()

    video = args.video.resolve()
    if not video.exists():
        sys.exit(f"video not found: {video}")

    edit_dir = (args.edit_dir or (video.parent / "edit")).resolve()
    api_key = load_api_key() if args.provider == "elevenlabs" else None

    transcribe_one(
        video=video,
        edit_dir=edit_dir,
        api_key=api_key,
        provider=args.provider,
        language=args.language,
        num_speakers=args.num_speakers,
        force=args.force,
    )


if __name__ == "__main__":
    main()
