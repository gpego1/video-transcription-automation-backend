import whisper

_model_cache = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        _model_cache[model_name] = whisper.load_model(model_name)
    return _model_cache[model_name]


def transcribe_audio(audio_path: str, model_name: str = "base", language: str | None = None) -> str:
    model = _get_model(model_name)
    result = model.transcribe(audio_path, language=language)
    return result.get("text", "")


def transcribe_audio_segments(audio_path: str, model_name: str = "base", language: str | None = None) -> dict:
    model = _get_model(model_name)
    result = model.transcribe(audio_path, language=language)
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip(),
        })
    return {
        "text": result.get("text", ""),
        "segments": segments,
        "language": result.get("language", ""),
    }
