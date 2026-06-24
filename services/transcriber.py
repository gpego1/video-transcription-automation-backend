import gc
import whisper


def transcribe_audio(audio_path: str, model_name: str = "tiny", language: str | None = None) -> str:
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language=language)
    del model
    gc.collect()
    return result.get("text", "")


def transcribe_audio_segments(audio_path: str, model_name: str = "tiny", language: str | None = None) -> dict:
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language=language)
    del model
    gc.collect()
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
