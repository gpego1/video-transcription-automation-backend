import os
import whisper


def transcribe_audio(audio_path: str, model_name: str = "base", language: str | None = None) -> str:
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, language=language)
    return result.get("text", "")
