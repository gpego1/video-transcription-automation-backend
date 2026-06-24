import gc
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from database import get_db
from models.transcription import Transcription
from routers.auth import get_current_user
from services.audio_extractor import extract_audio
from services.transcriber import transcribe_audio_segments
from services.diarizer import diarize_segments

router = APIRouter(prefix="/transcriptions", tags=["transcriptions"])

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


@router.get("/features")
async def get_features():
    return {"diarization_available": True}


@router.post("/upload")
async def upload_transcription(
    file: UploadFile = File(...),
    model: str = Form("tiny"),
    language: str = Form(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported video format")

    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / f"upload{file_extension}"
        audio_path = Path(temp_dir) / "audio.wav"

        with open(video_path, "wb") as buffer:
            buffer.write(await file.read())

        extract_audio(str(video_path), str(audio_path))

        lang = None if not language or language.lower() == "auto" else language
        result = transcribe_audio_segments(str(audio_path), model_name=model, language=lang)

        segments = diarize_segments(str(audio_path), result["segments"])
        gc.collect()
        full_text = _format_text(segments)

        record = Transcription(
            user_id=current_user.id,
            filename=file.filename,
            transcription_text=full_text,
            model_used=model,
            language=language or "auto",
        )
        db.add(record)
        db.commit()
        db.refresh(record)

    return {
        "id": record.id,
        "text": full_text,
        "segments": segments,
        "model_used": record.model_used,
        "language": record.language,
        "created_at": str(record.created_at),
    }


def _format_text(segments):
    lines = []
    current_speaker = None
    for seg in segments:
        speaker = seg.get("speaker", "Voz 1")
        if speaker != current_speaker:
            current_speaker = speaker
            lines.append(f"\n[{speaker}]")
        lines.append(seg["text"])
    return " ".join(lines).strip()
