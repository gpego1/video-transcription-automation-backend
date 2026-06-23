import os
import tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from database import get_db
from models.transcription import Transcription
from routers.auth import get_current_user
from services.audio_extractor import extract_audio
from services.transcriber import transcribe_audio

router = APIRouter(prefix="/transcriptions", tags=["transcriptions"])

ALLOWED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


@router.post("/upload")
async def upload_transcription(
    file: UploadFile = File(...),
    model: str = Form("base"),
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

        transcription_text = transcribe_audio(
            str(audio_path),
            model_name=model,
            language=None if not language or language.lower() == "auto" else language,
        )

        record = Transcription(
            user_id=current_user.id,
            filename=file.filename,
            transcription_text=transcription_text,
            model_used=model,
            language=language or "auto",
        )
        db.add(record)
        db.commit()
        db.refresh(record)

    return {
        "id": record.id,
        "text": record.transcription_text,
        "model_used": record.model_used,
        "language": record.language,
        "created_at": record.created_at,
    }
