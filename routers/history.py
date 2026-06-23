from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from models.transcription import Transcription
from routers.auth import get_current_user

router = APIRouter(prefix="/history", tags=["history"])


class TranscriptionResponse(BaseModel):
    id: int
    filename: str
    transcription_text: str
    model_used: str
    language: str
    created_at: str

    model_config = {"from_attributes": True}


@router.get("/", response_model=list[TranscriptionResponse])
def get_history(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    records = (
        db.query(Transcription)
        .filter(Transcription.user_id == current_user.id)
        .order_by(Transcription.created_at.desc())
        .all()
    )
    return records


@router.get("/{record_id}", response_model=TranscriptionResponse)
def get_history_item(record_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    record = (
        db.query(Transcription)
        .filter(Transcription.id == record_id, Transcription.user_id == current_user.id)
        .first()
    )
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcription not found")
    return record


@router.delete("/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_history_item(record_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    record = (
        db.query(Transcription)
        .filter(Transcription.id == record_id, Transcription.user_id == current_user.id)
        .first()
    )
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transcription not found")
    db.delete(record)
    db.commit()
    return None
