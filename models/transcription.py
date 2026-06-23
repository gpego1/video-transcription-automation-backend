from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text

from database import Base


class Transcription(Base):
    __tablename__ = "transcriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    transcription_text = Column(Text, nullable=False)
    model_used = Column(String, nullable=False)
    language = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
