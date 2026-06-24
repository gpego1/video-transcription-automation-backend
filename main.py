import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import engine, Base
from routers import auth, transcriptions, history

app = FastAPI(title="Video Transcriber API")

frontend_url = os.getenv("FRONTEND_URL", "").rstrip("/")
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
]
if frontend_url:
    origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(transcriptions.router)
app.include_router(history.router)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Video Transcriber API is running."}
