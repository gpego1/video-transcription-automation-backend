import os
from moviepy.editor import VideoFileClip


def extract_audio(video_path: str, output_audio_path: str) -> None:
    with VideoFileClip(video_path) as clip:
        audio = clip.audio
        if audio is None:
            raise ValueError("Could not extract audio from video")
        audio.write_audiofile(output_audio_path, fps=16000, codec="pcm_s16le")
