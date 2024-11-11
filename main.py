import assemblyai as aai
from moviepy.editor import *
import os
from groq import Groq
import torch
from TTS.api import TTS
import webvtt
from moviepy.video.fx.all import loop
import time
start_time = time.time()


# Datei-Pfade
DEST_SUBTITLE = "pfad/zu/subtitles.vtt"  # Pfad zur VTT-Datei der Untertitel
FINAL_VIDEO_NAME = "pfad/zu/final_video.mp4"  # Pfad zum finalen Video
DEST_BACKCLIP = "pfad/zu/backclip.mp4"  # Pfad zum Hintergrundvideo

DEST_VIDEO_FOLDER = "pfad/zu/video_folder"  # Pfad zum Ordner, in dem das Video gespeichert wird
DEST_VOICEOVER = "pfad/zu/voiceover.wav"  # Pfad zur Voiceover-Audiodatei
DEST_BACKMUSIC = "pfad/zu/background_music.wav"  # Pfad zur Hintergrundmusik-Audiodatei
STORYTEXT_PATH = "pfad/zu/storytext.txt"  # Pfad zur Textdatei mit der Geschichte
PROMPT_PATH = "pfad/zu/prompt.txt"  # Pfad zur Eingabedatei für die Generierung des Textes
VOICE_PATH = "pfad/zu/voice.wav"  # Pfad zur Sprachdatei

# API-Schlüssel
aai.settings.api_key = "dein_assembly_schlüssel"  # Ersetze mit deinem AssemblyAI-Schlüssel
os.environ['GROQ_API_KEY'] = "dein_groq_schlüssel"  # Ersetze mit deinem Groq-Schlüssel

# Einstellungen
WIDTH, HEIGHT = 1080, 1920  # Auflösung des Videos (Breite x Höhe)
TXTCLIP_SIZE = (1080, 1920)  # Größe der Textclips, die in das Video eingebaut werden
CHARS_PER_CAPTION = 11  # Maximale Anzahl von Zeichen pro Untertitel
MAX_TOKEN = 10  # Maximale Anzahl von Tokens für die Textgenerierung
BACKMUSIC_VOLUME = 0.5  # Lautstärke der Hintergrundmusik (0.0 bis 1.0)
FONTSIZE_SUBCLIP = 150  # Schriftgröße der Untertitel
COLOR_SUBCLIP = 'white'  # Textfarbe der Untertitel
BG_COLOR_SUBCLIP = 'transparent'  # Hintergrundfarbe der Untertitel (transparent bedeutet ohne Hintergrund)
FONT_SUBCLIP = 'Arial-Bold'  # Schriftart der Untertitel
STROKE_COLOR_SUBCLIP = 'black'  # Randfarbe der Untertitel
STROKE_WIDTH_SUBCLIP = 0  # Randbreite der Untertitel (0 bedeutet kein Rand)
SHADOW_OFFSET = 5  # Versatz des Schattens der Untertitel
SHADOW_COLOR = 'black'  # Schattenfarbe der Untertitel
SHADOW_STROKE_WIDTH = 30  # Breite des Randes des Schattens


client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Remove the existing subtitle file if it exists to avoid conflicts
if os.path.exists(DEST_SUBTITLE):
    os.remove(DEST_SUBTITLE)

# Load prompt from file and generate story text
with open(PROMPT_PATH, 'r', encoding='utf8') as file:
    prompt = file.read()
chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="llama-3.2-90b-vision-preview",
    max_tokens=MAX_TOKEN
)
storytext = chat_completion.choices[0].message.content
with open(STORYTEXT_PATH, 'w', encoding='utf8') as file:
    file.write(storytext)

# Generate voiceover audio from story text
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(text=storytext, speaker_wav=VOICE_PATH, language="en", file_path=DEST_VOICEOVER)

# Transcribe the audio and save subtitles in VTT format
transcript = aai.Transcriber().transcribe(DEST_VOICEOVER)
subtitles = transcript.export_subtitles_vtt(chars_per_caption=CHARS_PER_CAPTION)
with open(DEST_SUBTITLE, "a") as subtitlefile:
    subtitlefile.write(subtitles)

# Helper function to convert VTT time format to seconds
def vtt_time_to_seconds(time_str):
    h, m, s = time_str.split(':')
    s, ms = s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

# Function to create subtitle clips from VTT file
def create_subtitle_clips(vtt_file):
    subtitle_clips = []
    for caption in webvtt.read(vtt_file):
        start_time = vtt_time_to_seconds(caption.start)
        end_time = vtt_time_to_seconds(caption.end)
        duration = end_time - start_time
        subtitle = TextClip(caption.text, fontsize=FONTSIZE_SUBCLIP, color=COLOR_SUBCLIP,
                            bg_color=BG_COLOR_SUBCLIP, font=FONT_SUBCLIP, 
                            stroke_color=STROKE_COLOR_SUBCLIP, stroke_width=STROKE_WIDTH_SUBCLIP)
        subtitle = subtitle.set_position((WIDTH / 2 - subtitle.w / 2, HEIGHT / 2 - subtitle.h / 2)).set_duration(duration)
        subtitle = subtitle.resize(lambda t: 1 + min(t / 0.08, 0.03))
        subtitle = subtitle.set_start(start_time)
        subtitle_clips.append(subtitle)
    return subtitle_clips

# Function to create shadow clips for subtitles
def create_shadow_clips(vtt_file):
    shadow_clips = []
    for caption in webvtt.read(vtt_file):
        start_time = vtt_time_to_seconds(caption.start)
        end_time = vtt_time_to_seconds(caption.end)
        duration = end_time - start_time
        shadow = TextClip(caption.text, fontsize=FONTSIZE_SUBCLIP, color=SHADOW_COLOR,
                          bg_color=BG_COLOR_SUBCLIP, font=FONT_SUBCLIP,
                          stroke_color=SHADOW_COLOR, stroke_width=SHADOW_STROKE_WIDTH)
        shadow = shadow.set_position((WIDTH / 2 - shadow.w / 2 + SHADOW_OFFSET, HEIGHT / 2 - shadow.h / 2)).set_duration(duration)
        shadow = shadow.resize(lambda t: 1 + min(t / 0.08, 0.03))
        shadow = shadow.set_start(start_time)
        shadow_clips.append(shadow)
    return shadow_clips

# Function to add subtitles and shadow clips to video
def add_subtitles_to_video(video_path, vtt_file):
    video = VideoFileClip(video_path).set_position(("center", "center"))
    video = video.resize(height=HEIGHT) if video.w > video.h else video.resize(width=WIDTH)
    subtitle_clips = create_subtitle_clips(vtt_file)
    shadow_clips = create_shadow_clips(vtt_file)
    return CompositeVideoClip([video] + shadow_clips + subtitle_clips, size=TXTCLIP_SIZE)

# Function to add background music to video with looping if needed
def add_background_music(video, music_path, volume):
    music = AudioFileClip(music_path).volumex(volume)
    music = afx.audio_loop(music, duration=video.duration) if music.duration < video.duration else music.subclip(0, video.duration)
    final_audio = CompositeAudioClip([video.audio, music.set_start(0)])
    return video.set_audio(final_audio)

# Process video: add audio, subtitles, and background music
audio_clip = AudioFileClip(DEST_VOICEOVER)
back_clip = VideoFileClip(DEST_BACKCLIP).subclip(0, audio_clip.duration).without_audio().set_audio(audio_clip)
back_clip.write_videofile(DEST_VIDEO_FOLDER + "/backvideo.mp4")

final_video = add_subtitles_to_video(DEST_VIDEO_FOLDER + "/backvideo.mp4", DEST_SUBTITLE)
final_video = add_background_music(final_video, DEST_BACKMUSIC, BACKMUSIC_VOLUME)
final_video.write_videofile(DEST_VIDEO_FOLDER + FINAL_VIDEO_NAME, codec="libx264")

# to measure the time 
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)