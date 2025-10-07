import platform
import os
import ffmpeg
import numpy as np
import librosa
import soundfile as sf
import traceback
import re


def wav2(input_path, output_path, target_format):
    """
    Convert audio file to another format using librosa + soundfile.
    Supports: wav, mp3, ogg, m4a, flac
    """
    input_path = clean_path(input_path)
    if not os.path.exists(input_path):
        raise RuntimeError(f"Input file does not exist: {input_path}")

    # librosa loads audio as float32 numpy array
    y, sr = librosa.load(input_path, sr=None, mono=False)

    # soundfile format mapping
    fmt_map = {
        "wav": "WAV",
        "flac": "FLAC",
        "ogg": "OGG",
        "m4a": "MP4",  # Some systems may need "MP4" or "AAC"
        "mp3": "MP3"
    }

    sf_format = fmt_map.get(target_format.lower())
    if sf_format is None:
        raise ValueError(f"Unsupported output format: {target_format}")

    # For mono conversion if necessary
    if y.ndim > 1 and sf_format != "OGG":  # OGG can handle multi-channel
        y = librosa.to_mono(y)

    # Write audio
    sf.write(output_path, y.T, sr, format=sf_format)


def load_audio(file, sr):
    try:
        # Clean path first
        file = clean_path(file)
        if not os.path.exists(file):
            raise RuntimeError("You input a wrong audio path that does not exist!")

        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path_str):
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    path_str = re.sub(r'[\u202a-\u202e]', '', path_str)  # Remove Unicode control chars
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
