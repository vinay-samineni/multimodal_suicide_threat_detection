import os
import json
import torch
import librosa
import numpy as np

EXCLUDED = {"342", "394", "398", "460"}

processed_dir = "../data/processed"   # contains folders like 300_P/
wav_dir = "../data/extracted"         # contains folders like 300_P/
output_dir = "../data/features/audio_mel"
os.makedirs(output_dir, exist_ok=True)

# Mel spectrogram parameters
sr = 16000
n_fft = 1024
hop_length = 256
n_mels = 64

def extract_mel_segment(wav_path, start_sec, stop_sec):
    y, _ = librosa.load(wav_path, sr=sr, offset=start_sec, duration=stop_sec - start_sec)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db  # (n_mels, time_frames)

def process_session(session_id):
    pid = session_id.split('_')[0]  # e.g. "300" from "300_P"
    if pid in EXCLUDED:
        print(f"Skipping excluded participant {pid}")
        return

    session_processed_dir = os.path.join(processed_dir, session_id)
    session_wav_dir = os.path.join(wav_dir, session_id)

    json_path = os.path.join(session_processed_dir, f"{pid}_PARTICIPANT_Transcript.json")
    wav_path = os.path.join(session_wav_dir, f"{pid}_AUDIO.wav")

    if not os.path.exists(json_path):
        print(f"Missing JSON file for session {session_id}, skipping...")
        return
    if not os.path.exists(wav_path):
        print(f"Missing WAV file for session {session_id}, skipping...")
        return

    with open(json_path, "r") as f:
        timestamps = json.load(f)

    mel_segments = []
    for utt in timestamps:
        start_time = utt.get("start_time")
        stop_time = utt.get("stop_time")
        if start_time is None or stop_time is None:
            continue
        mel = extract_mel_segment(wav_path, start_time, stop_time)
        mel_segments.append(torch.tensor(mel))

    if mel_segments:
        save_path = os.path.join(output_dir, f"{session_id}_mel.pt")
        torch.save(mel_segments, save_path)
        print(f"Saved mel segments for {session_id} with {len(mel_segments)} utterances at {save_path}")
    else:
        print(f"No mel segments extracted for {session_id}")

def batch_process_all():
    session_ids = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    for session_id in session_ids:
        process_session(session_id)

if __name__ == "__main__":
    batch_process_all()
