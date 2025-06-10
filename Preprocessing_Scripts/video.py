import os
import pandas as pd
import torch
import json
import numpy as np
from tqdm import tqdm

EXCLUDED = {"342", "394", "398", "460","458","480"}
PROCESSED_ROOT = "../data/processed"
EXTRACTED_ROOT = "../data/extracted"

def process_session(session_id):
    session_num = session_id.split("_")[0]
    if session_num in EXCLUDED:
        return f"⏭️ Skipped {session_id}"

    au_path = os.path.join(EXTRACTED_ROOT, session_id, f"{session_num}_CLNF_AUs.txt")
    transcript_path = os.path.join(PROCESSED_ROOT, session_id, f"{session_num}_TRANSCRIPT.json")
    output_path = os.path.join(PROCESSED_ROOT, session_id, "video_embeddings.pt")

    if not os.path.exists(au_path) or not os.path.exists(transcript_path):
        return f"⚠️ Missing files for {session_id}"

    try:
        df = pd.read_csv(au_path)
        df.columns = df.columns.str.strip()
        df = df[df['success'] == 1]
        if df.empty:
            return f"⚠️ No successful AU frames for {session_id}"

        timestamps = df['timestamp'].values
        au_features = df.drop(columns=['frame', 'timestamp', 'confidence', 'success']).values
        au_tensor = torch.tensor(au_features, dtype=torch.float32)

        with open(transcript_path, "r") as f:
            transcript = json.load(f)

        aligned_aus = []

        for turn in transcript:
            for ans in turn["answers"]:
                val = ans["value"].strip().lower()
                if val in ["", "mhm", "uh-huh", "hmm", "mm", "um", "no answer"]:
                    continue
                start = float(ans["start_time"])
                stop = float(ans["stop_time"])
                center = (start + stop) / 2
                nearest_idx = np.argmin(np.abs(timestamps - center))
                au_vector = au_tensor[nearest_idx]
                aligned_aus.append(au_vector)

        if aligned_aus:
            torch.save(torch.stack(aligned_aus), output_path)
            return f"✅ {session_id}: Saved {len(aligned_aus)} video vectors"
        else:
            return f"⚠️ {session_id}: No matched AUs"
    except Exception as e:
        return f"❌ {session_id} failed: {str(e)}"

# === Run Batch
sessions = sorted(os.listdir(PROCESSED_ROOT))
results = []

for session_id in tqdm(sessions, desc="Video Feature Extraction"):
    if os.path.isdir(os.path.join(PROCESSED_ROOT, session_id)):
        results.append(process_session(session_id))

# Print summary
print("\n".join(results))
