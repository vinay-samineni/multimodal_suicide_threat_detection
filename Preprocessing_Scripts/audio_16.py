import os
import pandas as pd
import numpy as np
import subprocess

# CONFIG
opensmile_path = r"C:\OpenSMILE\bin\SMILExtract.exe"
config_path = r"C:\OpenSMILE\config\is09-13\IS13_ComParE_Voc.conf"
audio_root = r"..\data\extracted"
output_root = r"..\data\features\audio_compare16"
metadata_path = r"..\full_dataset.csv"

# Load labels
metadata = pd.read_csv(metadata_path)
metadata["Participant_ID"] = metadata["Participant_ID"].astype(str)

X = []
y = []

for _, row in metadata.iterrows():
    pid = row["Participant_ID"]
    label = row["PHQ8_Binary"]  # or use PHQ8_Score for regression
    
    audio_path = os.path.join(audio_root, f"{pid}_P", f"{pid}_AUDIO.wav")
    feature_path = os.path.join(output_root, f"{pid}.csv")

    if not os.path.exists(audio_path):
        print(f"[SKIP] Missing audio for {pid}")
        continue

    if not os.path.exists(feature_path):
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        print(f"[EXTRACTING] {pid}")
        try:
            subprocess.run([
                opensmile_path,
                "-C", config_path,
                "-I", audio_path,
                "-O", feature_path
            ], check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed extraction for {pid}")
            continue

    try:
        df = pd.read_csv(feature_path, delimiter=';')

        # Keep only numeric columns
        df_numeric = df.select_dtypes(include=[np.number])

        if df_numeric.empty:
            print(f"[SKIP] No numeric data for {pid}")
            continue

        # Convert to 1D feature vector
        features = df_numeric.values.flatten()

        if np.isnan(features).any():
            print(f"[SKIP] NaN in features for {pid}")
            continue

        X.append(features)
        y.append(label)

    except Exception as e:
        print(f"[ERROR] Reading failed for {pid}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"\n✅ Feature matrix shape: {X.shape}")
print(f"✅ Labels shape: {y.shape}")
