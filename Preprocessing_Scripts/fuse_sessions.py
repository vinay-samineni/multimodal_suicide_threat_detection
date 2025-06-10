import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# === Encoders and Fusion ===
class MelEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, h_n = self.gru(x)
        return h_n.squeeze(0)

class AUEncoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128):  # Use 20 if that's your actual AU vector size
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, input_dims=[768, 64, 128], proj_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(input_dims[0], proj_dim)
        self.audio_proj = nn.Linear(input_dims[1], proj_dim)
        self.video_proj = nn.Linear(input_dims[2], proj_dim)
        self.attn = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, text_feat, audio_feat, video_feat):
        t = self.text_proj(text_feat)
        a = self.audio_proj(audio_feat)
        v = self.video_proj(video_feat)
        stack = torch.stack([t, a, v], dim=1)
        attn_weights = torch.softmax(self.attn(stack), dim=1)
        return torch.sum(attn_weights * stack, dim=1)

# === Paths ===
PROCESSED_ROOT = "../data/processed"
EXCLUDED = {"342", "394", "398", "460"}

# === Models (shared across sessions) ===
mel_encoder = MelEncoder()
au_encoder = AUEncoder()
fusion = CrossModalAttentionFusion()

# === Batch Processing ===
sessions = sorted(os.listdir(PROCESSED_ROOT))
results = []

for session_id in tqdm(sessions, desc="Fusing all sessions"):
    session_path = os.path.join(PROCESSED_ROOT, session_id)
    session_num = session_id.split("_")[0]

    if not os.path.isdir(session_path) or session_num in EXCLUDED:
        continue

    try:
        bert_path = os.path.join(session_path, "bert_embeddings.pt")
        mel_path = os.path.join(session_path, "mel_spectrograms.pt")
        au_path = os.path.join(session_path, "video_embeddings.pt")

        if not all(os.path.exists(p) for p in [bert_path, mel_path, au_path]):
            results.append(f"⚠️ Missing one or more inputs for {session_id}")
            continue

        bert = torch.load(bert_path)
        mel = torch.load(mel_path)
        au = torch.load(au_path)

        min_len = min(len(bert), len(mel), len(au))
        bert = bert[:min_len]
        mel = mel[:min_len]
        au = au[:min_len]

        with torch.no_grad():
            mel_feat = mel_encoder(mel)
            au_feat = au_encoder(au)
            fused = fusion(bert, mel_feat, au_feat)

        out_path = os.path.join(session_path, "fused_embeddings.pt")
        torch.save(fused, out_path)
        results.append(f"✅ {session_id}: {fused.shape}")

    except Exception as e:
        results.append(f"❌ {session_id} failed: {str(e)}")

# === Summary ===
print("\n".join(results))
