import os
import json
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# Set paths
processed_root = "../data/processed"
excluded_sessions = {"342", "394", "398", "460"}
embedding_filename = "bert_embeddings.pt"
transcript_filename = "PARTICIPANT_TRANSCRIPT.json"

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
device = torch.device("cpu")
model.to(device)
model.eval()

def extract_bert_embeddings(transcript_path, save_path):
    with open(transcript_path, "r") as f:
        data = json.load(f)

    embeddings = []
    timestamps = []

    for entry in data:
        text = entry["value"].strip()
        if text and text.lower() not in ["mhm", "uh-huh", "hmm", "mm", "um", "no answer"]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # Mean pooling over token embeddings
            mean_pooled = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            embeddings.append(mean_pooled)
            timestamps.append((entry["start_time"], entry["stop_time"]))

    if embeddings:
        stacked = torch.stack(embeddings)
        torch.save({"embeddings": stacked, "timestamps": timestamps}, save_path)
        print(f"✅ Saved {len(embeddings)} embeddings + timestamps → {save_path}")
    else:
        print(f"⚠️ No valid embeddings in {transcript_path}")


# Loop through all session folders
for session_id in tqdm(os.listdir(processed_root), desc="Processing sessions"):
    session_number = session_id.split("_")[0]
    if session_number in excluded_sessions:
        continue

    session_path = os.path.join(processed_root, session_id)
    transcript_path = os.path.join(session_path, f"{session_number}_PARTICIPANT_TRANSCRIPT.json")
    save_path = os.path.join(session_path, embedding_filename)

    if not os.path.exists(transcript_path):
        print(f"❌ Missing participant transcript for {session_id}")
        continue

    extract_bert_embeddings(transcript_path, save_path)
