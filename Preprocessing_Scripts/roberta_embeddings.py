import os
import json
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXCLUDED_SESSIONS = {"342", "394", "398", "460" ,"451","480","458"}
PROCESSED_PATH = '../data/processed'
SAVE_PATH = '../data/features/text'

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base').to(DEVICE)
model.eval()

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / \
           torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def extract_embeddings_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Combine ellie + participant text for context
    texts = []
    for entry in data:
        ellie_text = entry.get("ellie", "").strip()
        participant_text = entry.get("participant", "").strip()
        combined = (ellie_text + " " + participant_text).strip()
        texts.append(combined)

    embeddings = []

    with torch.no_grad():
        for text in tqdm(texts, desc=f"Encoding {os.path.basename(json_path)}"):
            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
            model_output = model(**encoded_input)
            sentence_embedding = mean_pooling(model_output.last_hidden_state, encoded_input['attention_mask'])
            embeddings.append(sentence_embedding.squeeze(0).cpu())

    return torch.stack(embeddings)

def process_all_sessions(start_from=459):
    os.makedirs(SAVE_PATH, exist_ok=True)

    for session_folder in sorted(os.listdir(PROCESSED_PATH)):
        session_id = session_folder.split('_')[0]

        if not session_id.isdigit() or int(session_id) < start_from:
            continue  # Skip if not numeric or before 452

        if session_id in EXCLUDED_SESSIONS:
            continue

        json_file = os.path.join(PROCESSED_PATH, session_folder, f"{session_id}_CONTEXT_TRANSCRIPT.json")
        if not os.path.exists(json_file):
            print(f"Missing JSON for session {session_id}, skipping...")
            continue

        save_file = os.path.join(SAVE_PATH, f"{session_id}_roberta.pt")
        if os.path.exists(save_file):
            print(f"Already exists: {save_file}")
            continue

        print(f"Processing session {session_id}")
        session_embeddings = extract_embeddings_from_json(json_file)

        if session_embeddings is None or session_embeddings.shape[0] == 0:
            print(f"No valid embeddings for session {session_id}, skipping...")
            continue

        torch.save(session_embeddings, save_file)

if __name__ == "__main__":
    process_all_sessions()
