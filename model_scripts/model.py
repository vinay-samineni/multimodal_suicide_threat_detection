# === Binary Classification Pipeline ===
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
import torch.nn as nn
import numpy as np

EXCLUDED = {"342", "394", "398", "460","458","480"}

# === Dataset for Binary Classification ===
class FusedPHQ8BinaryDataset(Dataset):
    def __init__(self, csv_path, processed_dir):
        self.samples = []
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            pid = int(row["Participant_ID"])
            if pid in EXCLUDED:
                continue
            score = row.get("PHQ8_Score", row.get("PHQ_Score"))
            label = int(score >= 10)  # Binary label: 1 if depressed
            session_id = f"{pid}_P"
            path = os.path.join(processed_dir, session_id, "fused_embeddings.pt")
            if os.path.exists(path):
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = torch.load(path)
        x = torch.mean(x, dim=0)  # (256,)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# === Classifier Model ===
class SessionClassifier(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary classification logits
        )

    def forward(self, x):
        return self.net(x)


# === Training Utilities ===
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())
    return total_loss / len(loader.dataset), all_preds, all_targets


# === Training Script ===
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    processed_dir = "../data/processed"
    train_csv = "../train_split_Depression_AVEC2017.csv"
    dev_csv = "../dev_split_Depression_AVEC2017.csv"
    test_csv = "../full_test_split.csv"

    batch_size = 8
    epochs = 30
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = FusedPHQ8BinaryDataset(train_csv, processed_dir)
    dev_set = FusedPHQ8BinaryDataset(dev_csv, processed_dir)
    test_set = FusedPHQ8BinaryDataset(test_csv, processed_dir)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = SessionClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        dev_loss, dev_preds, dev_targets = evaluate(model, dev_loader, criterion, device)
        dev_f1 = f1_score(dev_targets, dev_preds)
        acc = accuracy_score(dev_targets, dev_preds)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | Dev Loss={dev_loss:.4f} | F1={dev_f1:.3f} | Acc={acc:.3f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model.state_dict(), os.path.join("models", "best_binary_model.pt"))

    # === Final Test ===
    model.load_state_dict(torch.load(os.path.join("models", "best_model.pt")))
    test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, device)

    test_f1 = f1_score(test_targets, test_preds)
    test_acc = accuracy_score(test_targets, test_preds)

    print(f"\n✅ Test Accuracy: {test_acc:.3f} | F1 Score: {test_f1:.3f}")
