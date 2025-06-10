import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch.nn as nn
import numpy as np

EXCLUDED = {"342", "394", "398", "460"}

class TextOnlyBinaryDataset(Dataset):
    def __init__(self, csv_path, processed_dir):
        self.samples = []
        self.labels = []
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        for _, row in df.iterrows():
            pid = str(int(row["Participant_ID"]))
            if pid in EXCLUDED:
                continue
            session_id = f"{pid}_P"
            path = os.path.join(processed_dir, session_id, "bert_embeddings.pt")
            if os.path.exists(path):
                score = row.get("PHQ8_Score", row.get("PHQ_Score"))
                label = 1 if score >= 8 else 0
                self.samples.append(path)
                self.labels.append(label)
            else:
                print(f"Missing file: {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        data = torch.load(path)
        if isinstance(data, dict):
            emb = data.get("embeddings", list(data.values())[0])
        else:
            emb = data
        emb = torch.mean(emb, dim=0)
        return emb, torch.tensor(label, dtype=torch.long)

class TextClassificationModel(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    f1 = f1_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)

    print(f"✅ Eval Loss: {avg_loss:.4f}")
    print(f"F1 Score: {f1:.4f} | Accuracy: {acc:.4f}")
    print(classification_report(all_targets, all_preds, digits=4))

    return avg_loss, f1, acc

def main():
    processed_dir = "../data/processed"
    train_csv = "../train_split_Depression_AVEC2017.csv"
    dev_csv = "../dev_split_Depression_AVEC2017.csv"
    test_csv = "../full_test_split.csv"
    os.makedirs("models", exist_ok=True)

    batch_size = 8
    epochs = 50
    lr = 1e-3
    patience = 7
    weight_decay = 1e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = TextOnlyBinaryDataset(train_csv, processed_dir)
    dev_set = TextOnlyBinaryDataset(dev_csv, processed_dir)
    test_set = TextOnlyBinaryDataset(test_csv, processed_dir)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = TextClassificationModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_f1 = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        dev_loss, dev_f1, dev_acc = evaluate(model, dev_loader, criterion, device)

        print(f"Epoch {epoch:02d}: Train Loss={train_loss:.4f} | Dev Loss={dev_loss:.4f} | F1={dev_f1:.4f} | Acc={dev_acc:.4f}")

        scheduler.step(dev_loss)

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join("models", "best_text_classification_model.pt"))
            print(f"🎉 New best model saved with F1={best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⚠️ Early stopping triggered.")
                break

    model.load_state_dict(torch.load(os.path.join("models", "best_text_classification_model.pt")))
    test_loss, test_f1, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n✅ Test Loss: {test_loss:.4f} | F1: {test_f1:.4f} | Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
