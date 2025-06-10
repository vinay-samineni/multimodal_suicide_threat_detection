import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch.nn as nn

# Sessions to exclude (invalid/corrupted/missing files)
EXCLUDED = {"342", "394", "398", "460"}

class RobertaBinaryDataset(Dataset):
    def __init__(self, csv_path, roberta_dir):
        self.samples = []
        self.labels = []

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")

        for _, row in df.iterrows():
            pid = str(int(row["Participant_ID"]))
            if pid in EXCLUDED:
                continue

            path = os.path.join(roberta_dir, f"{pid}_roberta.pt")
            if os.path.exists(path):
                score = row.get("PHQ8_Score", row.get("PHQ_Score"))
                label = 1 if score >=8 else 0
                self.samples.append(path)
                self.labels.append(label)
            else:
                print(f"Missing file: {path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        id = self.ids[idx]
    # Example filename pattern; adjust if needed
        feature_path = os.path.join(self.feature_dir, f"{id}_mel.pt")  
    
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Missing feature file: {feature_path}")
    
        features = torch.load(feature_path)  # load .pt file, returns a tensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

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
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * len(x)

            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    f1 = f1_score(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)

    print(f"✅ Eval Loss: {avg_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")
    print(classification_report(all_targets, all_preds, digits=4))

    return avg_loss, f1, acc

def main():
    # Paths
    roberta_dir = "../data/features/text"
    train_csv = "../train_split_Depression_AVEC2017.csv"
    dev_csv = "../dev_split_Depression_AVEC2017.csv"
    test_csv = "../full_test_split.csv"
    os.makedirs("models", exist_ok=True)

    # Hyperparameters
    batch_size = 8
    epochs = 50
    lr = 1e-3
    weight_decay = 1e-5
    patience = 7

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets and loaders
    train_set = RobertaBinaryDataset(train_csv, roberta_dir)
    dev_set = RobertaBinaryDataset(dev_csv, roberta_dir)
    test_set = RobertaBinaryDataset(test_csv, roberta_dir)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Model, Loss, Optimizer
    model = TextClassificationModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_f1 = 0
    patience_counter = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        dev_loss, dev_f1, dev_acc = evaluate(model, dev_loader, criterion, device)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Dev Loss: {dev_loss:.4f} | F1: {dev_f1:.4f} | Acc: {dev_acc:.4f}")

        scheduler.step(dev_loss)

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_text_model.pt")
            print(f"🎯 New best model saved (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    # Final test evaluation
    model.load_state_dict(torch.load("models/best_text_model.pt"))
    test_loss, test_f1, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n✅ Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f} | Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
