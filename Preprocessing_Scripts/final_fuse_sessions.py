import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# === Function to Load openSMILE Embeddings ===
def load_opensmile_arff_agg(path):
    print(f"[DEBUG] Loading file: {path}")
    if not os.path.isfile(path):
        print(f"[ERROR] File does not exist: {path}")
        return None

    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[ERROR] Failed to read file {path}: {str(e)}")
        return None

    print(f"[DEBUG] Number of lines in file: {len(lines)}")
    print(f"[DEBUG] First 5 lines of file:\n{''.join(lines[:5])}")

    start_idx = next((i for i, line in enumerate(lines) if '@data' in line.lower()), None)
    if start_idx is None:
        print("[ERROR] No '@data' found in file header")
        return None
    print(f"[DEBUG] '@data' found at line {start_idx}")

    if start_idx + 1 >= len(lines):
        print(f"[ERROR] No data after '@data' section in {path}")
        return None

    # Read CSV data starting after @data
    try:
        df = pd.read_csv(path, delimiter=',', skiprows=start_idx + 1, header=None)
    except Exception as e:
        print(f"[ERROR] Failed to parse with comma delimiter: {str(e)}")
        try:
            df = pd.read_csv(path, delimiter=';', skiprows=start_idx + 1, header=None)
            print(f"[DEBUG] Successfully parsed with semicolon delimiter")
        except Exception as e:
            print(f"[ERROR] Failed to parse with semicolon delimiter: {str(e)}")
            return None

    print(f"[DEBUG] Raw dataframe shape: {df.shape}")
    print(f"[DEBUG] Sample of raw data (first 2 rows):\n{df.head(2).to_string()}")

    if df.shape[1] < 3:
        print(f"[ERROR] Dataframe has too few columns: {df.shape[1]} for {os.path.basename(path)}")
        return None

    # Drop first (name) and last (class) columns
    df = df.drop(columns=[0, df.shape[1]-1])
    print(f"[DEBUG] Dataframe shape after dropping columns: {df.shape}")

    # Check for non-numeric data and drop problematic columns early
    non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns
    if len(non_numeric_cols) > 0:
        print(f"[DEBUG] Non-numeric columns detected: {list(non_numeric_cols)}")
        print(f"[DEBUG] Sample of non-numeric data:\n{df[non_numeric_cols].head(2).to_string()}")
        df = df.drop(columns=non_numeric_cols)
        print(f"[DEBUG] Dropped {len(non_numeric_cols)} non-numeric columns. New shape: {df.shape}")

    if df.empty or df.shape[1] == 0:
        print(f"[ERROR] No numeric columns remaining after dropping non-numeric columns for {os.path.basename(path)}")
        return None

    # Convert to numeric
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    nan_cols = df_numeric.columns[df_numeric.isna().any()].tolist()
    nan_count = df_numeric.isna().sum().sum()
    print(f"[DEBUG] Columns with NaNs: {nan_cols}")
    print(f"[DEBUG] Total number of NaNs: {nan_count}")
    if nan_count > 0:
        print(f"[DEBUG] Sample of rows with NaNs (first 2):\n{df_numeric[df_numeric.isna().any(axis=1)].head(2).to_string()}")

    # Drop columns with all NaNs
    initial_cols = df_numeric.shape[1]
    df_numeric = df_numeric.dropna(axis=1, how='all')
    print(f"[DEBUG] Dropped {initial_cols - df_numeric.shape[1]} columns with all NaNs. New shape: {df_numeric.shape}")

    # Drop rows with NaNs
    df_numeric = df_numeric.dropna(axis=0)
    print(f"[DEBUG] Dataframe shape after dropping NaN rows: {df_numeric.shape}")

    if df_numeric.empty:
        print(f"[ERROR] Empty numeric data after cleaning for {os.path.basename(path)}")
        return None

    features = df_numeric.values
    print(f"[DEBUG] Shape of feature array: {features.shape}")
    feature_vector = np.concatenate([features.mean(axis=0), features.std(axis=0)])
    print(f"[DEBUG] Aggregated feature vector length: {len(feature_vector)}")
    print(f"[DEBUG] First 5 values of feature vector: {feature_vector[:5]}")
    return feature_vector

# === Fusion Module ===
class CrossModalAttentionFusion(nn.Module):
    def __init__(self, input_dims=[768, 284], proj_dim=256):  # Dimensions from previous fix
        super().__init__()
        self.text_proj = nn.Linear(input_dims[0], proj_dim)
        self.audio_proj = nn.Linear(input_dims[1], proj_dim)
        self.attn = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Debug the shapes of the projection layers
        print(f"[DEBUG] text_proj weight shape: {self.text_proj.weight.shape}")  # Should be (256, 768)
        print(f"[DEBUG] audio_proj weight shape: {self.audio_proj.weight.shape}")  # Should be (256, 284)

    def forward(self, text_feat, audio_feat):
        print(f"[DEBUG] text_feat shape before projection: {text_feat.shape}")
        print(f"[DEBUG] audio_feat shape before projection: {audio_feat.shape}")
        t = self.text_proj(text_feat)
        a = self.audio_proj(audio_feat)
        print(f"[DEBUG] text_feat shape after projection: {t.shape}")
        print(f"[DEBUG] audio_feat shape after projection: {a.shape}")
        stack = torch.stack([t, a], dim=1)
        attn_weights = torch.softmax(self.attn(stack), dim=1)
        return torch.sum(attn_weights * stack, dim=1)

# === Classifier ===
class Classifier(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# === Paths ===
TEXT_FEATURES_DIR = "../data/features/text"
AUDIO_FEATURES_DIR = "../data/features/audio_compare16"
METADATA_PATH = "../full_dataset.csv"
EXCLUDED = {"342", "394", "398", "460"}

# === Models ===
fusion = CrossModalAttentionFusion()
classifier = Classifier()

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion.to(device)
classifier.to(device)

# === Load Metadata ===
metadata = pd.read_csv(METADATA_PATH)
metadata["Participant_ID"] = metadata["Participant_ID"].astype(str)
print(f"[DEBUG] Metadata shape: {metadata.shape}")
print(f"[DEBUG] Metadata columns: {list(metadata.columns)}")
print(f"[DEBUG] PHQ8_Binary value counts:\n{metadata['PHQ8_Binary'].value_counts().to_string()}")

# === Batch Processing to Fuse Embeddings ===
X = []
y = []
results = []

for participant_id in tqdm(metadata["Participant_ID"], desc="Fusing all participants"):
    if participant_id in EXCLUDED:
        continue

    try:
        # Load text embeddings
        text_file = os.path.join(TEXT_FEATURES_DIR, f"{participant_id}_roberta.pt")
        if not os.path.isfile(text_file):
            results.append(f"⚠️ Missing RoBERTa embeddings for {participant_id} at {text_file}")
            continue

        text = torch.load(text_file, map_location=device)
        # Debug the structure of text
        print(f"[DEBUG] Structure of {participant_id}_roberta.pt: {type(text)}")
        if isinstance(text, dict):
            print(f"[DEBUG] Keys in text dictionary: {list(text.keys())}")
            # Attempt to access embeddings under common keys
            possible_keys = ['embeddings', 'features', 'last_hidden_state', 'pooled_output']
            for key in possible_keys:
                if key in text:
                    text = text[key]
                    print(f"[DEBUG] Using '{key}' key for embeddings. Shape: {text.shape}")
                    break
            else:
                raise KeyError(f"No suitable key found in text dictionary. Available keys: {list(text.keys())}")

        # Ensure text is a tensor
        if not isinstance(text, torch.Tensor):
            raise TypeError(f"RoBERTa embeddings must be a torch.Tensor, got {type(text)}")

        # Average RoBERTa embeddings if necessary
        if len(text.shape) > 1:
            print(f"[DEBUG] RoBERTa shape before averaging: {text.shape}")
            text = text.mean(dim=0)  # Shape: (768,)
            print(f"[DEBUG] RoBERTa shape after averaging: {text.shape}")
        else:
            print(f"[DEBUG] RoBERTa shape (already averaged): {text.shape}")

        # Load openSMILE embeddings
        audio_file = os.path.join(AUDIO_FEATURES_DIR, f"{participant_id}.csv")
        if not os.path.isfile(audio_file):
            results.append(f"⚠️ Missing openSMILE embeddings for {participant_id} at {audio_file}")
            continue

        audio_feat = load_opensmile_arff_agg(audio_file)
        if audio_feat is None:
            results.append(f"❌ Failed to load openSMILE embeddings for {participant_id}")
            continue

        audio_feat = torch.tensor(audio_feat, dtype=torch.float32).to(device)

        # Fuse embeddings
        with torch.no_grad():
            fusion.eval()
            fused = fusion(text, audio_feat)

        # Get label from metadata
        label = metadata[metadata["Participant_ID"] == participant_id]["PHQ8_Binary"].values[0]

        X.append(fused.cpu().numpy())
        y.append(label)
        results.append(f"✅ {participant_id}: Fused shape {fused.shape}")

    except Exception as e:
        results.append(f"❌ {participant_id} failed: {str(e)}")

# === Summary of Fusion ===
print("\nFusion Summary:")
print("\n".join(results))

# === Prepare Data for Training ===
X = np.array(X)
y = np.array(y)
print(f"[DEBUG] Total loaded samples: {len(X)}")
print(f"[DEBUG] Shape of X: {X.shape}")
print(f"[DEBUG] Shape of y: {y.shape}")
if len(X) > 0:  # Only print class distribution if y is not empty
    print(f"[DEBUG] Class distribution in y: {np.bincount(y.astype(int))}")
else:
    print("[ERROR] No data loaded, aborting training.")
    exit()

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
print(f"[DEBUG] Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"[DEBUG] Training labels distribution: {np.bincount(y_train.astype(int))}")
print(f"[DEBUG] Test labels distribution: {np.bincount(y_test.astype(int))}")

# Balance training data using SMOTE
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"[DEBUG] Shape of X_train_res after SMOTE: {X_train_res.shape}")
print(f"[DEBUG] Class distribution after SMOTE: {np.bincount(y_train_res)}")

# Convert to PyTorch tensors
X_train_res = torch.tensor(X_train_res, dtype=torch.float32).to(device)
y_train_res = torch.tensor(y_train_res, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# === Training ===
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
criterion = nn.BCELoss()
num_epochs = 50

classifier.train()
for epoch in tqdm(range(num_epochs), desc="Training"):
    optimizer.zero_grad()
    outputs = classifier(X_train_res).squeeze()
    loss = criterion(outputs, y_train_res)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"[DEBUG] Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# === Evaluation ===
classifier.eval()
with torch.no_grad():
    y_pred_proba = classifier(X_test).squeeze().cpu().numpy()
    threshold = 0.2  # Lower threshold to prioritize recall for class 1
    y_pred = (y_pred_proba >= threshold).astype(int)

print("[RESULTS] Classification Report:")
print(classification_report(y_test, y_pred, digits=4))
print(f"[RESULTS] Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"[RESULTS] F1 Score (class 1): {f1_score(y_test, y_pred):.4f}")

# Save the classifier
torch.save(classifier.state_dict(), "text_audio_fusion_classifier.pt")
print("[INFO] Classifier saved to text_audio_fusion_classifier.pt")