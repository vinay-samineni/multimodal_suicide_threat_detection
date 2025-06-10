import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

def load_opensmile_arff_agg(path):

    print(f"[DEBUG] Loading file: {path}")
    with open(path, 'r') as f:
        lines = f.readlines()

    start_idx = next((i for i, line in enumerate(lines) if '@data' in line.lower()), None)
    if start_idx is None:
        print("[ERROR] No '@data' found in file header")
        return None

    print(f"[DEBUG] '@data' found at line {start_idx}, reading CSV from here...")
    df = pd.read_csv(path, delimiter=',', skiprows=start_idx + 1, header=None)
    print(f"[DEBUG] Raw dataframe shape: {df.shape}")

    # Drop first and last columns (non-numeric)
    if df.shape[1] > 2:
        df = df.drop(columns=[0, df.shape[1]-1])
    else:
        print("[ERROR] Dataframe too small after reading")
        return None

    print(f"[DEBUG] Dataframe shape after dropping columns: {df.shape}")

    df = df.apply(pd.to_numeric, errors='coerce')
    nan_count = df.isna().sum().sum()
    print(f"[DEBUG] Number of NaNs before dropping rows: {nan_count}")

    df = df.dropna(axis=0)
    print(f"[DEBUG] Dataframe shape after dropping NaN rows: {df.shape}")

    if df.empty:
        print(f"[SKIP] Empty numeric data after cleaning for {os.path.basename(path)}")
        return None

    features = df.values
    feature_vector = np.concatenate([features.mean(axis=0), features.std(axis=0)])

    print(f"[INFO] Aggregated feature vector length: {len(feature_vector)}")
    return feature_vector


def main():
    features_dir = "../data/features/audio_compare16"
    metadata_path = "../full_dataset.csv"  # Your CSV with columns: Participant_ID, PHQ8_Binary
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    metadata["Participant_ID"] = metadata["Participant_ID"].astype(str)

    X = []
    y = []

    for idx, row in metadata.iterrows():
        pid = row["Participant_ID"]
        label = row["PHQ8_Binary"]

        feature_file = os.path.join(features_dir, f"{pid}.csv")
        if not os.path.isfile(feature_file):
            print(f"[SKIP] Missing feature file for {pid}")
            continue

        feat_vec = load_opensmile_arff_agg(feature_file)
        if feat_vec is None:
            continue

        X.append(feat_vec)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"[INFO] Total loaded samples: {len(X)}")
    if len(X) == 0:
        print("No data loaded, aborting training.")
        return

    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Balance training data using SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"[INFO] Class distribution after SMOTE: {np.bincount(y_train_res)}")

    # Train classifier
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_res, y_train_res)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("[RESULTS] Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    # Save model
    joblib.dump(clf, "random_forest_compare16_arff.pkl")
    print("[INFO] Model saved to random_forest_compare16_arff.pkl")

if __name__ == "__main__":
    main()
