import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib

def load_opensmile_arff_agg(path):
    print(f"[DEBUG] Loading file: {path}")
    if not os.path.isfile(path):
        print(f"[ERROR] File does not exist: {path}")
        return None

    try:
        with open(path, 'r') as f:
            lines = f.readlines()
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

    except Exception as e:
        print(f"[ERROR] Failed to process file {path}: {str(e)}")
        return None

def main():
    features_dir = "../data/features/audio_compare16"
    metadata_path = "../full_dataset.csv"

    print(f"[DEBUG] Checking features directory: {features_dir}")
    if not os.path.exists(features_dir):
        print(f"[ERROR] Features directory does not exist: {features_dir}")
        return
    print(f"[DEBUG] Number of files in features directory: {len(os.listdir(features_dir))}")
    print(f"[DEBUG] Sample of files in directory: {os.listdir(features_dir)[:5]}")

    print(f"[DEBUG] Checking metadata file: {metadata_path}")
    if not os.path.exists(metadata_path):
        print(f"[ERROR] Metadata file does not exist: {metadata_path}")
        return

    try:
        # Load metadata
        metadata = pd.read_csv(metadata_path)
        print(f"[DEBUG] Metadata shape: {metadata.shape}")
        print(f"[DEBUG] Metadata columns: {list(metadata.columns)}")
        print(f"[DEBUG] Sample of metadata (first 2 rows):\n{metadata.head(2).to_string()}")

        metadata["Participant_ID"] = metadata["Participant_ID"].astype(str)
        print(f"[DEBUG] Unique Participant_IDs: {len(metadata['Participant_ID'].unique())}")
        print(f"[DEBUG] PHQ8_Binary value counts:\n{metadata['PHQ8_Binary'].value_counts().to_string()}")

        X = []
        y = []

        for idx, row in metadata.iterrows():
            pid = row["Participant_ID"]
            label = row["PHQ8_Binary"]
            print(f"[DEBUG] Processing Participant_ID: {pid}, Label: {label}")

            feature_file = os.path.join(features_dir, f"{pid}.csv")
            if not os.path.isfile(feature_file):
                print(f"[ERROR] Missing feature file for {pid}: {feature_file}")
                continue

            feat_vec = load_opensmile_arff_agg(feature_file)
            if feat_vec is None:
                print(f"[ERROR] Failed to extract features for {pid}")
                continue

            X.append(feat_vec)
            y.append(label)
            print(f"[DEBUG] Added features for {pid}, Feature vector length: {len(feat_vec)}")

        X = np.array(X)
        y = np.array(y)
        print(f"[DEBUG] Total loaded samples: {len(X)}")
        print(f"[DEBUG] Shape of X: {X.shape}")
        print(f"[DEBUG] Shape of y: {y.shape}")
        print(f"[DEBUG] Class distribution in y: {np.bincount(y.astype(int))}")

        if len(X) == 0:
            print("[ERROR] No data loaded, aborting training.")
            return

        # Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2, random_state=42
        )
        print(f"[DEBUG] Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        print(f"[DEBUG] Training labels distribution: {np.bincount(y_train.astype(int))}")
        print(f"[DEBUG] Test labels distribution: {np.bincount(y_test.astype(int))}")

        if len(np.unique(y_train)) < 2:
            print("[ERROR] Training data has fewer than two classes, cannot apply SMOTE.")
            return
        if len(X_train) == 0:
            print("[ERROR] No training data available after split.")
            return

        # Balance training data using SMOTE
        smote = SMOTE(sampling_strategy=1.0, random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"[DEBUG] Shape of X_train_res after SMOTE: {X_train_res.shape}")
        print(f"[DEBUG] Class distribution after SMOTE: {np.bincount(y_train_res.astype(int))}")

        # Train XGBoost classifier with scale_pos_weight
        scale_pos_weight = len(y_train_res[y_train_res == 0]) / len(y_train_res[y_train_res == 1])
        clf = XGBClassifier(scale_pos_weight=5, random_state=42)  # Increase weight for class 1
        print(f"[DEBUG] Starting model training with XGBoost...")
        clf.fit(X_train_res, y_train_res)
        print(f"[DEBUG] Model training completed.")

        # Feature selection using feature importance
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"[DEBUG] Top 10 feature indices by importance: {indices[:10]}")
        top_k = 30  # Reduced to top 30 features to reduce variability
        if top_k > X_train_res.shape[1]:
            top_k = X_train_res.shape[1]
        X_train_selected = X_train_res[:, indices[:top_k]]
        X_test_selected = X_test[:, indices[:top_k]]

        # Retrain with selected features
        clf.fit(X_train_selected, y_train_res)
        print(f"[DEBUG] Model retrained with top {top_k} features.")

        # Cross-validation with StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train_selected, y_train_res, cv=cv, scoring='f1')
        print(f"[DEBUG] Cross-validation F1 scores: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Evaluate with adjusted threshold
        y_pred_proba = clf.predict_proba(X_test_selected)[:, 1]
        threshold = 0.2  # Lowered further to improve recall for class 1
        y_pred = (y_pred_proba >= threshold).astype(int)
        print(f"[DEBUG] Shape of y_pred: {y_pred.shape}")
        print("[RESULTS] Classification Report:\n", classification_report(y_test, y_pred, digits=4))
        print(f"[RESULTS] Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"[RESULTS] F1 Score (class 1): {f1_score(y_test, y_pred):.4f}")

        # Save model
        joblib.dump(clf, "xgboost_compare16_arff.pkl")
        print("[INFO] Model saved to xgboost_compare16_arff.pkl")

    except Exception as e:
        print(f"[ERROR] An error occurred in main: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Fatal error: {str(e)}")
        raise