Multimodal Suicide Risk Detection using DAIC-WOZ Dataset
This repository contains the implementation of a multimodal deep learning framework for suicide risk detection using the DAIC-WOZ dataset. The model leverages textual (RoBERTa) and audio (ComParE16) features, integrating them using a cross-modal attention mechanism for improved classification performance.
🚀 Objective
To develop a system that classifies participants in the DAIC-WOZ dataset as at-risk or not-at-risk for suicide using multiple modalities: text and audio.
📂 Project Structure
multimodal_suicide_detection/
│
├── config/ # Configuration files
│   └── params.yaml # Hyperparameters and settings
│
├── data/ # Processed feature files and splits
│   ├── features/
│   │   ├── text/ # RoBERTa embeddings (.pt files)
│   │   └── audio/ # ComParE16 audio features (.csv or .pt)
│   └── labels.csv # Ground-truth PHQ8 scores or binary labels
│
├── src/ # Source code
│   ├── models/ # Model definitions
│   │   ├── text_model.py # RoBERTa-based classifier
│   │   ├── audio_model.py # GRU-based classifier for audio features
│   │   └── fusion_model.py # Cross-modal attention fusion model
│   ├── preprocess/ # Preprocessing and feature extraction scripts
│   │   ├── extract_text_features.py
│   │   ├── extract_audio_features.py
│   │   └── preprocess_labels.py
│   ├── utils/ # Utility scripts (loaders, metrics, etc.)
│   ├── train.py # Main training script for fusion model
│   └── evaluate.py # Evaluation script
│
├── requirements.txt # Python dependencies
└── README.md # Project documentation

🧠 Modalities Used

Text: Combined prompt-response utterances from transcripts were encoded using RoBERTa.
Audio: ComParE16 features extracted using openSMILE toolkit.
Fusion: Attention-based module for combining modalities, followed by a classification head.

📊 Results



Modality
F1 Score



Text (RoBERTa)
0.76


Audio (ComParE16)
0.75


Fusion (Text + Audio)
0.79


⚙️ Setup

Clone this repository:

git clone https://github.com/vinay-samineni/multimodal_suicide_detection.git
cd multimodal_suicide_detection


Install dependencies:

pip install -r requirements.txt


Download and preprocess the DAIC-WOZ dataset (requires access approval from provider; it takes around 7 days for reply).

Run feature extraction scripts:


python src/preprocess/extract_text_features.py
python src/preprocess/extract_audio_features.py
python src/preprocess/preprocess_labels.py


Train and evaluate the model:

python src/train.py
python src/evaluate.py

📌 Highlights

Handles utterance-level alignment between modalities using timestamps.
Uses SMOTE to handle class imbalance in training.
Final model achieves competitive F1 score on DAIC-WOZ benchmark.

📜 Citation
If you use this code in your research, please cite:
@misc{vinay2025suicidedetection,
  title={Multimodal Suicide Detection using DAIC-WOZ Dataset},
  author={Vinay Samineni},
  year={2025},
  note={Undergraduate Research Project, RVR & JC College of Engineering}
}

🧑‍🏫 Guide
Dr. M. Sridevi, NIT Trichy
📬 Contact
For questions or collaborations, feel free to contact [saminenivinay999@gmail.com].
