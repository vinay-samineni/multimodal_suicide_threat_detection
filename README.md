Multimodal Suicide Risk Detection using DAIC-WOZ Dataset
This repository contains the implementation of a multimodal deep learning framework for suicide risk detection using the DAIC-WOZ dataset. The model leverages textual (RoBERTa) and audio (ComParE16) features, integrating them using a cross-modal attention mechanism for improved classification performance.
🚀 Objective
To develop a system that classifies participants in the DAIC-WOZ dataset as at-risk or not-at-risk for suicide using multiple modalities: text and audio.
📂 Project Structure
multimodal_suicide_detection/
│
├── data/ # Processed feature files and labels
│   ├── text_features/ # RoBERTa embeddings (.pt files)
│   ├── audio_features/ # ComParE16 audio features (.csv or .pt)
│   └── labels.csv # Ground-truth PHQ8 scores or binary labels
│
├── models/ # Model definitions
│   ├── text_model.py # RoBERTa-based classifier
│   ├── audio_model.py # GRU-based classifier for audio features
│   └── fusion_model.py # Cross-modal attention fusion model
│
├── preprocess_text.py # Script to extract text features
├── preprocess_audio.py # Script to extract audio features
├── preprocess_labels.py # Script to preprocess labels
├── utils.py # Utility functions (loaders, metrics, etc.)
├── train.py # Main training script
├── evaluate.py # Evaluation script
├── config.yaml # Hyperparameters and settings
└── requirements.txt # Python dependencies

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


python preprocess_text.py
python preprocess_audio.py
python preprocess_labels.py


Train and evaluate the model:

python train.py
python evaluate.py

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
