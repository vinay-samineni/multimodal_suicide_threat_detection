# Multimodal Suicide Risk Detection using DAIC-WOZ Dataset

This repository contains the implementation of a multimodal deep learning framework for suicide risk detection using the DAIC-WOZ dataset. The model leverages **textual (RoBERTa)** and **audio (ComParE16)** features, integrating them using a **cross-modal attention mechanism** for improved classification performance.

## 🚀 Objective
To develop a system that classifies participants in the DAIC-WOZ dataset as at-risk or not-at-risk for suicide using multiple modalities: **text** and **audio**.

## 📂 Project Structure
![Screenshot 2025-06-10 133945](https://github.com/user-attachments/assets/22b1ddab-b016-43f0-95a3-a824f4e52f1b)


## 🧠 Modalities Used

- **Text**: Combined prompt-response utterances from transcripts were encoded using RoBERTa.
- **Audio**: ComParE16 features extracted using openSMILE toolkit.
- **Fusion**: Attention-based module for combining modalities, followed by a classification head.

## 📊 Results

| Modality      | F1 Score |
|---------------|----------|
| Text (RoBERTa) | 0.76     |
| Audio (ComParE16) | 0.75     |
| Fusion (Text + Audio) | **0.79**     |

## ⚙️ Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/multimodal_suicide_detection.git
   cd multimodal_suicide_detection
   
2.Install dependencies:
pip install -r requirements.txt

3.Download and preprocess the DAIC-WOZ dataset (requires access approval it approximately takes around a week).

4.Run feature extraction scripts:
python scripts/extract_text_features.py
python scripts/extract_audio_features.py
python scripts/preprocess_labels.py

5.Train and test
python train.py
python test.py

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
For questions or collaborations, feel free to contact [y22cs163@rvrjc.ac.in].

---
Let me know if you want a version with badges, a license, dataset disclaimer, or deployment instructions.
