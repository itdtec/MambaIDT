# MambaITD: An Efficient Cross-Modal Mamba Network for Insider Threat Detection

MambaIDT is a pipeline for detecting insider threats by combining behavioral sequences and statistical features using a Mamba-inspired neural architecture.

## 📐 Architecture Diagram

The following diagram illustrates the overall structure of the **MambaITD framework**, including:

- Multi-modal input processing (sequence & statistical views)
- Dual Mamba-style encoders
- Gated fusion mechanism
- OTSU-based anomaly thresholding

📄 The full architecture is shown in the PDF below:

<img src="./assets/Framwork_Structrue.pdf" alt="main" style="zoom: 33%;" />

👉 [View Framework Structure Diagram (PDF)](./assets/Framwork_Structrue.pdf)


> If GitHub doesn't render the preview, click the link above to download or open the architecture PDF.

## 📁 Project Structure

```
MambaIDT/
├── preprocess/                      # Log preprocessing & feature engineering
│   ├── features/
│   │   ├── sequence/               # Sequence features (action_id_seq, time_diff_seq)
│   │   └── statistics/             # Statistical user behavior features
│   ├── output/                     # Intermediate processing outputs
│   ├── step1_log_split.py
│   ├── step2_log_merge.py
│   ├── step3_log_labeling.py
│   ├── step41_sequence_feature_engineering.py
│   ├── step42_stat_feature_engineering.py
│   └── utils_for_feature.py
│
├── src/                             # Core model and pipeline
│   ├── checkpoints/
│   │   └── mamba_itd_best.pth      # Saved model checkpoint
│   ├── config.yaml                 # Model and experiment configuration
│   ├── config.py                   # YAML parser for config
│   ├── dataset.py                  # Custom dataset class for sequence + stat features
│   ├── encoder.py                  # Mamba-style encoder block
│   ├── model.py                    # Main model architecture (MambaITD)
│   ├── training.py                 # Training logic (loss, optimizer, validation)
│   ├── run_experiment.py          # Entry point for training
│   ├── utils_.py                   # Helper functions (split, metrics, etc.)
│   └── __init__.py
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---
## 📂 Dataset Description

This project uses the [CERT Insider Threat Dataset (R4.2,R5.2)](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099), a publicly available dataset developed by Carnegie Mellon University's Software Engineering Institute (SEI).

### 📌 Overview

The CERT R4.2, R5.2, dataset simulates real-world enterprise behavior from thousands of users over multiple months. It includes both benign and malicious behaviors such as data theft, sabotage, and policy violations.

**Key modalities:**

- `logon.csv` – User logon and logoff events  
- `device.csv` – Removable media usage  
- `http.csv` – Web browsing activities  
- `file.csv` – File access events  
- `email.csv` – Internal and external email traffic  

Each log includes timestamps, user IDs, device IDs, and activity metadata.


## 🔧 Installation

```bash
# Recommended: set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Preprocessing Pipeline

Prepare your CERT-like logs and follow these steps:

```bash
# Split logs by user/date
python preprocess/step1_log_split.py --config config.yaml

# Merge different source logs
python preprocess/step2_log_merge.py --config config.yaml

# Add labels based on threat scenarios
python preprocess/step3_log_labeling.py --config config.yaml

# Extract behavioral sequence features
python preprocess/step41_sequence_feature_engineering.py --config config.yaml

# Extract statistical features
python preprocess/step42_stat_feature_engineering.py --config config.yaml
```

Output will be saved under `preprocess/features/sequence/` and `statistics/`.

---

## 🧠 Model Training

```bash
python src/run_experiment.py  --config config.yaml
```

- Loads sequence/stat features from `preprocess/features/`
- Trains the MambaITD model
- Applies OTSU thresholding for unsupervised decision boundary
- Saves best model to `src/checkpoints/mamba_itd_best.pth`

---

## 🧠 Model Description

- Sequence modeling via Mamba-style encoder layers
- Feature fusion using a gated mechanism between behavior and stat features
- Sigmoid score output per session
- OTSU-based thresholding (no need for manually-tuned threshold)


