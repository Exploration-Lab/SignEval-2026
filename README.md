# SignEval-2026
# EL-IITK @ SignEval 2026: BiLSTM and Transformer Based Alignment for Skeleton-Based Continuous Sign Language Recognition
<img width="855" height="479" alt="image" src="https://github.com/user-attachments/assets/223cfde1-a4b8-4000-b761-d1ca47a3bd2c" />

Figure 1: Overview of the proposed skeleton-based CSLR pipeline.  

**Challenge:** SignEval 2026 – Track 1 (Continuous Sign Language Recognition)  
**Dataset:** Isharah2000 (~30,000 videos, 2,000 unique Saudi sign language sentences, 18 signers)  

**Leaderboard Rank:**  
**3rd place** in Unseen Sentences sub-task (**WER 28.28%**)
**4th place** in Signer-Independent sub-task (**WER 7.14%**)  


---

Skeleton-based continuous sign language recognition (CSLR) provides a robust, privacy-preserving, and efficient alternative to video-based methods for real-world applications. We present a study for the SignEval 2026 Challenge (Track 1) on the Isharah2000 dataset.

We adopt a **two-stream GCN-based architecture** (skeleton + motion) with group-wise spatial modeling and short-term 1D convolutions. Our key contribution is **task-adaptive sequence modeling**: we compare **BiLSTM** and **Transformer** for long-term alignment and find that:

- **Transformer** excels on unseen sentences  
- **BiLSTM** (with expanded short-term receptive field) performs better on signer-independent recognition  

These task-specific architectural choices enable strong generalization and place our team (EL–IITK) among the top performers.

**Full code and pretrained models are released here.**

---

## ✨ Key Contributions

- Comprehensive analysis of design choices in skeleton-based CSLR with emphasis on **task-adaptive long-term modeling** (Transformer vs. BiLSTM)
- Task-specific optimal configurations:
  - Unseen Sentences → **Transformer + compact receptive field** (K3-P2-K3-P2)
  - Signer-Independent → **BiLSTM + expanded receptive field** (K3-K7-P2-K7-K3-P2)


---

## 🏆 Official Leaderboard Results 

### Signer-Independent Sub-task

| Rank | Team          | WER (%) |
|------|---------------|---------|
| 1    | ahemedmo10    | 5.68    |
| 2    | anhnamxtanh   | 5.70    |
| 3    | VIPL_SLP      | 6.37    |
| **4** | **EL-IITK (ours)** | **7.14** |

### Unseen Sentences Sub-task

| Rank | Team          | WER (%) |
|------|---------------|---------|
| 1    | anhnamxtanh   | 27.35   |
| 2    | VIPL_SLP      | 27.62   |
| **3** | **EL-IITK (ours)** | **28.28** |
| 4    | ahemedmo10    | 39.73   |

---

## 🏗️ Architecture Overview


Skeleton keypoint sequences → **GCN Encoder** (group-wise) → **1D CNN** (short-term) → **BiLSTM or Transformer** (long-term alignment) → Gloss sequence (CTC).


### Feature Extractor
- Multi-layer **group-wise GCNs** (CoSign-style)
- Separate adjacency matrices per group (body, L-hand, R-hand, face, mouth)

### Temporal Modeling (Task-Adaptive)
- Short-term: 1D convolutions with different receptive fields
- Long-term alignment module (chosen per sub-task):
  - **BiLSTM** (2-layer, hidden dim 1024) – best for signer-independent
  - **Transformer** (2-layer, 8 heads, hidden dim 1024) – best for unseen sentences

---
### 📚 Qualitative Results
<img width="1006" height="853" alt="image" src="https://github.com/user-attachments/assets/409ea1e8-bbc6-4dd1-a64f-dd0357bef488" />

Figure 2: Exact match prediction (Arabic gloss sequence).

<img width="920" height="867" alt="image" src="https://github.com/user-attachments/assets/3a116a14-2622-4f5d-b81e-584501e7b7d1" />

Figure 3: Model correctly captures core meaning (“craving pasta”) even when phrasing differs slightly from ground truth



### Setup & Usage Guide

Follow these steps to setup environment, prepare data, and run training/evaluation

#### 1. Install Dependencies

Install virtualenv (optional)

pip install virtualenv

Create virtual environment

python -m venv pose

Activate environment

Linux / Mac

source pose/bin/activate

Windows

pose\Scripts\activate

Install required libraries

pip install torch==1.13 torchvision==0.14 tqdm numpy==1.23.5 pandas opencv-python

Install CTCDecode

git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
cd ..

#### 2. Clone Repository
git clone https://github.com/Exploration-Lab/SignEval-2026.git
cd SignEval-2026
#### 3. Dataset Setup

Download the dataset TASK1[https://www.kaggle.com/datasets/gufransabri3/mslr-task1] and 
TASK2[https://www.kaggle.com/datasets/gufransabri3/mslr-task2]
 and place it in the ./datasets folder

SignEval-2026/
│── datasets/

Download the annotation [https://github.com/gufranSabri/Pose86K-CSLR-Isharah/tree/main/annotations_v2] and place it in:

./preprocess/mslr2025
4. Install sclite (for evaluation)

Install kaldi toolkit to get sclite from sctk

After installation, create a soft link:

mkdir ./software
ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite
5. Preprocess Dataset

Run preprocessing to generate:

gloss dictionary

dataset info

ground truth for evaluation

cd ./preprocess/mslr2025
python mslr_process.py
6. Training & Evaluation
Signer Independent
Train
python main.py --config ./configs/Double_Cosign_si.yaml --long_term_model bilstm

or

python main.py --config ./configs/Double_Cosign_si.yaml --long_term_model transformer
Test
python main.py --config ./configs/Double_Cosign_si.yaml \
--long_term_model bilstm \
--phase test \
--load-weights PATH_TO_PRETRAINED_MODEL

or

python main.py --config ./configs/Double_Cosign_si.yaml \
--long_term_model transformer \
--phase test \
--load-weights PATH_TO_PRETRAINED_MODEL
Unseen Sentences
Train
python main.py --config ./configs/Double_Cosign_us.yaml --long_term_model bilstm

or

python main.py --config ./configs/Double_Cosign_us.yaml --long_term_model transformer
Test
python main.py --config ./configs/Double_Cosign_us.yaml \
--phase test \
--long_term_model bilstm \
--load-weights PATH_TO_PRETRAINED_MODEL

or

python main.py --config ./configs/Double_Cosign_us.yaml \
--phase test \
--long_term_model transformer \
--load-weights PATH_TO_PRETRAINED_MODEL
7. Notes

Different tasks need different data augmentation strategies during training

Modify it in:

./datasets/skeleton_feeder.py  (line 207)

Important for:

signer independent setup

unseen sentence generalization

--long_term_model supports:

bilstm

transformer

Replace PATH_TO_PRETRAINED_MODEL with actual checkpoint path

For additional arguments / parameters:
./utils/paramters.py
