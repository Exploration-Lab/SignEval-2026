# SignEval-2026
# EL-IITK @ SignEval 2026: BiLSTM and Transformer Based Alignment for Skeleton-Based Continuous Sign Language Recognition

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

<img width="855" height="479" alt="image" src="https://github.com/user-attachments/assets/223cfde1-a4b8-4000-b761-d1ca47a3bd2c" />

Figure 1: Overview of the proposed skeleton-based CSLR pipeline.  
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
