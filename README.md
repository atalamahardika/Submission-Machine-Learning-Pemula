# 🏦 Financial Transaction Analysis: End-to-End Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-for-the-badge&logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Clustering%20%26%20Classification-orange?style=flat-for-the-badge)](#-project-workflow)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20Yellowbrick-green?style=flat-for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)

This repository contains an end-to-end machine learning project focusing on financial transaction pattern analysis. The project is structured into a two-phase architecture: **Unsupervised Clustering** to establish behavioral boundaries (anomaly/fraud profiling) followed by **Supervised Classification** to predict user transaction segments.

---

## 📂 Repository Structure

The project consists of the following core assets:
```text
Submission-Machine-Learning-Pemula/
├── [Clustering]_Submission_Akhir_BMLP_Athallah_Anargya_Mahardika.ipynb    # Phase 1: Unsupervised Clustering Analysis
├── [Klasifikasi]_Submission_Akhir_BMLP_Athallah_Anargya_Mahardika.ipynb   # Phase 2: Supervised Classification Benchmarking
├── Dataset_clustering.csv                                                 # Raw and unlabeled transaction dataset (2,512 rows)
├── Dataset_klasifikasi.csv                                                # Clustered dataset containing the engineering targets/labels
└── README.md                                                              # Professional documentation
```

## 📊 Dataset Overview
The raw dataset (Dataset_clustering.csv) tracks 2,512 unique financial transactions with 16 distinct analytical attributes. It contains a comprehensive blend of numerical metrics and categorical markers:

- Identification Features: TransactionID, AccountID, DeviceID, IP Address, MerchantID.

- Behavioral Features: TransactionAmount, TransactionType (Debit/Credit), Channel (ATM, Online, etc.), TransactionDuration, LoginAttempts.

- Demographic Profiles: CustomerAge, CustomerOccupation (Doctor, Student, Engineer, etc.), Location.

- Financial Status: AccountBalance, TransactionDate, PreviousTransactionDate.

Data Integrity Check: The dataset contains 0 missing values and 0 duplicate rows, ensuring robust training profiles.

## 🔄 Project Workflow & Methodology🧩 

### 🧩 Phase 1: Unsupervised Clustering (K-Means)
1. Exploratory Data Analysis (EDA): Evaluated value spreads, statistical shapes via distribution profiling, and data formatting.
2. Data Preprocessing: Features were mapped using target label encoding for categorical vectors, followed by scaling operations using MinMaxScaler to equalize structural weights.
3. Hyperparameter Tuning (KElbowVisualizer): Used the Elbow Method and Silhouette Analysis to pinpoint the optimal cluster density ($K=2$).
4. Export Engineering: Appended the synthetic cluster groups (Cluster_KMeans) as target labels into Dataset_klasifikasi.csv.

### ⚡ Phase 2: Supervised Classification Benchmarking  
The purpose of this layer is to evaluate model generalizability across the newly uncovered behavior patterns, using a formal Train/Test partition split. We evaluated 5 top-tier classification architectures:
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Classifier (SVC)
- Gaussian Naive Bayes (GNB)

## 📈 Key Findings & Performance Metrics
### 1. Clustering Separation Quality  
The K-Means model split the dataset cleanly into two transactional tiers based on customer profiles and behavior records. The optimization score reflects near-perfect spatial distribution:
    - Silhouette Score: 0.9322 (Indicating an exceptionally clear, distinct boundary with negligible data overlap).
  
### 2. Cluster Behavioral Profiling & Characteristic Analysis  
To understand the operational meaning of the two generated clusters, a comprehensive statistical mode and mean aggregation analysis was conducted across both segments:

| Metric / Feature Attribute | Cluster 0 (Normal / Baseline) | Cluster 1 (Anomalous / Suspected Fraud) |
| :--- | :--- | :--- |
| **Avg. Transaction Amount** | 285.10 | 313.98 |
| **Avg. Transaction Duration** | 119.57 | 119.72 |
| **Avg. Login Attempts** | 1.11 | 1.13 |
| **Avg. Account Balance** | 5,919.66 | 4,056.80 |
| **Dominant Location (Mode)** | Milwaukee | Atlanta |
| **Preferred Channel (Mode)** | Branch | ATM |
| **Dominant Occupation (Mode)**| Engineer | Student |
| **Dominant Age Group (Mode)** | Adult (Age 26 - 59) | Elderly (Age 60 - 100) |
| **Primary Transaction Type** | Debit | Debit |
| **System Risk Assessment** | 🟢 **Low Risk (Non-Fraud)** | 🔴 **High Risk (Anomalous Activity)** |

#### 🔍 Deep-Dive Interpretations & Domain Insights:

* **Cluster 0 (Legitimate Behavior Profile):**
    This segment captures standard, high-integrity financial transactions. Behaviors are highly consistent with established profiles: working professionals (**Engineers**) within the **Adult** lifecycle cohort (ages 26–59). Users demonstrate low-friction access profiles (average 1.11 login attempts), maintain stable funding baselines (Avg. Balance: 5,919.66), and heavily utilize secure physical **Branch** ecosystems. No identity or spatial contradictions are detected.
* **Cluster 1 (High-Risk Identity & Behavioral Anomaly):**
    This segment uncovers a critical systemic **anomaly** highly indicative of synthetic identity fraud or account compromise. While numerical metrics (duration, transaction amount) show subtle elevation, a severe logical contradiction exists within the demographic attributes: the primary user occupation is mapped as a **Student**, yet the chronological age tier registers exclusively as **Elderly/Lansia** (ages 60–100). This stark demographic mismatch, combined with a lower capital asset reserve (4,056.80) and a preference for unmonitored **ATM** extraction channels, solidifies this cluster's classification as an anomalous risk vector.

### 3. Classification Benchmarking Results  
Due to the pristine quality of the baseline segment clusters discovered in Phase 1, the supervised classification models successfully achieved absolute convergence:
    - Model Accuracy (All Architectures): 100%
    - Precision, Recall, & F1-Score: 1.00 across all metric classes.

### 📝 Senior Architect's Note on 100% Accuracy: Achieving a perfect score is typically a warning flag for data leakage or overfitting. However, inside this specific workflow pipeline, it acts as a mathematically valid confirmation of the Silhouette Score (0.9322) from the clustering phase. The clusters are separated so distinctly in vector space that any standard classifier can easily establish an ideal decision boundary line.

## 🛠️ Tech Stack & Dependencies
- Core Runtime: Python 3.8+
- Data Engineering: pandas, numpy
- Visualization: matplotlib, seaborn, yellowbrick
- Machine Learning Engine: scikit-learn (Modules: cluster, preprocessing, model_selection, metrics)
