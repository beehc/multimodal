# 📋 Final Project Specification: Multi-Modal Emergency Triage System (v7.0)

---

## 1. Project Overview

### 1.1 Objective

Develop an end-to-end deep learning pipeline for intelligent emergency department triage prediction, leveraging multi-modal data fusion (Text + ECG + Tabular) to classify patients into 3 urgency levels.

### 1.2 Task Definition

- **Task Type**: 3-Class Classification
- **Target Classes**:

| Original Acuity | Target Label | Clinical Meaning     |
| --------------- | ------------ | -------------------- |
| 1               | 0            | Critical (Immediate) |
| 2               | 1            | Urgent (Emergency)   |
| 3               | 2            | Stable (Less Urgent) |

### 1.3 Data Sources & Dimensions

| Modality               | Data Type                 | Raw Shape               | Description                                               |
| ---------------------- | ------------------------- | ----------------------- | --------------------------------------------------------- |
| Chief Complaint (Text) | Unstructured NLP          | (5899, variable_length) | Patient-reported symptoms, pain characteristics, duration |
| 12-Lead ECG            | Multi-channel Time Series | (5899, 12, 5000)        | Cardiac electrical activity at 500Hz sampling rate        |
| Vital Signs (Tabular)  | Structured Numeric        | (5899, 9)               | Temperature, HR, RR, SpO2, SBP, DBP, Pain, Gender, Age    |

### 1.4 Key Technical Challenges

1. **Modal Heterogeneity**: Bridging semantic gap between unstructured text, physiological signals, and structured numerics
2. **Data Synchronization**: Maintaining 1:1 correspondence after filtering across CSV and NPY files
3. **Class Imbalance**: Handling skewed distribution typical in emergency triage datasets
4. **Clinical Text Noise**: Processing abbreviations, typos, and non-standard medical terminology
5. **Signal Quality**: Denoising ECG artifacts from baseline wander and powerline interference

---

## 2. Data Preprocessing Pipeline

### 2.1 Dataset Filtering & Alignment

**CRITICAL REQUIREMENT: Modal Synchronization**

**Step 1: Class Filtering**

- Filter the CSV file to retain only rows where acuity is in the set {1, 2, 3}
- Record the original row indices of retained samples

**Step 2: ECG Array Alignment**

- Load the ECG numpy array with shape (5899, 12, 5000)
- Use the recorded indices to slice corresponding ECG signals
- Validation Check: Assert that the number of filtered CSV rows equals the number of filtered ECG samples

**Step 3: Label Remapping**

- Original acuity 1 maps to target label 0 (Critical)
- Original acuity 2 maps to target label 1 (Urgent)
- Original acuity 3 maps to target label 2 (Stable)

### 2.2 Tabular Data Cleaning

#### 2.2.1 Temperature Normalization Logic

- If temperature value is between 25.0 and 45.0, apply Celsius-to-Fahrenheit conversion (multiply by 1.8 and add 32)
- If temperature value is below 25.0 or above 115.0, set to missing (outlier removal)

#### 2.2.2 Pain Score Cleaning

- **Problem**: Pain column contains mixed types including numeric strings and text descriptions such as "unable", "ok", "uta", "leg pain"
- **Solution**: Convert all non-numeric strings to missing values using coercion

#### 2.2.3 Missing Value Imputation

- **Strategy**: Median imputation for all 9 numeric features
- **Rationale**: Median is robust to outliers common in emergency vital signs

#### 2.2.4 Feature Scaling

- **Method**: Z-score standardization
- **Output**: All features transformed to mean=0, standard deviation=1

### 2.3 Text Data Cleaning (Chief Complaints)

#### 2.3.1 Noise Removal Pipeline

1. Replace underscores with spaces
2. Collapse multiple whitespaces into single space
3. Remove special characters except alphanumeric and basic punctuation
4. Convert to lowercase

#### 2.3.2 Medical Abbreviation Expansion

| Abbreviation | Expansion                |
| ------------ | ------------------------ |
| sob          | shortness of breath      |
| abd          | abdominal                |
| cp           | chest pain               |
| s/p          | status post              |
| n/v/d        | nausea vomiting diarrhea |
| n/v          | nausea vomiting          |
| pna          | pneumonia                |
| loc          | loss of consciousness    |
| htn          | hypertension             |
| dm           | diabetes mellitus        |
| cva          | cerebrovascular accident |
| mi           | myocardial infarction    |
| chf          | congestive heart failure |
| etoh         | alcohol                  |
| ams          | altered mental status    |
| fx           | fracture                 |
| ha           | headache                 |
| r/o          | rule out                 |
| w/           | with                     |
| pt           | patient                  |

#### 2.3.3 Tokenization Configuration

- **Model**: Bio-ClinicalBERT from emilyalsentzer
- **Maximum Length**: 128 tokens
- **Padding**: Pad to maximum length
- **Truncation**: Enabled

### 2.4 ECG Signal Preprocessing

#### 2.4.1 Digital Filter Specifications

| Filter Type             | Cutoff Frequency | Order | Purpose                                           |
| ----------------------- | ---------------- | ----- | ------------------------------------------------- |
| High-pass (Butterworth) | 0.5 Hz           | 4     | Remove baseline wander from respiratory artifacts |
| Low-pass (Butterworth)  | 50 Hz            | 4     | Remove powerline interference and EMG noise       |

**Implementation**: Apply zero-phase filtering using forward-backward filter application

#### 2.4.2 Amplitude Normalization

- **Method**: Per-lead Z-score normalization
- **Rationale**: Ensures amplitude consistency across patients with different electrode impedances

---

## 3. Class Imbalance Handling Strategy

### 3.1 Weighted Random Sampling

- Calculate inverse frequency weights for each class
- Apply weighted random sampling during training to ensure each batch has roughly equal representation of all three classes
- Use replacement-based sampling to achieve balanced mini-batches

### 3.2 Focal Loss Implementation

- **Focusing Parameter (gamma)**: 2.0
- **Class Balancing Factor (alpha)**: 0.25
- **Purpose**: Down-weight easy examples and focus training on hard misclassified samples, particularly important for distinguishing between adjacent urgency levels

---

## 4. Model Architecture Specification

### 4.1 Architecture Overview

The model consists of three parallel encoding pathways for each modality, followed by a sophisticated fusion mechanism and classification head.

**Input Layer**:

- Text Input: Batch size × 128 tokens
- ECG Input: Batch size × 12 leads × 5000 samples
- Tabular Input: Batch size × 9 features

**Encoding Layer**:

- Text Branch: Bio-ClinicalBERT producing 768-dimensional features, projected to 256 dimensions
- ECG Branch: 1D-ResNet with 8 residual blocks producing 256-dimensional features
- Tabular Branch: 3-layer MLP producing 64-dimensional features

**Fusion Layer**:

- Bidirectional Cross-Attention between Text and ECG features producing 512-dimensional output
- Gated Multimodal Unit combining fused features with tabular features
- Final concatenated representation of 576 dimensions

**Output Layer**:

- MLP classifier mapping 576 dimensions to 3 class logits
- Softmax activation for probability distribution

### 4.2 Component Specifications

#### 4.2.1 Text Encoder: Bio-ClinicalBERT

- **Pretrained Model**: Bio-ClinicalBERT from emilyalsentzer
- **Architecture**: 12-layer Transformer with 768 hidden dimensions
- **Fine-tuning Strategy**:
  - Freeze: Layers 0-8 (embeddings plus first 9 transformer layers)
  - Train: Layers 9-11 (last 3 transformer layers)
- **Output**: CLS token embedding projected from 768 to 256 dimensions

#### 4.2.2 ECG Encoder: 1D-ResNet

- **Input Shape**: Batch × 12 × 5000
- **Initial Convolution**: 12 input channels to 64 output channels, kernel size 15, stride 2
- **Residual Block Configuration**:

| Block Group | Input Channels | Output Channels | Stride |
| ----------- | -------------- | --------------- | ------ |
| Blocks 1-2  | 64             | 64              | 1      |
| Blocks 3-4  | 64             | 128             | 2      |
| Blocks 5-6  | 128            | 256             | 2      |
| Blocks 7-8  | 256            | 256             | 2      |

- **Kernel Size**: 15 (designed to capture P-QRS-T complex spanning approximately 200ms at 500Hz)
- **Final Layer**: Adaptive average pooling followed by flattening to produce 256-dimensional output

#### 4.2.3 Tabular Encoder: MLP

- **Architecture**: Three-layer network
- **Layer Configuration**: Input 9 → Hidden 64 → Hidden 64 → Output 64
- **Activation**: ReLU with Batch Normalization
- **Regularization**: Dropout rate 0.3 between layers

#### 4.2.4 Bidirectional Cross-Attention Module

- **Attention Type**: Multi-Head Scaled Dot-Product Attention
- **Configuration**:
  - Number of Heads: 8
  - Head Dimension: 32
  - Total Dimension: 256

**Path A (Text-queries-ECG)**:

- Text features serve as Query
- ECG features serve as Key and Value
- Purpose: Contextualize ECG signals through symptom descriptions

**Path B (ECG-queries-Text)**:

- ECG features serve as Query
- Text features serve as Key and Value
- Purpose: Verify symptom descriptions through physiological signals

**Output**: Concatenation of both attention outputs producing 512-dimensional representation

#### 4.2.5 Gated Multimodal Unit (GMU)

- **Input**: Fused features (512 dimensions) and tabular features (64 dimensions)
- **Projection**: Both inputs projected to 256 dimensions
- **Gate Mechanism**: Sigmoid-activated learnable gate determining fusion ratio
- **Output**: Gated combination concatenated with original features producing 576 dimensions

#### 4.2.6 Classification Head

- **Architecture**: Three-layer MLP
- **Layer Configuration**: 576 → 256 → 128 → 3
- **Activation**: ReLU with Batch Normalization
- **Regularization**: Dropout rates of 0.5 and 0.3 for successive layers

### 4.3 Parameter Count Estimation

| Module          | Core Components  | Parameters  | Key Configuration     |
| --------------- | ---------------- | ----------- | --------------------- |
| ECG Encoder     | Custom 1D-ResNet | ~2.5M       | kernel=15, 8 blocks   |
| Text Encoder    | Bio-ClinicalBERT | ~110M       | hidden=768, 12 layers |
| Text Projection | Linear Layer     | ~200K       | 768→256               |
| Tabular Encoder | 3-Layer MLP      | ~15K        | 9→64→64               |
| Cross-Attention | Bi-directional   | ~530K       | heads=8, dim=256      |
| GMU             | Gating Network   | ~400K       | sigmoid gate          |
| Classifier      | MLP Head         | ~180K       | 576→256→128→3         |
| **Total**       | End-to-End Model | **~113.8M** | -                     |

---

## 5. Architecture Optimization Recommendations

Based on current advances in deep learning and multi-modal fusion best practices, this section presents architecture optimization recommendations for enhanced performance.

### 5.1 Current Architecture Potential Issues Analysis

| Issue Category                  | Specific Problem                                             | Impact                                                | Severity   |
| ------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------- | ---------- |
| **ECG Encoder**                 | 1D-ResNet has limited capacity for global dependency modeling in long sequences | May miss cross-lead arrhythmia patterns               | Medium     |
| **Feature Dimension Imbalance** | Text(256) + ECG(256) significantly larger than Tabular(64)   | Tabular feature contribution may be diluted           | Medium     |
| **Fusion Timing**               | Cross-Attention only between Text-ECG, Tabular concatenated late | Tabular data cannot guide Text/ECG feature extraction | Medium     |
| **Position Information Loss**   | ECG encoding loses temporal position information             | Difficult to localize specific abnormal moments       | Low-Medium |
| **Single CLS Representation**   | BERT uses only CLS token                                     | May lose fine-grained symptom information             | Low        |
| **GMU Gating Granularity**      | Global single gate controls fusion ratio                     | Cannot achieve feature-level fine-grained fusion      | Low        |

### 5.2 Recommended Optimization Approaches

#### 5.2.1 ECG Encoder Enhancement: Hybrid CNN-Transformer

**Problem**: Pure CNN architecture struggles to capture long-range dependencies in ECG signals such as periodic patterns in arrhythmias

**Optimization Approach**: Adopt hybrid architecture adding lightweight Transformer layers after ResNet

**Architecture Description**:

- Input: Batch × 12 × 5000
- Stage 1: 1D-ResNet with 4 blocks for local feature extraction, output shape Batch × 256 × 312
- Stage 2: Learnable 1D positional embeddings added to feature sequence
- Stage 3: 2-layer Transformer encoder with 4 attention heads for global temporal modeling
- Stage 4: Attentive pooling layer with learned attention weights over temporal dimension
- Output: Batch × 256

**Configuration Parameters**:

- ResNet portion: 4 residual blocks (reduced from 8), kernel size 15
- Transformer portion: 2 layers, 4 heads, hidden dimension 256
- Positional Encoding: Learnable position embeddings
- Attentive Pooling: Attention-weighted temporal aggregation

**Expected Benefits**: Better capture of heartbeat periodicity patterns and cross-lead correlations

#### 5.2.2 Text Encoding Enhancement: Multi-Granularity Representation

**Problem**: Using only CLS token may lose fine-grained information in symptom descriptions

**Optimization Approach**: Combine CLS token with attention-weighted representation of key tokens

**Architecture Description**:

- Extract CLS token embedding (768 dimensions)
- Extract all other token embeddings (127 × 768)
- Apply learnable attention weights to token embeddings
- Compute weighted sum of token representations
- Concatenate CLS and weighted representations
- Project concatenated features to 256 dimensions

**Expected Benefits**: Preserve key symptom word semantic information, improve compound symptom recognition

#### 5.2.3 Feature Dimension Balancing: Tabular Feature Enhancement

**Problem**: Tabular feature dimension (64) is much smaller than other modalities (256), contribution may be diluted

**Optimization Approach**: Enhance tabular encoder with feature interaction

**Architecture Description**:

- Input: 9 original features
- Feature Interaction Layer: Generate pairwise feature interactions
  - Examples: heart rate × respiratory rate (cardiopulmonary coupling), systolic BP minus diastolic BP (pulse pressure), temperature × heart rate (fever-tachycardia association)
  - Total: 9 original features + 36 pairwise interactions = 45 features
- Deep MLP with Residual: 45 → 128 → 128 → 128 with skip connections
- Output: 128 dimensions (doubled from original 64)

**Expected Benefits**: Better model clinical associations between vital signs such as pulse pressure and shock risk

#### 5.2.4 Early Fusion Enhancement: Tabular-Guided Attention

**Problem**: Tabular data only concatenated late, cannot guide Text/ECG feature extraction

**Optimization Approach**: Introduce tabular-conditioned attention bias

**Architecture Description**:

- Tabular features generate modulation factors through condition generators
- Text query modulated by tabular-derived scale factor before cross-attention
- ECG query similarly modulated by separate tabular-derived scale factor
- Standard cross-attention proceeds with conditioned queries

**Clinical Significance**: For example, when tabular data shows high heart rate combined with low oxygen saturation, the system will pay more attention to respiratory-related descriptions in text and tachycardia features in ECG

#### 5.2.5 Fine-Grained Gating Fusion: Channel-wise GMU

**Problem**: Current GMU uses single global gate, fusion granularity is coarse

**Optimization Approach**: Implement channel-level gating

**Architecture Description**:

- Generate separate gate vectors for fused features and tabular features
- Each channel independently determines fusion weight
- Apply element-wise gating to each feature dimension
- Concatenate gated features for final representation

**Expected Benefits**: Allow model to adopt different fusion strategies across different feature dimensions

#### 5.2.6 Auxiliary Loss: Multi-Task Learning

**Problem**: Single-task training may result in insufficient training of modality-specific encoders

**Optimization Approach**: Add modality-specific auxiliary classification heads

**Architecture Description**:

- Text encoder output connects to auxiliary classifier 1
- ECG encoder output connects to auxiliary classifier 2
- Tabular encoder output connects to auxiliary classifier 3
- Main fusion output connects to primary classifier
- Total Loss = Main Loss + auxiliary weight × (Text Aux Loss + ECG Aux Loss + Tabular Aux Loss)
- Recommended auxiliary weight: 0.1

**Expected Benefits**:

- Ensure each modality encoder learns discriminative representations
- Provide implicit regularization effect
- Enable analysis of each modality's independent prediction capability

### 5.3 Optimization Priority and Implementation Recommendations

| Priority | Optimization Item           | Implementation Difficulty | Expected Benefit | Recommendation                                             |
| -------- | --------------------------- | ------------------------- | ---------------- | ---------------------------------------------------------- |
| **P0**   | Multi-Task Auxiliary Loss   | Low                       | Medium-High      | **Must Implement** - Low cost with clear benefits          |
| **P1**   | Tabular Feature Enhancement | Low                       | Medium           | **Strongly Recommended** - Fully utilize clinical features |
| **P1**   | Channel-wise GMU            | Low                       | Medium           | **Strongly Recommended** - Small change with good effect   |
| **P2**   | Multi-Granularity Text      | Medium                    | Medium           | Recommended - Improve symptom recognition                  |
| **P2**   | Tabular-Guided Attention    | Medium                    | Medium           | Recommended - Clear clinical significance                  |
| **P3**   | Hybrid CNN-Transformer ECG  | High                      | Medium-High      | Optional - Requires additional tuning                      |

### 5.4 Optimized Architecture Parameter Estimation

| Module                | Original    | Optimized   | Delta                               |
| --------------------- | ----------- | ----------- | ----------------------------------- |
| ECG Encoder           | ~2.5M       | ~3.2M       | +0.7M (Transformer layers)          |
| Text Encoder          | ~110.2M     | ~110.8M     | +0.6M (Multi-granularity attention) |
| Tabular Encoder       | ~15K        | ~50K        | +35K (Feature interaction)          |
| Cross-Attention       | ~530K       | ~650K       | +120K (Conditioning)                |
| GMU                   | ~400K       | ~500K       | +100K (Channel-wise)                |
| Auxiliary Classifiers | 0           | ~100K       | +100K (3 small heads)               |
| Main Classifier       | ~180K       | ~200K       | +20K (Larger input)                 |
| **Total**             | **~113.8M** | **~115.5M** | **+1.7M (+1.5%)**                   |

---

## 6. Baseline Models for Comparative Experiments

To validate the effectiveness of the proposed multi-modal fusion architecture, the following baseline models must be implemented for comparative experiments.

### 6.1 Baseline Models Overview

| Baseline ID | Model Name                 | Modalities Used                             | Fusion Strategy         | Purpose                                        |
| ----------- | -------------------------- | ------------------------------------------- | ----------------------- | ---------------------------------------------- |
| B1          | Tabular-Only MLP           | Tabular                                     | None                    | Establish traditional feature baseline         |
| B2          | Text-Only BERT             | Text                                        | None                    | Establish text modality performance ceiling    |
| B3          | ECG-Only ResNet            | ECG                                         | None                    | Establish ECG modality performance ceiling     |
| B4          | Early Fusion (Concat)      | All                                         | Early Concatenation     | Baseline for simple fusion approach            |
| B5          | Late Fusion (Ensemble)     | All                                         | Decision-Level Ensemble | Compare feature-level vs decision-level fusion |
| B6          | Attention Fusion (w/o GMU) | All                                         | Cross-Attention Only    | Ablate GMU module contribution                 |
| B7          | Traditional ML (XGBoost)   | Tabular + Text (TF-IDF) + ECG (handcrafted) | Feature Concatenation   | Compare deep learning vs traditional methods   |

### 6.2 Baseline Model Detailed Specifications

#### 6.2.1 B1: Tabular-Only MLP

**Architecture Description**:

- Input: 9 tabular features
- Hidden layers: 128 → 256 → 128 neurons with Batch Normalization and ReLU activation
- Dropout: 0.3 between layers
- Output: 3 class logits with Softmax

**Parameter Count**: Approximately 50K

**Training Configuration**: Identical to main model (Focal Loss, AdamW optimizer, same epochs)

#### 6.2.2 B2: Text-Only BERT

**Architecture Description**:

- Input: 128 tokenized text tokens
- Encoder: Bio-ClinicalBERT with first 9 layers frozen
- CLS token projected through linear layer (768 → 256)
- Classification head: Linear layer with ReLU and Dropout 0.5
- Output: 3 class logits

**Parameter Count**: Approximately 110.5M

**Special Configuration**:

- Text preprocessing identical to main model
- Learning rate: 2e-5 (BERT standard)

#### 6.2.3 B3: ECG-Only ResNet

**Architecture Description**:

- Input: 12 leads × 5000 samples
- Encoder: 1D-ResNet with 8 blocks identical to main model ECG encoder
- Adaptive average pooling followed by flattening to 256 dimensions
- Classification head: 256 → 128 with ReLU and Dropout 0.5
- Output: 3 class logits

**Parameter Count**: Approximately 2.7M

**Special Configuration**:

- ECG preprocessing identical to main model
- Can use higher learning rate (1e-3)

#### 6.2.4 B4: Early Fusion (Simple Concatenation)

**Architecture Description**:

- Three separate encoders identical to main model:
  - Text Encoder producing 256 dimensions
  - ECG Encoder producing 256 dimensions
  - Tabular Encoder producing 64 dimensions
- Simple concatenation of all encoder outputs (576 dimensions)
- MLP classifier: 576 → 256 → 128 → 3

**Key Difference**: No Cross-Attention, No GMU - direct concatenation before classification

**Parameter Count**: Approximately 113M

**Purpose**: Validate the benefit of attention-based fusion mechanisms

#### 6.2.5 B5: Late Fusion (Decision-Level Ensemble)

**Architecture Description**:

- Three independent classification branches:
  - Text Encoder + Text Classifier → 3 class probabilities
  - ECG Encoder + ECG Classifier → 3 class probabilities
  - Tabular Encoder + Tabular Classifier → 3 class probabilities
- Each branch applies Softmax independently
- Final prediction: Weighted average of three probability distributions

**Ensemble Weight Strategies**:

- Option A: Equal weights (1/3, 1/3, 1/3)
- Option B: Validation-based weights (normalized by each modality's F1 score)
- Option C: Learnable weights (train a small weight network)

**Parameter Count**: Approximately 113M plus small weight parameters

**Purpose**: Compare feature-level fusion versus decision-level fusion

#### 6.2.6 B6: Cross-Attention Only (Ablation of GMU)

**Architecture Description**:

- Identical to main model except GMU module is removed
- Cross-Attention output (512 dimensions) directly concatenated with Tabular features (64 dimensions)
- Simple concatenation produces 576 dimensions
- MLP classifier identical to main model

**Parameter Count**: Approximately 113.4M

**Purpose**: Ablation experiment to validate GMU gating mechanism contribution

#### 6.2.7 B7: Traditional ML Baseline (XGBoost)

**Feature Engineering Description**:

Tabular Features (9 dimensions):

- Standardized vital signs after preprocessing

Text Features (500 dimensions):

- TF-IDF vectorization with maximum 500 features
- Unigrams and bigrams included
- English stop words removed

ECG Features (60 dimensions):

- Statistical features per lead:
  - Mean value per lead (12 features)
  - Standard deviation per lead (12 features)
  - Maximum value per lead (12 features)
  - Minimum value per lead (12 features)
  - Root mean square per lead (12 features)

Combined Feature Vector: 569 dimensions total

**Model Configuration**:

- Objective: Multi-class softmax probability
- Number of classes: 3
- Maximum depth: 6
- Learning rate: 0.1
- Number of estimators: 200
- Subsample ratio: 0.8
- Column subsample ratio: 0.8
- Class weight handling: Balanced
- Evaluation metric: Multi-class log loss
- Early stopping rounds: 20

**Purpose**:

- Validate deep learning advantages over traditional machine learning
- Provide highly interpretable baseline reference

### 6.3 Baseline Experiment Comparison Matrix

| Model                       | Text | ECG  | Tabular | Fusion Type        | Expected F1-Macro |
| --------------------------- | :--: | :--: | :-----: | ------------------ | ----------------- |
| B1: Tab-MLP                 |  ✗   |  ✗   |    ✓    | None               | 0.55-0.60         |
| B2: Text-BERT               |  ✓   |  ✗   |    ✗    | None               | 0.62-0.68         |
| B3: ECG-ResNet              |  ✗   |  ✓   |    ✗    | None               | 0.58-0.65         |
| B4: Early Concat            |  ✓   |  ✓   |    ✓    | Concatenation      | 0.68-0.72         |
| B5: Late Ensemble           |  ✓   |  ✓   |    ✓    | Decision-level     | 0.66-0.70         |
| B6: CrossAttn Only          |  ✓   |  ✓   |    ✓    | Attention (no GMU) | 0.70-0.74         |
| B7: XGBoost                 |  ✓   |  ✓   |    ✓    | Feature Concat     | 0.60-0.66         |
| **Proposed (v6)**           |  ✓   |  ✓   |    ✓    | CrossAttn + GMU    | **0.73-0.78**     |
| **Proposed (v7 Optimized)** |  ✓   |  ✓   |    ✓    | Enhanced Fusion    | **0.75-0.80**     |

### 6.4 Ablation Study Design

In addition to the baseline models above, the following ablation experiments must be conducted to validate each component's contribution:

| Experiment ID | Component Removed/Modified                     | Comparison Target                                     |
| ------------- | ---------------------------------------------- | ----------------------------------------------------- |
| A1            | Replace Focal Loss with Cross-Entropy          | Validate Focal Loss effectiveness for class imbalance |
| A2            | Remove Weighted Random Sampler                 | Validate sampling strategy contribution               |
| A3            | Freeze all BERT layers (all 12 layers)         | Validate BERT fine-tuning necessity                   |
| A4            | Unfreeze all BERT layers                       | Validate partial freezing regularization effect       |
| A5            | Unidirectional Cross-Attention (Text→ECG only) | Validate bidirectional attention necessity            |
| A6            | Reduce ResNet depth (4 blocks instead of 8)    | Validate ECG encoder depth impact                     |
| A7            | Remove ECG filtering preprocessing             | Validate signal preprocessing importance              |

### 6.5 Baseline Model File Deliverables

**Directory Structure**:

- baselines/models/ - Contains individual baseline model definitions
  - tabular_mlp.py (B1)
  - text_bert.py (B2)
  - ecg_resnet.py (B3)
  - early_fusion.py (B4)
  - late_fusion.py (B5)
  - crossattn_only.py (B6)
  - xgboost_baseline.py (B7)
- baselines/train_baselines.py - Unified baseline training script
- baselines/evaluate_baselines.py - Unified baseline evaluation script
- baselines/ablation_study.py - Ablation experiment runner
- baselines/results/ - Results storage
  - baseline_comparison.csv - All baseline metrics summary
  - baseline_comparison.png - Bar chart comparison
  - ablation_results.csv - Ablation experiment results

### 6.6 Comparative Experiment Visualization Requirements

#### 6.6.1 Model Performance Comparison Chart (model_comparison.png)

**Chart Type**: Grouped bar chart

**X-axis**: Model names (B1 through B7, Proposed, Proposed-Optimized)

**Y-axis**: Metric values ranging from 0 to 1

**Groupings**: Three bars per model representing F1-Macro, Accuracy, and AUC-ROC

**Styling**:

- Baseline models displayed in gray tones
- Proposed model displayed in blue
- Optimized model displayed in green
- Exact values annotated on top of each bar

#### 6.6.2 Modality Contribution Analysis Chart (modality_contribution.png)

**Chart Type**: Stacked bar chart or radar chart

**Content**: Display performance for different modality combinations

- Tabular Only
- Text Only
- ECG Only
- Tabular + Text
- Tabular + ECG
- Text + ECG
- All Modalities

#### 6.6.3 Ablation Study Results Table (ablation_heatmap.png)

**Chart Type**: Annotated heatmap table

**Rows**: Each ablation configuration (A1 through A7)

**Columns**: Each evaluation metric (F1-Macro, Accuracy, AUC, Critical Recall)

**Color Encoding**: Diverging color scale (red indicates decrease, white indicates baseline, green indicates increase)

**Annotations**: Delta values showing change from baseline (e.g., "-0.05", "+0.02")

---

## 7. Training Configuration

### 7.1 Data Split Strategy

- **Training Set**: 70% of data
- **Validation Set**: 15% of data
- **Test Set**: 15% of data
- **Method**: Stratified split to preserve class distribution
- **Random Seed**: 42 for reproducibility

### 7.2 Hyperparameters

| Parameter         | Value                               | Notes                           |
| ----------------- | ----------------------------------- | ------------------------------- |
| Batch Size        | 32                                  | Limited by GPU memory with BERT |
| Learning Rate     | 2e-5                                | Standard for BERT fine-tuning   |
| Optimizer         | AdamW                               | Weight decay = 0.01             |
| LR Scheduler      | Cosine Annealing with Warm Restarts | T_0=10, T_mult=2                |
| Maximum Epochs    | 100                                 | With early stopping             |
| Gradient Clipping | Maximum norm 1.0                    | Prevent exploding gradients     |

### 7.3 Early Stopping Configuration

- **Monitor Metric**: Validation F1-Macro Score
- **Patience**: 15 epochs
- **Mode**: Maximize
- **Minimum Delta**: 0.001
- **Restore Best Weights**: True

### 7.4 Regularization Techniques

1. **Dropout**: As specified in each module
2. **Weight Decay**: 0.01 (AdamW)
3. **Label Smoothing**: 0.1 (optional in loss function)
4. **BERT Freezing**: First 9 layers frozen

### 7.5 Multi-Task Learning Loss (If Adopting Optimization)

**Total Loss Computation**:

- Main loss: Focal loss on main classifier output
- Text auxiliary loss: Cross-entropy on text auxiliary head
- ECG auxiliary loss: Cross-entropy on ECG auxiliary head
- Tabular auxiliary loss: Cross-entropy on tabular auxiliary head
- Auxiliary weight (alpha): 0.1
- Total Loss = Main Loss + alpha × (Text Aux Loss + ECG Aux Loss + Tabular Aux Loss)

---

## 8. Evaluation Metrics & Visualization Requirements

### 8.1 Primary Metrics

| Metric          | Description                            | Priority                               |
| --------------- | -------------------------------------- | -------------------------------------- |
| F1-Macro        | Unweighted mean of per-class F1 scores | **Primary** (Early Stopping criterion) |
| Accuracy        | Correct predictions divided by total   | Secondary                              |
| AUC-ROC (Macro) | Average of per-class AUC values        | Secondary                              |

### 8.2 Per-Class Metrics

For each class (Critical, Urgent, Stable):

- Precision
- Recall (Sensitivity)
- F1-Score
- Specificity

### 8.3 Required Visualization Outputs

#### 8.3.1 Training Progress Plot (training_curves.png)

**Layout**: 2×1 subplot grid

**Subplot 1 - Loss Curves**:

- X-axis: Epoch number
- Y-axis: Loss value
- Lines: Training Loss (blue), Validation Loss (orange)
- Title: "Training and Validation Loss"

**Subplot 2 - F1-Macro Curves**:

- X-axis: Epoch number
- Y-axis: F1-Macro Score
- Lines: Training F1 (blue), Validation F1 (orange)
- Annotation: Mark best epoch with vertical dashed line
- Title: "Training and Validation F1-Macro"

#### 8.3.2 Confusion Matrix (confusion_matrix.png)

- **Type**: Normalized heatmap (row-normalized to show percentages)
- **Visualization Library**: Seaborn heatmap
- **Labels**: Critical, Urgent, Stable
- **Colormap**: Blues
- **Annotations**: Percentage values with 2 decimal places
- **Title**: "Normalized Confusion Matrix on Test Set"

#### 8.3.3 ROC-AUC Curves (roc_curves.png)

- **Type**: One-vs-Rest ROC curves
- **Lines**: One curve per class with different colors
- **Reference**: Include dashed diagonal reference line
- **Legend**: Show AUC value for each class
- **Title**: "Multi-class ROC Curves (One-vs-Rest)"

#### 8.3.4 Classification Report (classification_report.txt)

- Detailed table format showing Precision, Recall, F1-score, and Support for each class
- Include overall accuracy
- Include macro average and weighted average rows

#### 8.3.5 Baseline Comparison Chart (model_comparison.png)

- **Type**: Grouped bar chart
- **X-axis**: Model names (B1 through B7, Proposed, Proposed-Optimized)
- **Y-axis**: Metric values (0-1 scale)
- **Groups**: F1-Macro, Accuracy, AUC-ROC (3 bars per model)
- **Colors**: Baselines in gray tones, Proposed in blue, Optimized in green
- **Annotations**: Exact values displayed on top of each bar

#### 8.3.6 Ablation Study Heatmap (ablation_heatmap.png)

- **Type**: Annotated heatmap
- **Rows**: Ablation configurations (A1 through A7)
- **Columns**: Metrics (F1-Macro, Accuracy, AUC, Critical Recall)
- **Color Scale**: Diverging (red for decrease, white for baseline, green for increase)
- **Annotations**: Delta values (e.g., "-0.05", "+0.02")

---

## 9. Interpretability Module

### 9.1 SHAP Analysis

- **Method**: KernelSHAP or DeepSHAP for tabular feature importance
- **Output**: Bar plot showing contribution of each of 9 vital sign features
- **Aggregation**: Mean absolute SHAP values across test samples

### 9.2 Attention Map Visualization

- **Text Attention**: Highlight tokens with highest attention weights for sample predictions
- **ECG Attention**: Visualize which temporal segments the cross-attention focuses on
- **Output Format**: HTML or PNG with overlaid heatmaps

### 9.3 Modality Contribution Analysis

- **Method**: Based on auxiliary task head prediction comparison
- **Output**: Analysis of consistency/discrepancy between each modality's individual prediction and fused prediction
- **Visualization**:
  - Prediction confidence distribution per modality
  - Modality conflict case analysis (when different modalities give different predictions)

---

## 10. File Deliverables Structure

**Project Root Directory**:

**data/**

- raw/ - Contains original data files
  - triage_icu_subject_id_note.csv
  - ecg_merged.npy
- processed/ - Contains processed data files
  - train_data.pkl
  - val_data.pkl
  - test_data.pkl

**src/**

- preprocess.py - Data cleaning and preprocessing modules
- dataset.py - PyTorch Dataset and DataLoader definitions
- model.py - Main model architecture (v6 and v7 optimized versions)
- losses.py - Focal Loss and Multi-Task Loss implementations
- train.py - Training loop and validation logic
- evaluate.py - Test evaluation and metrics computation
- interpret.py - SHAP and attention visualization

**baselines/**

- models/ - Individual baseline model definitions
  - tabular_mlp.py (B1)
  - text_bert.py (B2)
  - ecg_resnet.py (B3)
  - early_fusion.py (B4)
  - late_fusion.py (B5)
  - crossattn_only.py (B6)
  - xgboost_baseline.py (B7)
- train_baselines.py - Unified baseline training
- evaluate_baselines.py - Unified baseline evaluation
- ablation_study.py - Ablation experiment runner

**configs/**

- config.yaml - Main model hyperparameters and paths
- baseline_configs.yaml - Baseline-specific configurations

**outputs/**

- checkpoints/ - Model checkpoint storage
  - best_model.pt
  - best_model_optimized.pt (v7 optimized version)
  - baselines/ - Baseline model checkpoints
- figures/ - Generated visualization files
  - training_curves.png
  - confusion_matrix.png
  - roc_curves.png
  - shap_importance.png
  - model_comparison.png
  - modality_contribution.png
  - ablation_heatmap.png
- reports/ - Generated report files
  - classification_report.txt
  - baseline_comparison.csv
  - ablation_results.csv

**Root Files**:

- requirements.txt - Python dependencies
- README.md - Project documentation

---

## 11. Module Specifications

### 11.1 Preprocessing Module (preprocess.py)

**TriageDataCleaner Class**:

- Method for temperature cleaning with Celsius-to-Fahrenheit conversion and outlier removal
- Method for pain score cleaning with non-numeric string handling
- Method for imputation using median strategy and standardization
- Method for generating interaction features (v7 optimization)

**TextCleaner Class**:

- Method for noise removal (underscores, multiple spaces, special characters)
- Method for medical abbreviation expansion
- Combined cleaning method

**ECGProcessor Class**:

- Method for bandpass filter application (high-pass 0.5Hz, low-pass 50Hz)
- Method for per-lead Z-score normalization
- Combined processing method
- Method for handcrafted feature extraction (for B7 baseline)

**Utility Function**:

- Function to filter by acuity classes and synchronize modality indices

### 11.2 Dataset Module (dataset.py)

**MultiModalTriageDataset Class**:

- Initialization with dataframe, ECG array, tokenizer, cleaners, and processors
- Option for interaction features (v7 optimization)
- Returns dictionary with input IDs, attention mask, ECG signal, tabular features, and label

**SingleModalDataset Class**:

- For single-modality baseline models
- Parameterized by modality type (text, ecg, or tabular)

**Utility Function**:

- Function to create WeightedRandomSampler from label distribution

### 11.3 Model Module (model.py)

**ECG Encoder Components**:

- ResidualBlock1D class for single residual block
- ResNet1D class for 8-block 1D ResNet
- TransformerEncoderLayer1D class for lightweight transformer layer (v7 optimization)
- HybridECGEncoder class for ResNet + Transformer hybrid (v7 optimization)

**Text Encoder Components**:

- MultiGranularityTextEncoder class combining CLS and attention-weighted pooling (v7 optimization)

**Tabular Encoder Components**:

- FeatureInteractionLayer class for pairwise feature interactions (v7 optimization)
- EnhancedTabularEncoder class with interaction and residual connections (v7 optimization)

**Fusion Components**:

- CrossAttentionModule class for bidirectional cross-attention
- TabularConditionedCrossAttention class with tabular-based query conditioning (v7 optimization)
- GatedMultimodalUnit class for learnable gating
- ChannelWiseGMU class for fine-grained channel-wise gating (v7 optimization)

**Main Models**:

- MultiModalTriageModel class (v6 baseline architecture)
- MultiModalTriageModelOptimized class with all optimization features (v7)

### 11.4 Loss Module (losses.py)

**FocalLoss Class**:

- Initialization with gamma and alpha parameters
- Forward method computing focal loss

**MultiTaskLoss Class** (v7 optimization):

- Initialization with main loss function and auxiliary weight
- Forward method combining main and auxiliary losses

### 11.5 Training Module (train.py)

**EarlyStopping Class**:

- Initialization with patience and minimum delta
- Call method returning whether to stop

**MetricTracker Class**:

- Method to update with new metrics
- Method to retrieve training history

**Functions**:

- Function for single epoch training with optional multi-task support
- Function for validation with optional multi-task support
- Function for plotting training curves
- Main function orchestrating full training pipeline

### 11.6 Evaluation Module (evaluate.py)

**Functions**:

- Function to load model from checkpoint with model class specification
- Function to evaluate on test set returning predictions, probabilities, and metrics
- Function to plot confusion matrix with class names
- Function to plot ROC curves with class names
- Function to generate classification report
- Main function for full evaluation pipeline

### 11.7 Baseline Training Module (baselines/train_baselines.py)

**Functions**:

- Function to train single baseline model given model name and configuration
- Function to train all baseline models sequentially
- Main function as entry point

### 11.8 Baseline Evaluation Module (baselines/evaluate_baselines.py)

**Functions**:

- Function to evaluate single baseline model
- Function to evaluate all baselines and return comparison dataframe
- Function to plot baseline comparison grouped bar chart
- Function to plot modality contribution analysis
- Function to generate comparison table for paper inclusion
- Main function as entry point

### 11.9 Ablation Study Module (baselines/ablation_study.py)

**Ablation Configurations Dictionary**:

- A1: Cross-entropy loss instead of Focal Loss
- A2: Weighted sampler disabled
- A3: All BERT layers frozen (12 layers)
- A4: All BERT layers unfrozen (0 layers frozen)
- A5: Unidirectional attention only
- A6: Shallower ECG encoder (4 blocks)
- A7: ECG preprocessing disabled

**Functions**:

- Function to run single ablation experiment
- Function to run all ablation experiments and return results dataframe
- Function to plot ablation heatmap
- Function to analyze ablation results and generate text summary
- Main function as entry point

---

## 12. Configuration File Specifications

### 12.1 Main Configuration (configs/config.yaml)

**Data Section**:

- Raw CSV file path
- Raw ECG numpy file path
- Processed data directory path
- Target classes list (1, 2, 3)

**Model Section**:

- Version specification (v6 or v7)
- BERT model name
- BERT freeze layers count
- ECG encoder configuration:
  - Type (hybrid or resnet)
  - ResNet blocks count
  - Transformer layers count (for v7)
  - Kernel size
- Tabular encoder configuration:
  - Interaction features flag
  - Hidden dimensions list
- Fusion configuration:
  - Conditioned attention flag
  - Channel-wise GMU flag
  - Attention heads count
- Auxiliary tasks flag
- Auxiliary weight value

**Training Section**:

- Batch size
- Learning rate
- Weight decay
- Maximum epochs
- Early stopping configuration:
  - Patience
  - Minimum delta
- Gradient clip value
- Scheduler configuration:
  - Type
  - T_0 and T_mult parameters

**Loss Section**:

- Loss type (focal)
- Gamma value
- Alpha value

**Evaluation Section**:

- Metrics list

**Output Section**:

- Checkpoint directory path
- Figures directory path
- Reports directory path

**Reproducibility Section**:

- Random seed value

### 12.2 Baseline Configuration (configs/baseline_configs.yaml)

**Shared Settings**:

- Batch size
- Maximum epochs
- Early stopping patience
- Random seed

**Individual Baseline Configurations**:

B1 Tabular MLP:

- Learning rate
- Hidden dimensions list
- Dropout rate

B2 Text BERT:

- Learning rate
- Freeze layers count
- Hidden dimension
- Dropout rate

B3 ECG ResNet:

- Learning rate
- ResNet blocks count
- Kernel size
- Dropout rate

B4 Early Fusion:

- Learning rate
- BERT freeze layers count

B5 Late Fusion:

- Learning rate
- Fusion weights strategy (learnable, equal, or validation-based)

B6 Cross-Attention Only:

- Learning rate
- Attention heads count

B7 XGBoost:

- Maximum depth
- Learning rate
- Number of estimators
- Subsample ratio
- Column subsample ratio
- TF-IDF maximum features
- ECG feature type

---

## 13. Dependencies

**Core Deep Learning**:

- PyTorch version 2.0.0 or higher
- Transformers version 4.30.0 or higher

**Data Processing**:

- NumPy version 1.24.0 or higher
- Pandas version 2.0.0 or higher
- SciPy version 1.11.0 or higher

**Machine Learning**:

- Scikit-learn version 1.3.0 or higher
- XGBoost version 1.7.0 or higher (for B7 baseline)

**Visualization**:

- Matplotlib version 3.7.0 or higher
- Seaborn version 0.12.0 or higher

**Interpretability**:

- SHAP version 0.42.0 or higher

**Utilities**:

- tqdm version 4.65.0 or higher
- PyYAML version 6.0 or higher

**Optional Experiment Tracking**:

- Weights & Biases (wandb) version 0.15.0 or higher
- TensorBoard version 2.13.0 or higher

---

## 14. Success Criteria

### 14.1 Primary Model Performance Targets

| Metric                | Minimum Threshold | Target | Stretch Goal |
| --------------------- | ----------------- | ------ | ------------ |
| Test F1-Macro         | ≥ 0.70            | ≥ 0.75 | ≥ 0.80       |
| Test Accuracy         | ≥ 0.72            | ≥ 0.78 | ≥ 0.82       |
| AUC-ROC (Macro)       | ≥ 0.80            | ≥ 0.85 | ≥ 0.90       |
| Critical Class Recall | ≥ 0.75            | ≥ 0.85 | ≥ 0.90       |

**Note**: Critical class (Class 0) recall is emphasized as missing critical patients has severe clinical consequences.

### 14.2 Baseline Comparison Success Criteria

| Comparison                    | Required Outcome            |
| ----------------------------- | --------------------------- |
| Proposed vs B1 (Tabular-Only) | F1-Macro improvement ≥ 0.15 |
| Proposed vs B2 (Text-Only)    | F1-Macro improvement ≥ 0.08 |
| Proposed vs B3 (ECG-Only)     | F1-Macro improvement ≥ 0.10 |
| Proposed vs B4 (Early Fusion) | F1-Macro improvement ≥ 0.03 |
| Proposed vs B5 (Late Fusion)  | F1-Macro improvement ≥ 0.05 |
| Proposed vs B6 (No GMU)       | F1-Macro improvement ≥ 0.02 |
| Proposed vs B7 (XGBoost)      | F1-Macro improvement ≥ 0.10 |

### 14.3 Optimization Improvement Targets (v7 vs v6)

| Metric             | Expected Improvement                 |
| ------------------ | ------------------------------------ |
| F1-Macro           | +0.02 to +0.05                       |
| Critical Recall    | +0.03 to +0.07                       |
| Training Stability | Reduced validation loss variance     |
| Convergence Speed  | Reach best performance 10-20% faster |

### 14.4 Ablation Study Expected Outcomes

| Ablation                      | Expected Impact             | Validates                     |
| ----------------------------- | --------------------------- | ----------------------------- |
| A1 (No Focal Loss)            | F1-Macro decrease 0.03-0.05 | Class imbalance handling      |
| A2 (No Weighted Sampler)      | F1-Macro decrease 0.02-0.04 | Sampling strategy             |
| A3 (Freeze all BERT)          | F1-Macro decrease 0.02-0.04 | Fine-tuning necessity         |
| A4 (Unfreeze all BERT)        | F1-Macro decrease 0.01-0.03 | Partial freeze regularization |
| A5 (Unidirectional Attention) | F1-Macro decrease 0.01-0.03 | Bidirectional attention value |
| A6 (Shallow ResNet)           | F1-Macro decrease 0.02-0.04 | ECG encoder depth             |
| A7 (No ECG preprocessing)     | F1-Macro decrease 0.03-0.06 | Signal quality importance     |

---

## 15. Experiment Execution Plan

### 15.1 Phase 1: Data Preparation (Days 1-2)

1. Implement preprocessing module with all cleaning functions
2. Filter dataset to 3 classes and align modalities
3. Verify data integrity with assertion checks
4. Generate stratified train/validation/test splits
5. Save processed data to pickle files

### 15.2 Phase 2: Baseline Implementation (Days 3-5)

1. Implement all baseline models (B1 through B7)
2. Create unified training script for baselines
3. Train all baselines with consistent hyperparameters
4. Evaluate and record baseline metrics

### 15.3 Phase 3: Main Model Development (Days 6-9)

1. Implement v6 architecture (original specification)
2. Train and validate main model
3. Implement v7 optimizations incrementally
4. Compare v6 versus v7 performance

### 15.4 Phase 4: Ablation Studies (Days 10-11)

1. Run all ablation experiments (A1 through A7)
2. Analyze component contributions
3. Document findings

### 15.5 Phase 5: Evaluation & Visualization (Days 12-14)

1. Generate all required visualizations
2. Create baseline comparison charts
3. Generate ablation heatmaps
4. Produce final classification reports
5. Write results summary

---

## 16. Additional Notes & Constraints

### 16.1 GPU Requirements

- Minimum: NVIDIA GPU with 16GB VRAM (for BERT + ResNet combined)
- Recommended: 24GB or more for larger batch sizes
- Baseline B7 (XGBoost): CPU sufficient

### 16.2 Reproducibility Requirements

- Set all random seeds to 42:
  - Python random module
  - NumPy random
  - PyTorch manual seed
  - CUDA manual seed for all GPUs
- Enable deterministic mode in cuDNN
- Disable cuDNN benchmark mode

### 16.3 Logging Requirements

- Use Python logging module with INFO level
- Log: epoch progress, loss values, metric scores, early stopping triggers
- Save logs to outputs/logs/ directory
- Include timestamps for all log entries

### 16.4 Error Handling Requirements

- Validate data shapes at each preprocessing step
- Check for NaN and Inf values in tensors during training
- Graceful handling of CUDA out-of-memory errors
- Checkpoint saving every N epochs for recovery

### 16.5 Code Quality Standards

- Type hints for all function signatures
- Docstrings for all classes and public methods
- Unit tests for preprocessing functions
- Configuration via YAML files, avoid hardcoded values

---

## 17. Glossary

| Term                    | Definition                                                   |
| ----------------------- | ------------------------------------------------------------ |
| **Acuity**              | Emergency severity level (1=Critical, 2=Urgent, 3=Stable)    |
| **Cross-Attention**     | Attention mechanism where Query comes from one modality and Key/Value from another |
| **GMU**                 | Gated Multimodal Unit - learnable gating mechanism for feature fusion |
| **Focal Loss**          | Loss function that down-weights easy examples to focus on hard cases |
| **Bio-ClinicalBERT**    | BERT model pretrained on clinical notes from MIMIC-III database |
| **ResNet**              | Residual Network with skip connections enabling deep learning |
| **SHAP**                | SHapley Additive exPlanations for model interpretability     |
| **F1-Macro**            | Unweighted average of F1 scores across all classes           |
| **AUC-ROC**             | Area Under the Receiver Operating Characteristic curve       |
| **Ablation Study**      | Systematic removal of components to measure their individual contribution |
| **TF-IDF**              | Term Frequency-Inverse Document Frequency for text vectorization |
| **Attentive Pooling**   | Attention-weighted aggregation of sequence features          |
| **Channel-wise Gating** | Fine-grained gating applied independently to each feature dimension |

---

## 18. Document Change Log

| Version | Date       | Changes                                                      |
| ------- | ---------- | ------------------------------------------------------------ |
| 6.0     | 2026-02-04 | Initial comprehensive specification                          |
| 7.0     | 2026-02-05 | Added architecture optimization recommendations (Section 5), baseline models specification (Section 6), ablation study design, updated file deliverables, experiment execution plan, configuration templates, converted to code-free English format |

---

*Document Version: 7.0*  
*Last Updated: 2026-02-05*  
*Author: beehc*  
*Status: Final Specification for Code Generation*
