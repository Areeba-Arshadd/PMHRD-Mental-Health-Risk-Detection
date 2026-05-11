PMHRD: Parallelized Mental Health Risk Detection

Depression and Self-Harm Identification from Social Media with Safety-Aware Evaluation

Overview

PMHRD (Parallelized Mental Health Risk Detection) is an end-to-end NLP system designed to detect and classify mental health risks from social media posts. It goes beyond traditional binary classification by introducing severity-aware multi-class prediction and safety-sensitive evaluation, making it more aligned with real-world clinical triage needs.

The system is built on a fine-tuned MentalBERT model and optimized using parameter-efficient fine-tuning (LoRA), mixed precision training (FP16), and parallel/distributed NLP pipelines for scalability.

Problem Statement

Existing mental health detection systems suffer from:

❌ Binary classification (ignores severity levels)
❌ Sequential NLP pipelines (poor scalability for real-time data)
❌ Symmetric evaluation metrics (ignore cost of false negatives in high-risk cases)
💡 Proposed Solution: PMHRD

PMHRD addresses these limitations through:

🔹 1. Multi-Class Severity Detection
4-class taxonomy:
No Risk
Mild Depression
Moderate/Severe Depression
Self-Harm / Suicidal Ideation

🔹 2. Parallelized NLP Pipeline
CPU-based multiprocessing for preprocessing & tokenization
GPU batched inference using PyTorch DataLoader
Up to 5.39× speedup in tokenization
Up to 2.58× speedup in inference

🔹 3. Safety-Aware Learning
Asymmetric weighted cross-entropy loss
Higher penalty for false negatives in high-risk classes
Significant reduction in High-Risk FNR

🔹 4. Parameter-Efficient Fine-Tuning
MentalBERT backbone (110M parameters)
LoRA (rank = 16)
Only 0.54% trainable parameters

Model Architecture

Base Model: MentalBERT (Reddit-pretrained BERT variant)
Fine-Tuning: LoRA adapters
Precision: FP16 mixed precision (AMP)
Classification Head: 4-class softmax layer

📊 Dataset

Dataset	Samples	Purpose

DS1 (Sarkar, 2024)	52,681	Training / Validation

DS2 (Namdari, 2023)	27,977	Training / Validation

DS3 (InfamousCoder, 2022)	7,731	Held-out binary test

Total training data: 79,044 samples

Test set: 7,905 samples (primary)

📈 Results

Overall Performance (PMHRD v4)
Macro-F1: 0.776
Risk-Sensitive F1: 0.744
High-Risk FNR: 0.161
Binary Accuracy (DS3): 91.92%
Binary F1: 0.920

Ablation Study

Metric	Asymmetric Loss	Symmetric Loss	Improvement
Macro-F1	0.776	0.684	+0.0917
Risk-Sensitive F1	0.744	0.588	+0.1556
High-Risk FNR	0.161	0.514	↓ 68.7%
⚙️ Performance Benchmarks
Parallelization Gains
Tokenization Speedup: 5.39×
Inference Speedup (best batch): 2.58×
Preprocessing throughput: 4,252 samples/sec
GPU Efficiency (Tesla T4)
Batch Size	Speedup	VRAM Usage

8	2.58×	531 MB
32	2.19×	801 MB
128	1.43×	1881 MB

 Key Features
 
✔ Multi-class mental health severity detection
✔ Reddit-trained domain-specific transformer (MentalBERT)
✔ LoRA-based efficient fine-tuning
✔ Parallel CPU-GPU pipeline
✔ Safety-aware asymmetric loss
✔ Real-time inference capability

System Pipeline

Data collection & label mapping
Parallel preprocessing (multiprocessing)
Parallel tokenization (ProcessPoolExecutor)
LoRA-based fine-tuning of MentalBERT
Batched GPU inference (FP16)
Safety-aware evaluation (F1, FNR, RS-F1)

Ethical Considerations
Not a diagnostic tool
Intended for early risk triage only
Requires human supervision in deployment
No PII stored or processed
Bias monitoring recommended for real-world use

Future Work
Multi-GPU distributed training
Multilingual mental health detection
Explainability (SHAP / Grad-CAM)
Retrieval-Augmented Generation (RAG) for support suggestions
Larger and more diverse datasets

References

Key foundations include:

MentalBERT (Ji et al., 2022)
BERT (Devlin et al., 2019)
LoRA (Hu et al., 2022)
Focal / cost-sensitive loss methods
Social media mental health studies (De Choudhury et al., 2013)
Authors

Hadia Khawar – National University of Computer and Emerging Sciences (FAST), Islamabad
Areeba Arshad – National University of Computer and Emerging Sciences (FAST), Islamabad
