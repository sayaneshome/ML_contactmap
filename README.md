Protein Contact-Map Prediction from Tessellation Features

This project implements a 1D Convolutional Neural Network (CNN) to classify residue–residue contact-map categories using tessellation-derived geometric features. Each row in the dataset represents a pair of amino acids and its corresponding structural descriptors extracted from protein tessellation analysis. The model predicts one of 10 contact classes based on these features.

📂 Dataset Description

The input file Step1_output.csv contains:

Residue pair (e.g., GLU-LEU, ALA-GLY)

12 tessellation features per residue pair

One-hot encoded labels for the 10 contact-map classes

Rows are shuffled at runtime to remove bias.

🧬 Pipeline Overview
1. Residue-Pair Encoding

Residue-pair strings are converted into numeric IDs using LabelEncoder, enabling them to be used as model features.

2. Feature & Label Preparation

Features: first 12 numeric columns

Labels: remaining columns (10-class one-hot vectors)

Features are reshaped into (samples, 12, 1) for 1D convolution.

🧠 Model Architecture (1D CNN)
Input (12 × 1)
   ↓
Conv1D (64 filters, kernel size 3, ReLU)
   ↓
Flatten
   ↓
Dense (N neurons, ReLU)
Dense (N neurons, ReLU)
Dense (N neurons, ReLU)
Dense (N neurons, ReLU)
   ↓
Dense (10, Softmax)


Where N (number of neurons per dense layer) is randomly sampled for each run.

🎛 Hyperparameter Search

For each of 100 iterations, the script samples:

Learning rate: 0 → 0.2

Momentum: 0 → 1

Hidden neurons: 10 → 50

A new CNN is trained using these randomly chosen values.

🚀 Training

Optimizer: SGD with Nesterov momentum

Loss: Categorical crossentropy

Metrics: Accuracy, Precision

Epochs: 10

Batch size: 100

Validation split: 20%

Training time for each run is printed.

📊 Evaluation Utilities

A helper function (currently commented out) can compute:

Per-class accuracy

Confusion matrix

Overall accuracy

This can be activated for deeper analysis.

🎯 Purpose

This pipeline identifies optimal hyperparameter configurations for predicting protein contact-map categories based on tessellation-derived residue-pair features, forming a key component of a larger protein-structure analysis framework.
