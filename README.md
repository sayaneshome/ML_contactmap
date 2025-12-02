Overview

This project implements a 1D Convolutional Neural Network (CNN) to predict protein contact-map classes using tessellation-derived residue-pair features. Each sample represents a pair of amino acid residues, along with geometric and structural descriptors extracted from protein tessellation analysis. The model learns to classify each residue pair into one of 10 contact categories, reflecting spatial or functional relationships within the protein structure.

What the Code Does

Loads tessellation feature data
The script reads Step1_output.csv, which contains:

A residue-pair identifier (e.g., GLU-LEU)

12 numerical tessellation features

One-hot encoded labels for 10 contact-map classes

Encodes residue pairs
The categorical residue-pair strings are converted into numeric IDs using LabelEncoder so they can be used as model input.

Prepares the dataset

First 12 columns → input features

Remaining columns → contact-class labels

Reshapes inputs to (samples, 12, 1) to match the Conv1D architecture

Defines a 1D CNN classifier
The model consists of:

1 Conv1D layer (64 filters, kernel size 3)

Flatten layer

Four Dense layers with ReLU activation

Softmax output layer for 10 classes

Random hyperparameter search (100 trials)
For each iteration, the script randomly samples:

Learning rate

Momentum

Number of neurons in hidden layers

A new CNN is then trained for each sampled hyperparameter set.

Trains the CNN
Each model trains for:

10 epochs

Batch size 100

80/20 train–validation split

Evaluation utilities included
A confusion-matrix and per-class accuracy function is implemented but currently unused.

Purpose

This pipeline helps identify the best-performing hyperparameter configurations for predicting residue-residue contact categories from tessellation-derived geometric features. It serves as a machine-learning module within a broader protein-structure analysis pipeline based on tessellation concepts.
