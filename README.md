# Image Classification using Centralized, Federated, and Distributed Swarm Learning

## Overview

This repository contains the implementation and comparison of three machine learning paradigms: **Centralized Learning (Level 1), Federated Learning (Level 2)**, and **Distributed Swarm Learning (Level 3)**. The models are trained and tested on the CIFAR-10 dataset to demonstrate their unique characteristics and performance differences.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Folder Structure](#folder-structure)
5. [Running the Models](#running-the-models)
6. [Model Comparison](#model-comparison)
7. [Troubleshooting](#troubleshooting)
8. [Conclusions](#conclusions)
9. [Future Work](#future-work)

## Introduction

In the evolving landscape of artificial intelligence, centralized, federated, and distributed swarm learning paradigms are crucial for processing data and generating intelligence. Each model offers unique advantages in handling data privacy, scalability, and efficiency, making them indispensable tools in sectors like healthcare, finance, and autonomous systems.

## Requirements

- Python (version 3.6 or newer)
- Libraries:
  - TensorFlow
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - Pandas (optional)
- CUDA and cuDNN (for GPU acceleration)

## Installation

Install the required libraries using the following command:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pandas
```

## Folder Structure

Each model is stored in a separate folder named `Level1`, `Level2`, and `Level3`. Inside each folder, you will find:

- `level1.ipynb` (or `level2.ipynb`, `level3.ipynb`): Jupyter notebook containing simulations and outputs.
- `trained_model_1.h5` (or corresponding model files in other levels): Trained model file.
- `test1.py` (or corresponding test files in other levels): Python script for testing the model.

## Running the Models

### Reviewing Simulation Results

1. Navigate to the respective folder for the level you wish to review (e.g., `Level1`).
2. Open the Jupyter notebook using:

    ```bash
    jupyter notebook level1.ipynb
    ```

3. Run the notebook cells to view simulations, training processes, visualizations, and results.

### Testing the Trained Model

1. Ensure that the path to the trained model file (e.g., `trained_model_1.h5`) is correct in the Python script (`test1.py`).
2. Open a terminal or command prompt.
3. Navigate to the folder containing the test script.
4. Run the test script using Python:

    ```bash
    python test1.py
    ```

The script will load the trained model and evaluate its accuracy on the test set. The accuracy and other evaluation metrics will be displayed in the terminal.

## Model Comparison

### Level 1: Centralized Learning

- **Accuracy**: Highest and most stable.
- **Privacy**: Low, as all data is centralized.
- **Use Cases**: In-house predictive analytics, centralized healthcare research.

### Level 2: Federated Learning

- **Accuracy**: Variable, lower than centralized.
- **Privacy**: High, as data remains local.
- **Use Cases**: Mobile device usage, cross-institutional healthcare studies.

### Level 3: Distributed Swarm Learning with swarm intelligence

- **Accuracy**: Lower than centralized but better than federated.
- **Privacy**: High decentralization, robust against failures.
- **Use Cases**: IoT networks, autonomous vehicle fleets.

## Troubleshooting

- **Library Installation Issues**: Ensure correct Python and pip versions. Re-run the installation commands.
- **Model Loading Errors**: Check the path to the `.h5` model file in the test script.
- **Jupyter Notebook Issues**: Check Jupyter installation or try opening via Anaconda Navigator.

## Conclusions

The centralized model offers the highest accuracy but at the cost of data privacy and high computational resources. Federated learning enhances privacy and local data control but may struggle with model accuracy. Distributed swarm learning provides robust decentralization and fault tolerance, making it suitable for highly distributed systems.

## Future Work

Future research may consider hybrid approaches or further tuning of each model type. Additional considerations include computational and communication overhead, the need for specialized infrastructure, and adaptability to evolving data and conditions.
