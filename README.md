# ML From First Principles

A personal educational project implementing machine learning algorithms from scratch. This repository serves as a learning resource to understand the fundamental mathematics and implementation details behind popular ML algorithms.

## Features

### Currently Implemented

- **Linear Models**
  - **Linear Regression**: Ordinary least squares with analytical solution
  - **Ridge Regression**: L2-regularized linear regression with hyperparameter tuning
  - **Logistic Regression**: Binary classification with gradient descent optimization
  - **Linear SVM**: Support Vector Machine with hinge loss optimization

- **Core Utilities**
  - Base classes (`Regressor`, `Classifier`) for consistent API
  - Evaluation metrics: MSE, R² score, accuracy, precision, recall
  - Data splitting utilities: train/validation/test splits
  - Helper functions: sigmoid activation

### Planned Implementations

- **Neighbors**: k-Nearest Neighbors (k-NN)
- **Neural Networks**: Multi-layer perceptrons and backpropagation
- **Trees**: Decision trees and ensemble methods
- **Bayes**: Naive Bayes classifier

## Installation

### Requirements

- Python >= 3.8
- NumPy

### Setup
Clone the repository:
git clone https://github.com/yazanmashal03/ml-from-first
cd ml-from-first. Install the package:
pip install -e .Or install dependencies directly:
pip install numpy
