# BUS-BRA Breast Ultrasound Robustness Study

## Project
Robustness and Reliability of Breast Ultrasound Classification
Under Lesion Boundary Uncertainty

## Team
Mostafa Bdeir and Hadi Fardan

## Overview
This project investigates how the performance of breast
ultrasound classifiers changes when lesion segmentation
boundaries are slightly modified.

We compare Logistic Regression and RBF SVM using
handcrafted shape, intensity, and texture features.

## Dataset
BUS-BRA v1.0
https://zenodo.org/records/8231412

## Experiments
1. Clean-mask classification: Logistic Regression vs RBF SVM.
2. Robustness under boundary erosion, dilation, and translation.
3. Stability-aware feature selection to balance predictive
   performance and robustness.

## Evaluation
- Patient-level train/test separation
- Patient-grouped cross-validation
- Balanced accuracy, sensitivity, specificity, and ROC-AUC
- Prediction stability and calibration
- Failure analysis and computational complexity

## Setup
1. Install Python and create a virtual environment.
2. Install dependencies using:
   pip install -r requirements.txt
3. Download the BUS-BRA dataset from the official source.
4. Place the downloaded dataset under data/raw/.

## Status
Environment configured. Dataset auditing is the next stage.

## Reproducibility
Random seeds, split assignments, preprocessing settings,
and model configurations will be documented and versioned.