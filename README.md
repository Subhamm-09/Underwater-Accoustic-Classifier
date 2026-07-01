# Underwater Sound Classification

## Overview
This project aims to classify underwater audio recordings into different sound categories using machine learning. The goal is to extract meaningful audio features and train a model that can automatically identify the type of underwater sound.

## Problem Statement
Identifying underwater sounds manually is time-consuming. This project automates the process by analyzing audio recordings and classifying them into predefined categories.

## Categories
- Biological sounds
- Vessel sounds
- Ambient environmental sounds

## Dataset
The dataset contains `.wav` audio recordings collected from publicly available sources.

Directory structure:
```
data/
├── raw/
│   ├── biological/
│   ├── vessel/
│   └── ambient/
└── processed/
```

## Project Structure

```
underwater-sound-classification/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── outputs/
├── README.md
└── requirements.txt
```

## Workflow

1. Collect audio recordings
2. Organize the dataset
3. Visualize audio waveforms and spectrograms
4. Extract audio features (MFCC, Chroma, Spectral Centroid, Zero Crossing Rate)
5. Prepare the dataset
6. Train machine learning models
7. Evaluate performance
8. Predict sound categories for new audio

## Technologies Used

- Python
- Jupyter Notebook / Google Colab
- NumPy
- Pandas
- Librosa
- Matplotlib
- Scikit-learn

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Current Progress

| Step               | Status                                               |
| ------------------ | ---------------------------------------------------- |
| Problem Definition | [done]Underwater sound classification                |
| Data Collection    | [done] Collecting biological, vessel, and ambient sounds|
| Data Cleaning      | [] Remove bad/duplicate/inconsistent audio           |
| Data Visualization | [done] Your visualization notebook                   |
| Feature Extraction | [] Extract MFCC, Chroma, Spectral Centroid, ZCR, etc.|
| Train/Test Split   | [] After feature extraction                          |
| Model Training     | [] Train a classifier (e.g., Random Forest or SVM)   |
| Evaluation         | [] Measure accuracy, precision, recall, F1 score     |
| Tuning             | [] Optimize model and features                       |
| Deployment         | [] Build a prediction script or simple application   |

## Future Improvements

- Increase dataset size
- Try deep learning models
- Improve feature engineering
- Deploy as a web application

## Author

Subham Panda
B.Tech CSE, OUTR Bhubaneswar
