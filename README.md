# Underwater Accoustic Classifier

Live at - https://underwater-accoustic-classifier-fcw6yw5jmxbu84tvmzpwfk.streamlit.app/

## Overview
This project aims to classify underwater audio recordings into different sound categories using machine learning. The goal is to extract meaningful audio features and train a model that can automatically identify the type of underwater sound.

## Objectives
Build an end-to-end machine learning pipeline for underwater acoustic classification.

Compare the performance of multiple classification algorithms.

Evaluate models using standard classification metrics.

Save the best-performing model for inference.

Create a prediction pipeline for unseen data.

Deploy the model as an interactive web application.

## Categories
- Biological sounds
- Vessel sounds
- Ambient environmental sounds

## Dataset
The dataset contains `.wav` audio recordings collected from publicly available sources.

The sources of the Dataset are -
        -NOAA fisheries
        -Github repositories
        -Kaggle
        -fishsounds.net
        -Watkins database
        -accoustics.uk
        -oceannetworks
        -soundcloud

## Project structure:

Underwater-Acoustic-Classifier/
│

├── app.py

├── requirements.txt

├── README.md

├── .gitignore

│

├── data/

│   ├── augmented/

│   ├── features/

│   ├── processed/

│   ├── raw/

│   ├── scaled/

│   └── splits/

│

├── models/

│   ├── Logistic_regression.pkl

│   ├── SVM.pkl

│   ├── Random_forest.pkl

│   ├── KNN.pkl

│   ├── XGBoost.pkl

│   └── standard_scaler.pkl

│

├── notebooks/

│   ├── 01_Loading_files.ipynb

│   ├── 02_Data_visualisation.ipynb

│   ├── 03_Data_preprocessing.ipynb

│   ├── 04_Data_augmentation.ipynb

│   ├── 05_Feature_Extraction.ipynb

│   ├── 06_Train_test_split.ipynb

│   ├── 07_Feature_scaling.ipynb

│   ├── 08_Logistic_regression.ipynb

│   ├── 09_Support_vector_machine.ipynb

│   ├── 10_Random_forest.ipynb

│   ├── 11_KNN.ipynb

│   ├── 12_XGBoost.ipynb

│   ├── 13_Model_comparision.ipynb

│   ├──

│

└── src/

├── __pycache__/

    
├── __init__.py
   
├── features.py
    
├── preprocess.py
    
└── predict.py
    

## Workflow


1. Collect audio recordings
 
2. Organize the dataset
 
3. Data Visualisation
 
4. Data cleaning and Preprocessing
 
5. Data Augmentation
 
6. Extract audio features (MFCC, Chroma, Spectral Centroid, Zero Crossing Rate)

7. Splitting of the Data set Train/Test split

8. Train machine learning models - Logistic Regression

   - SVM
    
     - RandomForest
    
        - KNN
    
           - XGBoost
    
9. Evaluate performance and compare the models
 
10. Predict sound categories for new audio
 

## Technologies Used

- Python
  
- Google Colab
  
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

| Step               | Status                                                    |
| ------------------ | ----------------------------------------------------      |
| Problem Definition | [done]Underwater sound classification                     |
| Data Collection    | [done] Collecting biological, vessel, and ambient sounds  |  
| Data Visualization | [done] Your visualization notebook                        | 
| Data Cleaning      | [done] Remove bad/duplicate/inconsistent audio            | 
| Feature Extraction | [done] Extract MFCC, Chroma, Spectral Centroid, ZCR, etc. | 
| Train/Test Split   | [done] After feature extraction                           |
| Model Training     | [done] Train a classifier (e.g., Random Forest or SVM)    |
| Evaluation         | [done] Measure accuracy, precision, recall, F1 score      |
| Prediction         | [] Prediction of newly uploaded audio                     |
| Deployment         | [] Build a prediction script or simple application        |

## Future Improvements

- Increase dataset size
- Try deep learning models
- Improve feature engineering
- Deploy as a web application

## Author

Subham Panda
Intern at Coratia Technologies
B.Tech CSE, OUTR Bhubaneswar

