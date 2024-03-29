# Parkinson's Disease Classification 
![logo](parkinson.jpg)
## Overview

This project focuses on the classification and analysis of Parkinson's disease (PD) using machine learning techniques. Parkinson's disease is a neurodegenerative disorder that affects a person's movement, causing symptoms such as tremors, bradykinesia (slowness of movement), and muscle rigidity. Early diagnosis and accurate classification of PD are essential for effective treatment and management of the disease. In this project, I use a dataset containing various voice features to build and evaluate machine learning models for PD classification.

## Dataset

I start by obtaining a dataset named "pd_speech_features.csv," which contains a collection of voice features extracted from the speech signals of individuals, some with Parkinson's disease and some without. The dataset consists of the following columns:

- Features: These columns represent various voice-related features, which serve as input data for our machine learning models.(753 columns)

- Label: The "class" column is our target variable, where '1' indicates the presence of Parkinson's disease, and '0' indicates its absence.

## Data Preprocessing

The project begins with data preprocessing, including the following steps:

1. Reading the Dataset: I load the dataset into a pandas DataFrame for analysis and model building.

2. Column Renaming: I assign meaningful names to the columns to enhance readability.

3. Data Splitting: I split the dataset into training and testing sets to train and evaluate my machine learning models. I use a 70-30 split ratio.

4. Standard Scaling: I standardize the feature data by removing the mean and scaling it to unit variance. Standardization helps improve the performance of certain machine learning algorithms.

## Model Selection

I explore various machine learning algorithms to classify Parkinson's disease based on the voice features. My model selection process includes the following algorithms:

### Decision Tree

I use a decision tree classifier and tune its hyperparameters using cross-validation. I evaluate its accuracy and generate a confusion matrix to assess its performance on both the training and testing sets.

### K-Nearest Neighbors (KNN)

I employ the K-nearest neighbors algorithm and experiment with different values of 'k' (the number of neighbors). I perform cross-validation to find the optimal 'k' value and assess the model's accuracy and confusion matrix.

### Support Vector Machine (SVM)

I implement a Support Vector Machine classifier with a radial basis function (RBF) kernel and explore different values for the regularization parameter 'C' and polynomial degree. I use cross-validation to fine-tune the parameters and evaluate the model's accuracy and confusion matrix.

### Random Forest

I build a Random Forest classifier and vary the maximum depth of the trees and the number of estimators (trees) to optimize the model's performance. Cross-validation is used to select the best hyperparameters, and I assess accuracy and generate a confusion matrix.

## Principal Component Analysis (PCA)

To handle the high dimensionality of the dataset, I apply Principal Component Analysis (PCA) to reduce the number of features while retaining as much information as possible. I perform PCA separately on both the training and testing sets and evaluate its impact on model accuracy.

## Results and Analysis

I compare the performance of each machine learning model in terms of accuracy and confusion matrices on both the training and testing datasets. I also provide a classification report for a more detailed assessment of each model's precision, recall, and F1-score.

## Conclusion

This project aims to demonstrate the effectiveness of machine learning algorithms in the classification of Parkinson's disease using voice-related features. By preprocessing the data, selecting appropriate models, and fine-tuning hyperparameters, we can build accurate models for early PD detection. Additionally, I investigate the impact of dimensionality reduction through PCA on model performance.

The results obtained from this project can have significant implications for the early diagnosis and management of Parkinson's disease, potentially improving the quality of life for affected individuals.
