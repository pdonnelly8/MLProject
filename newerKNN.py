import os

import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf # librosa fails when reading files on Kaggle.

import matplotlib.pyplot as plt
import IPython.display as ipd

from sklearn.model_selection import (train_test_split, cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (confusion_matrix, accuracy_score)
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

# computes the mean mfcss for each audio sample
def mean_mfccs(x):
    return [np.mean(feature) for feature in librosa.feature.mfcc(x)]

# Flattens the shape of the audio file
def parse_audio(x):
    return x.flatten('F')[:x.shape[0]] 

# Retrieves the audio from the dataframe and parses the audio
def get_audios(df):
    train_file_names = df['FullName']
    print(train_file_names)
    samples = []
    for file_name in train_file_names:
        x, sr = sf.read(file_name, always_2d=True)
        x = parse_audio(x)
        samples.append(mean_mfccs(x))
        
    return np.array(samples)


# Reads in dataset csv and retrieves the audio
def get_samples():
    df = pd.read_csv('newerdataset.csv')
    return get_audios(df), df['Class'].values

# init function to run our training/testing
def init():
    # Retrieve samples and labels
    X, Y = get_samples()

    # split data into testing and training
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #Output the shape, values and labels of some of the data
    print(f'Shape: {x_train.shape}')
    print(f'Observation: \n{x_train[0]}')
    print(f'Labels: {y_train[:5]}')

    # Reducing Noise by means of PCA
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    pca = PCA().fit(x_train_scaled)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)')
    plt.show()

    # Adding params and setting up the model for KNN
    grid_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
    }

    model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=1)
    model.fit(x_train_scaled, y_train)
    """
    #Adding params and setting up for RF
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 5, 6, 7, 8],
        'criterion': ['gini', 'entropy']
    }

    rfc = RandomForestClassifier(random_state=42)
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    CV_rfc.fit(x_train, y_train)
    # Applying best params
    CV_rfc.best_params_
    rfc1 = RandomForestClassifier(random_state=42, max_features='auto', n_estimators=200, max_depth=8, criterion='gini')
    pred = rfc1.predict(x_test)
    print("Accuracy for Random Forest on CV data: ", accuracy_score(y_test, pred))
    """
    # Evaluating the results
    print(f'Model Score: {model.score(x_test_scaled, y_test)}')

    y_predict = model.predict(x_test_scaled)
    print(f'Confusion Matrix: \n{confusion_matrix(y_predict, y_test)}')

    raise SystemExit

init()