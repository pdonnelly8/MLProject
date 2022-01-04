from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import soundfile as sf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

def mean_mfccs(x):
    return [np.mean(feature) for feature in librosa.feature.mfcc(x)]

# Flattens the shape of the audio file
def parse_audio(x):
    return x.flatten('F')[:x.shape[0]]


# Retrieves the audio from the dataframe and parses the audio
def get_audios(df):
    train_file_names = df['FullName']
    base_path = Path(__file__).parent
    samples = []
    for file_name in train_file_names:
        file_name = (base_path / file_name).resolve()
        x, sr = librosa.load(file_name)
        x = parse_audio(x)
        samples.append(mean_mfccs(x))

    return np.asarray(samples, dtype=object)


# Reads in dataset csv and retrieves the audio
def get_samples():
    df = pd.read_csv('datasetWIP.csv')
    return get_audios(df), df['Class'].values


def init():
    audio, labels = get_samples()
    print(audio)
    audio = tf.convert_to_tensor(audio, tf.float32)
    train_data, test_data, train_labels, test_labels = train_test_split(audio, labels, test_size=0.2, random_state=21)

    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)
    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[train_labels]
    normal_test_data = test_data[test_labels]

    anomalous_train_data = train_data[~train_labels]
    anomalous_test_data = test_data[~test_labels]

    plt.grid()
    plt.plot(np.arange(140), normal_train_data[0])
    plt.title("A Normal File")
    plt.show()

init()
