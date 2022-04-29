import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from scipy import stats
from scipy.stats import norm
from scipy.stats import shapiro
from tensorflow.keras import layers, models

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#########################################################
# Importing the Dataset
#########################################################

DATASET_PATH = 'AudioSamples'
data_dir = pathlib.Path(DATASET_PATH)

classes = np.array(tf.io.gfile.listdir(str(data_dir)))
classes = classes[classes != 'dataset.csv']
print('Classes: ', classes)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Example file tensor:', filenames[0])

# 70:15:15 split for train, test and validation (808 samples in total)

train_files = filenames[:565]
val_files = filenames[565: 565 + 121]
test_files = filenames[-121:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))

#########################################################
# Read the Audio files and their labels
#########################################################

test_file = tf.io.read_file(DATASET_PATH+'/Healthy/4/4_Healthy_High.wav')
test_audio, _ = tf.audio.decode_wav(contents=test_file)
print(test_audio.shape)

def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-3]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label


AUTOTUNE = tf.data.AUTOTUNE

files_ds = tf.data.Dataset.from_tensor_slices(train_files)

waveform_ds = files_ds.map(
    map_func=get_waveform_and_label,
    num_parallel_calls=AUTOTUNE)

# Lengths of Audio plotted onto graph to prove that padding is required and should not affect results
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
for i, (audio, label) in enumerate(waveform_ds.take(808)):
  ax.scatter(audio.shape, label.numpy().decode('utf-8'), color='b')

ax.set_xlabel('Labels')
ax.set_ylabel('Length')
ax.set_title('scatter plot')
ax.axis('on')
plt.show()

#Shapiro-Wilks test to show that data is normally distributed
'''
rng = np.random.default_rng()
x = stats.norm.rvs(loc=5, scale=3, size=100, random_state=rng)
shapiro_test = stats.shapiro(x)
print(shapiro_test)
'''
data = []
for i, (audio, label) in enumerate(waveform_ds.take(808)):
    data.append(audio.shape[0])

mu, std = norm.fit(data)
plt.hist(data, bins=100, density=True, alpha=0.6, color='b')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
  
plt.plot(x, p, 'k', linewidth=2)
title = "Fit Values: mu = {:.2f} and std = {:.2f}".format(mu, std)
plt.title(title)
plt.xlabel("Audio Lengths")
plt.ylabel("Probability Density")  
plt.show()
for i in range(0,10):
  shapiro_wilks = shapiro(data)
  #print("Test " + str(i) + ": ", shapiro_wilks)