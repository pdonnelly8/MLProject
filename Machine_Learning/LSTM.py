import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Bidirectional, Input
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

rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

# Plotting waveforms onto a graph 
for i, (audio, label) in enumerate(waveform_ds.take(n)):
  r = i // cols
  c = i % cols
  ax = axes[r][c]
  ax.plot(audio.numpy())
  ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
  label = label.numpy().decode('utf-8')
  ax.set_title(label)

plt.show()



#########################################################
# Convert Waveforms to Spectrograms
#########################################################

# Getting the average shape of 100 waveform samples to use for padding
shapeTotal = 0
for j, (audio, label) in enumerate(waveform_ds.take(100)):
    tensor_shape_tuple = audio.get_shape()
    shapeTotal = shapeTotal + tensor_shape_tuple[0]
avgShape = round(shapeTotal/100)
print("Average Shape of Audio: " + str(avgShape))

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than the average shape of samples.
  input_len = avgShape
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [avgShape] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  return spectrogram

for waveform, label in waveform_ds.take(1):
  label = label.numpy().decode('utf-8')
  spectrogram = get_spectrogram(waveform)

print('Label:', label)
print('Waveform shape:', waveform.shape)
print('Spectrogram shape:', spectrogram.shape)

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, avgShape])

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.show()

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.argmax(label == classes)
  return spectrogram, label_id

spectrogram_ds = waveform_ds.map(
  map_func=get_spectrogram_and_label_id,
  num_parallel_calls=AUTOTUNE)

rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
  r = i // cols
  c = i % cols
  ax = axes[r][c]
  plot_spectrogram(spectrogram.numpy(), ax)
  ax.set_title(classes[label_id.numpy()])
  ax.axis('off')

plt.show()

#########################################################
# Build and Train RNN Model
#########################################################

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(
      map_func=get_waveform_and_label,
      num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
    map_func=get_spectrogram_and_label_id,
    num_parallel_calls=AUTOTUNE)
  return output_ds

train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 32
full_train = 565
full_val_test = 121
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(classes)

# Instantiate the `tf.keras.layers.Normalization` layer.
norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrogram with `Normalization.adapt`.
norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

# Defining layers within Neural Network
model = models.Sequential()

model.add(Input(shape=input_shape))
model.add(norm_layer)
model.add(Bidirectional(LSTM(32, activation='tanh', return_sequences=True)))
model.add(Bidirectional(LSTM(64, activation='tanh')))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_labels, activation='sigmoid'))



# model.add(layers.Input(shape=input_shape))
# model.add(norm_layer)
# model.add(LSTM(128,activation='tanh', return_sequences=True))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='tanh'))
# model.add(Dense(64, activation='tanh'))
# model.add(Dropout(0.4))
# model.add(Dense(48, activation='tanh'))
# model.add(Dropout(0.4))
# model.add(Dense(24, activation='tanh'))
# model.add(Dropout(0.4))
# model.add(Dense(num_labels, activation='sigmoid'))
model.summary()

optimiser = tf.keras.optimizers.Adam(learning_rate=0.001,clipvalue=0.5)
model.compile(optimizer=optimiser,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['acc'])

EPOCHS = 150
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
)

# visualkeras.layered_view(model, to_file='output.png').show() # write and show
model.save("lstm.h5")
metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

plt.plot(history.epoch, metrics['acc'], metrics['val_acc'])
plt.legend(['acc', 'val_acc'])
plt.show()

#########################################################
# Evaluating the Model Performance
#########################################################

test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

#Making a prediction against generated model using test samples
y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)

# Confusion Matrix
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 2))
sns.heatmap(confusion_mtx,
            xticklabels=classes,
            yticklabels=classes,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Groundtruth')
plt.show()

#Measuring Type I and Type II errors
truePositives=confusion_mtx[0][0].numpy()
falseNegatives=confusion_mtx[0][1].numpy()
falsePositives=confusion_mtx[1][0].numpy()
trueNegatives=confusion_mtx[1][1].numpy()

errorRate=(falseNegatives+falsePositives)/len(y_true)
recall=truePositives/(truePositives+falseNegatives)
precision=truePositives/(truePositives+falsePositives)
specificity=trueNegatives/(trueNegatives+falsePositives)
fMeasure=(2 * precision * recall)/(precision + recall)
falseAlarm=1-specificity

print(f'Test Set Accuracy: {test_acc:.0%}')
print(f'Test Set Error Rate: {errorRate:.0%}')
print(f'Test Set Recall: {recall:.0%}')
print(f'Test Set Precision: {precision:.0%}')
print(f'Test Set Specificity: {specificity:.0%}')
print(f'Test Set F-Measure (F1): {fMeasure:.0%}')
print(f'Test Set False Alarm Rate: {falseAlarm:.0%}')