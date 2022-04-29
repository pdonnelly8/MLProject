import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import errors
from pydub import AudioSegment

def predict_classification(file_path):
    """
    This function predicts the classification of a given audio recording.
    """
    
    AudioSegment.ffmpeg = 'E:\\ffmpeg-2022-04-11-git-d6d46a2c50-essentials_build\\bin\\ffmpeg.exe'
    AudioSegment.ffprobe = 'E:\\ffmpeg-2022-04-11-git-d6d46a2c50-essentials_build\\bin\\ffprobe.exe'
    AudioSegment.converter = 'E:\\ffmpeg-2022-04-11-git-d6d46a2c50-essentials_build\\bin\\ffmpeg.exe'
    
    #Load in Audio from the file
    if(file_path == 'D:\\pauricdonnellyfinalyearproject\\flaskServer\\recording.m4a'):
        wav_filepath = 'D:\\pauricdonnellyfinalyearproject\\flaskServer\\recording.wav'
        track = AudioSegment.from_file(file_path,  format= 'm4a')
        track.export(wav_filepath, format='wav')
        file_path = wav_filepath
    
    while True:
        try:
            audio = tf.io.read_file(file_path)
            break
        except errors.NotFoundError:
            print("File not found. Please try again.")
            return "Error loading audio. Please try again."
    audio = tf.io.read_file(file_path)
    
    # Load the model   
    while True:
        try:
            model = keras.models.load_model('D:\\pauricdonnellyfinalyearproject\\flaskServer\\lstm.h5')
            break
        except errors.NotFoundError:
            print("Model not found. Please try again.")
            return "Error loading model. Please try again."

    # Decode the audio recording
    audio_recording, _ = tf.audio.decode_wav(contents=audio, desired_channels=1)
    audio_recording = tf.squeeze(audio_recording, axis=-1)

    #Convert audio recording to a float32 tensor
    audio_recording = tf.cast(audio_recording, tf.float32)

    # Reshape the audio recording to a 2D tensor
    audio_recording = tf.reshape(audio_recording, [1, -1])

    #Convert audio to STFT
    spectrogram = tf.signal.stft(audio_recording, frame_length=255, frame_step=128)

    #Obtain magnitude of STFT
    spectrogram = tf.abs(spectrogram)

    # Trim or pad the magnitude spectrogram to the size of the model input
    if(spectrogram.shape[1] < 2700):
        padding_required = 2700 - spectrogram.shape[1]
        padding = tf.zeros([1, padding_required, 129])
        spectrogram = tf.concat([spectrogram, padding], axis=1)
    else:
        spectrogram = spectrogram[:, :2700, :]
    
    #Predict the classification
    prediction = np.argmax(model.predict(spectrogram), axis=1)
    #Return the prediction
    return prediction[0]



if __name__ == "__main__":
    predict_classification()