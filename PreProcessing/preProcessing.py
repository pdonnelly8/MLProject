# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:52:01 2021

@author: Pauric
"""
#Using NumPy
import numpy as np
# Using Pyplot to plot graphs for audio
import matplotlib.pyplot as plt
# Using IO module to read audio files
from scipy.io import wavfile
audioFile = input("Please enter location of audio file\n")
freq_sample, sig_audio = wavfile.read(audioFile)

# Output the parameters: Siugnal Data Type, Sampling Frequency and Duration
print('\nShape of Signal: ', sig_audio.shape)
print('Signal Data Type: ', sig_audio.dtype)
print('Signal duration: ', round(sig_audio.shape[0] /
                                 float(freq_sample), 2), 'seconds')

#Normalise the Signal Value and plot it on a graph
pow_audio_signal = sig_audio / np.power(2, 15)
pow_audio_signal = pow_audio_signal [:100]
time_axis = 100 * np.arange(0, len(pow_audio_signal), 1)

plt.ylabel('Amplitude')
plt.xlabel('Time [milliseconds]')
plt.plot(time_axis, pow_audio_signal, color='blue')