import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc,logfbank

frequency_sampling,audio_signal =wavfile.read("microphone.wav")
audio_signal=audio_signal[:15000]


features_mfcc=mfcc(audio_signal,frequency_sampling)

print("\n mfcc:\n number of windows=",features_mfcc.shape[0])
print("length of each feature=",features_mfcc.shape[1])

features_mfcc=features_mfcc.T
plt.matshow(features_mfcc)
plt.title('mfcc')
filterbank_features=logfbank(audio_signal,frequency_sampling)

print('\nfilter bank:\n number of window=',filterbank_features.shape[0])
print('\nlength of each feature=',filterbank_features.shape[1])
filterbank_features=filterbank_features.T
plt.matshow(filterbank_features)
plt.title("filter_bank")
plt.show()
