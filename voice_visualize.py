import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import speech_recognition as sr
from playsound import playsound
r=sr.Recognizer()
mic=sr.Microphone()
with mic as source:
    print("Listening:")
    r.adjust_for_ambient_noise(source)
    audio =r.listen(source)
with open("microphone.wav","wb") as f:
    f.write(audio.get_wav_data())

frequency_sampling,audio_signal =wavfile.read("microphone.wav")

print("signal shape:",audio_signal.shape)
print("signal datatype",audio_signal.dtype)
print("signal duration",round(audio_signal.shape[0]/float(frequency_sampling),2),'seconds')

audio_signal=audio_signal/np.power(2,15)

audio_signal=audio_signal[:100]
time_axis= 1000 * np.arange(0,len(audio_signal),1)/float(frequency_sampling)

plt.plot(time_axis,audio_signal,color='blue')
plt.xlabel('time (miliseconds)')
plt.ylabel('amplitude')
plt.title('Input audio signal')
plt.show()


playsound("microphone.wav")