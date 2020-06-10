import numpy as np
import speech_recognition as sr
r=sr.Recognizer()
mic=sr.Microphone()
with mic as source:
    print("Listening:")
    r.adjust_for_ambient_noise(source)
    audio =r.listen(source)
with open("microphone-result.wav","wb") as f:
    f.write(audio.get_wav_data())