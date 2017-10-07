import numpy as np
import scipy
import scipy.io.wavfile
from scipy import signal

import matplotlib.pyplot as plt

def audio_to_numpy(audio, mono=True):
    """
    convert audio file into a numpy array

    @input audio <string>: audio file location
    @input mono <bool>: True if compress to mono, False to keep stereo
    
    @return (rate, audio) <tuple>
    rate <int>: sample rate of wav file
    audio <np.array>: array format of audio
    """
    print(audio[-4:])
    if (audio[-4:] == ".wav"):
        arr = scipy.io.wavfile.read(audio)
        if (mono):
            return arr.sum(axis=1) / 2
        return arr
    
    raise AttributeError("Audio must be .wav file")

def audio_to_spect(rate, audio, nperseg=256, noverlap=None):
    """
    convert audio numpy array into a spectogram
    
    @input rate <int>: sample rate of wav file
    @input audio <np.array>: array format of audio
    @input nperseg <int>: number of 
    
    @return (f, t, Zxx) <tuple>
    f <np.array>: Array of sample frequencies
    t <np.array>: Array of segment times
    Zxx <np.array>: 2D array of spectogram
    """
    return signal.stft(audio, fs=rate, nperseg=nperseg, noverlap=None)

def spect_to_audio(rate, spect, nperseg=256, noverlap=None):
    """
    convert a spectogram array into an audio array
    
    @input spect <int>: sample rate of wav file
    @input spect <np.array>: 2D array of spectogram
    @input nperseg <int>: number of datapoints corresponding to each SFFT segment

    @return (t, x) <tuple>
    t <np.array>: array of output data times
    x <np.array>: array format of audio; iSTFT of Zxx
    """
    return signal.istft(spect, rate, nperseg=nperseg, noverlap=noverlap)

def save_audio(audio, rate, filename):
    """
    save the audio array into a wav file
    
    @input audio <np.array>: array of audio
    @input rate <int>: sampling rate
    @input filename <string>: file location to save to
    
    @return None
    """
    scipy.io.wavfile.write(filename, rate, audio)

#############################################

"""
Some sample test code.
"""

#############################################


# audio_loc = "AhCmonNowBaby.wav"
# nperseg = 100
# noverlap = 0

# rate, audio = audio_to_numpy(audio_loc)
# audio = audio.sum(axis=1) / 2
# f, t, Zxx = audio_to_spect(rate, audio, nperseg)

# plt.figure()
# amp = 2 * np.sqrt(2)
# plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
# plt.ylim([f[1], f[-1]])
# plt.show()
# print(audio.shape)
# print(Zxx)
# print("t:" + str(t.shape))

# t, new_audio = spect_to_audio(rate, Zxx, nperseg)
# new_audio = new_audio[26:]

# steps = range(55424)
# print(audio.shape)
# print(new_audio.shape)

# plt.figure()
# plt.plot(steps, audio, steps, new_audio)
# plt.xlabel('Time [sec]')
# plt.ylabel('Signal')
# plt.show()

# save_audio(new_audio, rate, "test.wav")


