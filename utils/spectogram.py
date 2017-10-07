"""

Everything spectogram related.
"""
import numpy as np
import scipy

import matplotlib.pyplot as plot 

def audio_to_numpy(audio):
	"""
	convert audio file into a numpy array

	@input audio <string>: audio file location

	@return (rate, audio) <tuple>
	rate <int>: sample rate of wav file
	audio <np.array>: array format of audio
	"""
	if (audio[:-4] == ".wav"):
		return scipyt.io.wavfile.read(audio)

	raise AttributeError("Must be .wav file")

def audio_to_spect(rate, audio, nperseg=256):
	"""
	convert audio numpy array into a spectogram
	
	@input rate <int>: sample rate of wav file
	@input audio <np.array>: array format of audio
	@input nperseg <int>: number of 
	
	@return <np.array>: 2D array of spectogram
	"""
	return scipy.signal.stft(audio, fs=rate, nperseg=nperseg)

def spect_to_audio(rate, spect, nperseg=256):
	"""
	convert a spectogram array into an audio array

	@input spect <int>: sample rate of wav file
	@input spect <np.array>: 2D array of spectogram
	@input nperseg <int>: number of datapoints corresponding to each SFFT segment

	@return <np.array>: array format of audio
	"""
	return scipy.signal.istft(spect, rate, nperseg=nperseg)



