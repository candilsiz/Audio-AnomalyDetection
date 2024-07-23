import numpy as np
import librosa
import random

def extract_mfccs(audio, sample_rate, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfccs_mean= np.mean(mfccs.T, axis=0)
    return mfccs_mean

def extract_spectral_features(audio, sample_rate):
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)[0] 
    spectral_melspectogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)[0] 
    return np.mean(spectral_centroids), np.mean(spectral_rolloff), np.mean(spectral_contrast), np.mean(spectral_melspectogram)

def extract_temporal_features(audio):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    autocorrelation = librosa.autocorrelate(audio)
    return np.mean(zero_crossing_rate), np.mean(autocorrelation)

def add_white_noise(audio, noise_factor=0.05):
    """
    Adds gaussian white noise to the signal, with zero mean No*fs/2 variance.
    """
    white_noise = np.random.normal(0, audio.std(), audio.size)
    noised_audio = audio + (white_noise * noise_factor)
    return noised_audio

def time_stretch(audio, stretch_rate=0.7):
    """
    Stretch factor. If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down.
    """
    stretched_audio = librosa.effects.time_stretch(audio, stretch_rate)
    return stretched_audio

def pitch_scaling(audio, sample_rate, n_tones=1):
    """
    Shift the pitch of a waveform by n_steps steps.
    """
    pitched_audio = librosa.effects.librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_tones)
    return pitched_audio

def random_gain_scaling(audio, min_scaling_factor, max_scaling_factor):
    """
    Multiplies the entire audio signal by this scaling factor, effectively changing the volume of the audio.
    """
    scaling_factor = random.uniform(min_scaling_factor, max_scaling_factor)
    gained_audio = audio * scaling_factor
    return gained_audio, scaling_factor