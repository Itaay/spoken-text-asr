import librosa
import numpy as np
from scipy.io.wavfile import read as read_audio
from scipy.signal import spectrogram
from .contextual_dataset import ContextualDataSet


def get_mfcc(y, sr):
    y = np.array(y, dtype=np.float32)
    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=4096)
    S = np.log10(S + 1)
    mfcc = librosa.feature.mfcc(S=S, n_mfcc=13).T
    mean = np.mean(mfcc)
    std = np.std(mfcc)
    mfcc = (mfcc - mean) / np.maximum(0.0001, std)
    return mfcc


def get_mfcc_for_audio_path(path):
    sr, y = read_audio(path)
    if len(y.shape) > 1:
        y = y[:, 0]
    return get_mfcc(y, sr)


def get_spectogram_for_audio_path(path):
    sr, y = read_audio(path)
    if len(y.shape) > 1:  # if several channels
        y = y[:, 0]  # take only the first
    # f, t, s = spectrogram(y, sr, scaling='spectrum', nfft=4096)
    f, t, s = spectrogram(y, sr, scaling='spectrum', nfft=4096)
    s = np.log10(s + 1)
    return s


class AudioDataset(ContextualDataSet):
    def __init__(self, audio_paths, cache_path=None):
        super().__init__(audio_paths, cache_path=cache_path)

    def get_path_sample(self, path):
        return get_mfcc_for_audio_path(path)
