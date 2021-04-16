import os
import subprocess
import time

import numpy as np

from pre_processing.audio_pre_processing import AudioDataset
from pre_processing.text_pre_processing import LinesTextDataSet
from .utils import get_all_paths_of_type

audio_file_extension = '.wav'


def convert_audio_types(root_path, orig_type, dst_type):
    paths_to_convert = get_all_paths_of_type(root_path, orig_type)
    dst_paths = [path[:-len(orig_type)] + dst_type for path in paths_to_convert]
    for orig_path, dst_path in zip(paths_to_convert, dst_paths):
        subprocess.check_output('ffmpeg -nostats -loglevel 0 -i {} {}'.format(orig_path, dst_path))


def efficient_convert_audio_types(root_path, orig_type, dst_type):
    paths_to_convert = get_all_paths_of_type(root_path, orig_type)
    dst_paths = [path[:-len(orig_type)] + dst_type for path in paths_to_convert]
    for orig_path, dst_path in zip(paths_to_convert, dst_paths):
        if not os.path.isfile(dst_path):
            subprocess.check_output('ffmpeg -nostats -loglevel 0 -i {} {}'.format(orig_path, dst_path))





def get_transcription_files(root_path):
    return get_all_paths_of_type(root_path, 'txt')


def match_audio_paths_to_transcriptions(transcription_path):
    root_dir = os.path.dirname(os.path.normpath(transcription_path))
    with open(transcription_path, 'r', encoding='utf-8') as f:
        lines = f.read().replace('\r', '').split('\n')
        utterances = [line.split(' ', 1) for line in lines if len(line) > 0]
    samples = {os.path.join(root_dir, utterance[0] + audio_file_extension): utterance[1] for utterance in utterances}
    return samples


def get_librispeech_sample_matching(root_path):
    transcription_files = get_transcription_files(root_path)
    transcription_matchings = {}
    for transcription_file in transcription_files:
        transcription_matchings.update(match_audio_paths_to_transcriptions(transcription_file))
    return transcription_matchings


class LibriSpeechDataset:
    def __init__(self, path, cache_path=None):
        self.path = path
        self.samples = get_librispeech_sample_matching(self.path)

        audio_cache_path = None
        text_cache_path = None
        if cache_path is not None:
            audio_cache_path = os.path.join(cache_path, 'audio')
            #   text_cache_path = os.path.join(cache_path, 'text')  # current dataset size too small for text caching to be useful

        self.text_dataset = LinesTextDataSet(list(self.samples.values()),
                                             os.path.join(self.path, 'word_dictionary.dict'),
                                             cache_path=text_cache_path)
        #   self.text_dataset.load()
        if len(
                self.text_dataset.word_dictionary.words) == 0:  # if dictionary is empty (otherwise, assume it has been created)
            self.text_dataset.create_dict()
        self.audio_dataset = AudioDataset(list(self.samples.keys()), cache_path=audio_cache_path)

    def get_samples_with_context_size(self, text_ctx, audio_ctx, text_stride=1, audio_stride=1):
        start_time = time.time()
        spec_samples = self.audio_dataset.get_samples_with_context_size_per_path(audio_ctx, audio_stride)
        elapsed_time = time.time() - start_time
        print('time to load audio_samples: {}'.format(elapsed_time))

        start_time = time.time()
        text_samples = self.text_dataset.get_samples_with_context_size_per_path(text_ctx, text_stride)
        elapsed_time = time.time() - start_time
        print('time to load text_samples: {}'.format(elapsed_time))

        start_time = time.time()

        matching_text_samples = []
        matching_audio_samples = []
        for text_sample, spec_sample in zip(text_samples, spec_samples):
            for i in range(spec_sample.shape[0]):
                if i >= text_sample.shape[0]:
                    matching_text_sample = text_sample[-1]
                else:
                    matching_text_sample = text_sample[i]
                matching_text_samples.append(matching_text_sample)
                matching_audio_samples.append(spec_sample[i])

        text_samples_tensor = np.array(matching_text_samples)
        audio_samples_tensor = np.array(matching_audio_samples)

        elapsed_time = time.time() - start_time
        print('time to align audio samples to text samples: {}'.format(elapsed_time))

        return text_samples_tensor, audio_samples_tensor
