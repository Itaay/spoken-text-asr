import os
from hashlib import md5

import numpy as np
from nltk.tokenize import word_tokenize

from pre_processing.contextual_dataset import ContextualDataSet


# tokenize text
def tokenize_text(text_sample):
    tokenized = word_tokenize(text_sample)
    return tokenized


class WordDictionary:
    def __init__(self, path):
        self.dictionary_path = path
        self.words = list(set([]))

    def load_dictionary(self):
        if os.path.isfile(self.dictionary_path):
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                txt = f.read()
                lines = txt.replace('\r', '').split('\n')
                self.words = lines

    def add_special_token(self, token):
        self.words = list(set([token] + self.words))

    def update_dictionary(self, tokens):
        self.words = sorted(list(set(self.words + tokens)))

    def accumulate_text_samples(self, text_samples):
        tokens = [token for text_sample in text_samples for token in tokenize_text(text_sample)]
        self.update_dictionary(tokens)

    def save_dictionary(self):
        with open(self.dictionary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.words))

    def one_hot_encode(self, tokens):
        one_hot = np.zeros((len(tokens), len(self.words)))
        for i in range(len(tokens)):
            one_hot[i, self.words.index(tokens[i])] = 1.0
        return one_hot

    def one_hot_decode(self, one_hots):
        top_choices = np.argmax(one_hots, axis=-1)
        return [self.words[choice] for choice in top_choices]


class TextDataSet(ContextualDataSet):
    def __init__(self, samples, dict_path, cache_path=None):
        super().__init__(samples, cache_path=cache_path)
        self.word_dictionary = WordDictionary(dict_path)
        self.text_samples = []
        self.tokenized_samples = []
        self.one_hots = []
        self.pad_value = '<sil>'

    def get_path_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read()
        return txt

    def get_path_sample(self, path):
        txt = self.get_path_data(path)
        tokenized = tokenize_text(txt)
        one_hot = self.word_dictionary.one_hot_encode(tokenized)
        return one_hot

    def get_all_raw_samples(self):
        return [self.get_path_sample(sample) for sample in self.sample_paths]

    def load(self):
        self.word_dictionary.load_dictionary()

    def create_dict(self):
        if self.pad_value is not None:
            self.word_dictionary.add_special_token(self.pad_value)
        self.word_dictionary.accumulate_text_samples([self.get_path_data(path) for path in self.sample_paths])
        self.word_dictionary.save_dictionary()

    def get_samples_with_context_size_per_path(self, ctx, stride=1, pad_value=None):
        return super().get_samples_with_context_size_per_path(ctx, stride=stride, pad_value=self.word_dictionary.one_hot_encode([self.pad_value])[0])


# in this Dataset. each sample isn't actually a path, but a transcription line (utterance)
class LinesTextDataSet(TextDataSet):
    def __init__(self, samples, dictionary_path, cache_path=None):
        super().__init__(samples, dictionary_path, cache_path=cache_path)

    def get_path_data(self, path):
        return path

    def get_cache_path_for_file(self, file_path, ctx, stride):
        unified_path_name = md5(file_path.encode('utf-8')).hexdigest()
        return super().get_cache_path_for_file(unified_path_name, ctx, stride)


"""
To do:
"""
# use word2vec or basic premade word-encoding
