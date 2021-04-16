import time

from torch.utils.data import DataLoader
from utils.device import device, allocate_iterable
from utils.metrics import wer
from embeddings import LinearEmbedding, OneHotLinearEmbedding, ConvolutionalEmbedding
from generic_cdt import GenericConcatenatedDomainsTransformer
from pre_processing.librispeech_preprocessing import LibriSpeechDataset
from spoken_text_trainer import SpokenTextDataset, ConcatenatedDomainsTrainer
import numpy as np
import torch
import pickle

"""
net = ConcatenatedDomainsTransformer(
    {'domains': [[256, 16, 32], [1024, 128, 36]], 'attention_blocks': [[10, [32, 36]]]})
"""
text_context = 8
audio_context = 128#   256

text_paths = ['data/text_files/sample0.txt']
audio_paths = ['data/audio_files/sample0.wav']
dictionary_path = 'data/tokens_dict.txt'
librispeech_path = 'data/LibriSpeech/dev-clean'

model_save_path = 'data/model_backup'
total_dataset_path = 'data/final_dataset'
"""
print('loading text dataset')
start = time.time()
text_dataset = TextDataSet(text_paths, dictionary_path)
text_dataset.load()
text_dataset.create_dict()

text_samples = text_dataset.get_samples_with_context_size(text_context, 1)
text_sample_dims = text_samples.shape[1:]

elapsed_time = time.time() - start
print('loaded text dataset. Time taken: {} seconds'.format(elapsed_time))
"""
"""
print('loading audio dataset')
start = time.time()
audio_dataset = AudioDataset(audio_paths)

audio_samples = audio_dataset.get_samples_with_context_size(audio_context, 1)

audio_sample_dims = audio_samples.shape[1:]
elapsed_time = time.time() - start
print('loaded audio dataset. Time taken: {} seconds'.format(elapsed_time))
"""

print('loading librispeech dataset')
start_time = time.time()

init_start = time.time()
librispeech_dataset = LibriSpeechDataset(librispeech_path, 'data/dataset_backups')
init_time = time.time() - init_start
print('time to initialize librispeech_dataset: {}'.format(init_time))
create_start = time.time()


libri_text_samples, libri_audio_samples = librispeech_dataset.get_samples_with_context_size(text_context, audio_context,
                                                                                            1, 20)  # 1, 60

np.save('data/libri_text_samples', libri_text_samples)
np.save('data/libri_audio_samples', libri_audio_samples)


create_time = time.time() - create_start
print('time to create librispeech_dataset: {}'.format(create_time))

print('creating stt dataset')

spoken_text_dataset = SpokenTextDataset()

#   spoken_text_dataset.add_text_samples(text_samples, audio_sample_dims)
# spoken_text_dataset.add_audio_samples(audio_samples, text_sample_dims)

#   spoken_text_dataset.add_stt_samples(np.load('data/libri_text_samples.npy'), np.load('data/libri_audio_samples.npy'), 1.0, 0.0)
spoken_text_dataset.add_stt_samples(libri_text_samples, libri_audio_samples, 1.0, 0.0)
text_sample_dims = spoken_text_dataset.input_text_samples.size()
audio_sample_dims = spoken_text_dataset.input_audio_samples.size()


text_input_dims = text_sample_dims[-1]
text_embedding = OneHotLinearEmbedding(text_input_dims, 200)

audio_input_dims = audio_sample_dims[-1]
audio_embedding = ConvolutionalEmbedding(audio_input_dims, 36)

attention_blocks_dims = [200, 200]

domain_context_sizes = [text_context, audio_context]

net = GenericConcatenatedDomainsTransformer(attention_blocks_dims, [text_embedding, audio_embedding],
                                            domain_context_sizes)
net.to(device)

trainer = ConcatenatedDomainsTrainer(net, domain_loss_weights=[1.0, 1.0])

batch_size = 32
dataset_loader = DataLoader(spoken_text_dataset, batch_size=batch_size, shuffle=True)


word_appearances = spoken_text_dataset.output_text_samples.flatten(end_dim=-2).sum(dim=-2)
total_appearances = torch.sum(word_appearances).item()
most_common_index = torch.argmax(word_appearances)
print('most common word: {}, with {}/{} appearances'.format(librispeech_dataset.text_dataset.word_dictionary.words[most_common_index], word_appearances[most_common_index], total_appearances))

print('training')
start = time.time()

sample_count = spoken_text_dataset.output_text_samples.size()[0]


def calculate_wer(model, data, word_dictionary):
    model.eval()
    with torch.no_grad():
        target_sentences = []
        pred_sentences = []
        for batch in data:
            inp_text, inp_audio, out_text, out_audio = batch
            pred_text, pred_audio = model(allocate_iterable([inp_text * 0.0, inp_audio]))
            pred_text = pred_text.detach().cpu().numpy()
            pred_audio = pred_audio.detach().cpu().numpy()
            target_sentences += [' '.join(word_dictionary.one_hot_decode(out_sentence)) for out_sentence in out_text]
            pred_sentences += [' '.join(word_dictionary.one_hot_decode(pred_sentence)) for pred_sentence in pred_text]
        word_error_rate = wer(pred_sentences, target_sentences)
    return word_error_rate


def evaluate_model(model):
    print('evaluation wer: {}'.format(calculate_wer(model, dataset_loader, librispeech_dataset.text_dataset.word_dictionary)))


net.train()
trainer.fill_missing_train_stt(dataset_loader, [1.0, 0.0], 250, (sample_count // batch_size) - 1, epoch_callback=evaluate_model)
elapsed_time = time.time() - start
print('time took to train: {}'.format(elapsed_time))
net.save(model_save_path)
net.eval()

evaluate_model(net)


while True:
    try:
        num = int(input('insert number: '))
        x_text, x_audio, y_text, y_audio = spoken_text_dataset[num]
        print(torch.argmax(x_text, dim=-1))
        text_result, audio_result = net(allocate_iterable([x_text.unsqueeze(0) * 0.0, x_audio.unsqueeze(0)]))
        text_result = text_result.detach().cpu().numpy()[0]
        print('top index: {}, best_score: {}'.format(np.argmax(text_result, axis=-1), np.max(text_result, axis=-1)))
        audio_result = audio_result.detach().cpu().numpy()[0]
        decoded_input_text = librispeech_dataset.text_dataset.word_dictionary.one_hot_decode(x_text)
        decoded_text = librispeech_dataset.text_dataset.word_dictionary.one_hot_decode(text_result)
        print('target: {}\npredicted: {}'.format(decoded_input_text, decoded_text))
    except Exception as e:
        print(e)

"""
TO DO:
-Create training pipeline: training the model with the 3 possible options
    
-Create pre-processing pipeline: tokenizing and embedding text, turning audio waves to spectograms or mfccs
-Find fine tuned fixes to transformer (layer norms, residuals, better encoders for domains)
-Split transformer logic: (CDA in different class, receive per_domain_embeddings and attention as parameters)
-Collect data
-Unit test different components (Optional but important)
-Run entire pipeline
-Visualize results
-Invesigate leads:
    -Maybe add some sort of quantitive collapse to text, and re-run the model several times
-Optimize calculations
"""
