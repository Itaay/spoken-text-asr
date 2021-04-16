import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset

from pre_processing.utils import assure_directory
from utils.device import device, allocate_iterable
NUM_TYPE = torch.float32


def my_cross_entropy_loss(y_true, y_pred):
    cross_entropy = -y_true * torch.log(y_pred + 0.000000001)
    per_sample_sum = torch.sum(torch.flatten(cross_entropy, start_dim=-2), dim=-1)
    return torch.mean(per_sample_sum)


class ConcatenatedDomainsLoss:
    def __init__(self, loss_func, domain_loss_weights=None):
        self.unit_loss = loss_func
        self.domain_loss_weights = [0.5, 0.5]
        if domain_loss_weights is not None:
            self.domain_loss_weights = domain_loss_weights

    def __call__(self, y_true, y_pred):
        if type(self.unit_loss) is list:
            domains_loss = [self.unit_loss[i](y_true[i], y_pred[i]) * self.domain_loss_weights[i] for i in range(len(y_true))]
        else:
            domains_loss = [self.unit_loss(y_true[i], y_pred[i]) * self.domain_loss_weights[i] for i in range(len(y_true))]
        total_loss = sum(domains_loss)
        return total_loss


class SpokenTextDataset(Dataset):
    def __init__(self):
        self.input_text_samples = None
        self.input_audio_samples = None
        self.output_text_samples = None
        self.output_audio_samples = None
        self.samples_count = 0

    def add_text_samples(self, text_samples, audio_dims):
        input_text = torch.tensor(text_samples, dtype=NUM_TYPE)
        batch_audio_dims = [input_text.size()[0]] + list(audio_dims)
        input_audio = torch.zeros(batch_audio_dims, dtype=NUM_TYPE)
        output_text = input_text.detach()
        output_audio = input_audio.detach()
        self.add_samples(input_text, input_audio, output_text, output_audio)

    def add_audio_samples(self, audio_samples, text_dims):
        input_audio = torch.tensor(audio_samples, dtype=NUM_TYPE)
        batch_text_dims = [input_audio.size()[0]] + list(text_dims)
        input_text = torch.zeros(batch_text_dims, dtype=NUM_TYPE)
        output_audio = input_audio.detach()
        output_text = input_text.detach()
        self.add_samples(input_text, input_audio, output_text, output_audio)

    def add_stt_samples(self, text_samples, audio_samples, text_grad_weight=1.0, audio_grad_weight=1.0):
        sum_of_grad_weights = text_grad_weight + audio_grad_weight
        text_grad_weight /= sum_of_grad_weights
        audio_grad_weight /= sum_of_grad_weights

        input_text = torch.tensor(text_samples, dtype=NUM_TYPE)
        input_audio = torch.tensor(audio_samples, dtype=NUM_TYPE)
        output_text = torch.tensor(text_samples, dtype=NUM_TYPE)
        #   output_text.register_hook(lambda grad: grad * text_grad_weight)
        output_audio = torch.tensor(audio_samples, dtype=NUM_TYPE)
        #   output_audio.register_hook(lambda grad: grad * audio_grad_weight)
        self.add_samples(input_text, input_audio, output_text, output_audio)

    def add_samples(self, input_text, input_audio, output_text, output_audio):
        if self.samples_count == 0:
            self.input_text_samples = input_text
            self.input_audio_samples = input_audio
            self.output_text_samples = output_text
            self.output_audio_samples = output_audio
        else:
            self.input_text_samples = torch.cat([self.input_text_samples, input_text], dim=0)
            self.input_audio_samples = torch.cat([self.input_audio_samples, input_audio], dim=0)
            self.output_text_samples = torch.cat([self.output_text_samples, output_text], dim=0)
            self.output_audio_samples = torch.cat([self.output_audio_samples, output_audio], dim=0)
        self.samples_count += input_text.size()[0]

    def __len__(self):
        return self.samples_count

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input_text = self.input_text_samples[idx]
        input_audio = self.input_audio_samples[idx]
        output_text = self.output_text_samples[idx]
        output_audio = self.output_audio_samples[idx]
        return input_text, input_audio, output_text, output_audio

    def save(self, root_path):
        assure_directory(root_path)
        file_type = 'tensor'

        items_to_save = {'txt_inp': self.input_text_samples, 'audio_inp': self.input_audio_samples,
         'txt_out': self.output_text_samples, 'audio_out': self.output_audio_samples}

        for file_name in items_to_save:
            torch.save(items_to_save[file_name], os.path.join(root_path, file_name + '.' + file_type))

    def load(self, root_path):
        file_type = 'tensor'
        self.input_text_samples = torch.load(os.path.join(root_path, 'txt_inp' + '.' + file_type))
        self.input_audio_samples = torch.load(os.path.join(root_path, 'audio_inp' + '.' + file_type))
        self.output_text_samples = torch.load(os.path.join(root_path, 'txt_out' + '.' + file_type))
        self.output_audio_samples = torch.load(os.path.join(root_path, 'audio_out' + '.' + file_type))


class ConcatenatedDomainsTrainer:
    def __init__(self, model, domain_loss_weights=None):
        self.model = model
        self.number_of_domains = self.model.number_of_domains
        self.loss_func = ConcatenatedDomainsLoss([my_cross_entropy_loss, nn.MSELoss()], domain_loss_weights=domain_loss_weights)
        #   self.loss_func = ConcatenatedDomainsLoss([my_cross_entropy_loss, lambda y_true, y_pred: 0.0 * nn.MSELoss()(y_true, y_pred)])
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train_stt(self, train_dataset, epochs, loss_log_interval, epoch_callback=None):
        # train the stt and block the gradients using given masks (in case output labels are missing or something)
        for epoch in range(epochs):
            running_loss = 0.0
            # train loop
            for i, data in enumerate(train_dataset, 0):
                x = data[:self.number_of_domains]
                y_true = data[self.number_of_domains:]

                x = allocate_iterable(x)
                y_true = allocate_iterable(y_true)

                self.optimizer.zero_grad()
                y_pred = self.model(x)

                loss = self.loss_func(y_true, y_pred)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % loss_log_interval == loss_log_interval - 1:
                    print('[%d, %5d] loss: %.9f' % (epoch + 1, i + 1, running_loss / loss_log_interval))
                    running_loss = 0.0
            if epoch_callback:
                epoch_callback(self.model)
                self.model.train()

    def fill_missing_train_stt(self, train_dataset, input_masks_coefficients, epochs, loss_log_interval, epoch_callback=None):
        # train the stt and block the gradients using given masks (in case output labels are missing or something)

        for epoch in range(epochs):
            running_loss = 0.0
            # train loop
            for i, data in enumerate(train_dataset, 0):
                x = data[:self.number_of_domains]

                input_masks = [torch.rand(domain_input.size()) > mask_coe for domain_input, mask_coe in
                               zip(x, input_masks_coefficients)]

                x = [domain_input * input_mask for domain_input, input_mask in zip(x, input_masks)]

                y_true = data[self.number_of_domains:]

                x = allocate_iterable(x)
                y_true = allocate_iterable(y_true)

                self.optimizer.zero_grad()
                y_pred = self.model(x)

                loss = self.loss_func(y_true, y_pred)

                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % loss_log_interval == loss_log_interval - 1:
                    print('[%d, %5d] loss: %.9f' % (epoch + 1, i + 1, running_loss / loss_log_interval))
                    running_loss = 0.0
            if epoch_callback:
                epoch_callback(self.model)
                self.model.train()

    def infer(self, text_input, audio_input):
        text_input, audio_input = allocate_iterable([text_input, audio_input])
        text_output, audio_output = self.model([text_input, audio_input])
        return text_output, audio_output
