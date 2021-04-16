import torch
from torch import nn as nn


class DeepModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, activations=None, dropout=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = []
        self.dims_list = [input_dim] + hidden_dims + [output_dim]
        self.number_of_layers = len(self.dims_list) - 1
        self.layers = nn.ModuleList()
        self.activations = activations
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            activation = self.activations
            if type(activation) is list:
                activation = activation[i]
            if activation is not None:
                x = activation(x)
            x = self.dropout(x)
        return x


class DeepLinear(DeepModel):
    def __init__(self, input_dim, output_dim, hidden_dims=None, activations=None, dropout=0.0):
        super().__init__(input_dim, output_dim, hidden_dims=hidden_dims, activations=activations, dropout=dropout)
        for i in range(self.number_of_layers):
            self.layers.append(nn.Linear(self.dims_list[i], self.dims_list[i + 1]))


class DeepConv1d(DeepModel):
    def __init__(self, input_dim, output_dim, hidden_dims=None, activations=None, dropout=0.0, kernel_sizes=3):
        super().__init__(input_dim, output_dim, hidden_dims=hidden_dims, activations=activations, dropout=dropout)
        kernel_sizes = kernel_sizes
        if type(kernel_sizes) is int:
            kernel_sizes = [kernel_sizes] * self.number_of_layers

        for i in range(self.number_of_layers):
            self.layers.append(nn.Conv1d(self.dims_list[i], self.dims_list[i + 1],
                                              kernel_sizes[i], padding=(kernel_sizes[i] - 1) // 2))

    def forward(self, x):
        x = torch.transpose(x, -2, -1)  # convoultions get the time axis as last, so swap dims temporarily
        x = super().forward(x)
        x = torch.transpose(x, -1, -2)  # flip back the dimensions
        return x


class DeepConv1dTrans(DeepModel):
    def __init__(self, input_dim, output_dim, hidden_dims=None, activations=None, dropout=0.0, kernel_sizes=3):
        super().__init__(input_dim, output_dim, hidden_dims=hidden_dims, activations=activations, dropout=dropout)
        kernel_sizes = kernel_sizes
        if type(kernel_sizes) is int:
            kernel_sizes = [kernel_sizes] * self.number_of_layers

        for i in range(self.number_of_layers):
            self.layers.append(nn.ConvTranspose1d(self.dims_list[i], self.dims_list[i + 1],
                                              kernel_sizes[i], padding=(kernel_sizes[i] - 1) // 2))

    def forward(self, x):
        x = torch.transpose(x, -2, -1)  # convoultions get the time axis as last, so swap dims temporarily
        x = super().forward(x)
        x = torch.transpose(x, -1, -2)  # flip back the dimensions
        return x


# automatically create the two sides of the embeddings: into and out of the latend space
class LinearEmbedding:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = DeepLinear(input_dim, output_dim, activations=nn.LeakyReLU(0.1))
        self.decoder = DeepLinear(output_dim, input_dim)


class ConvolutionalEmbedding:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.encoder = DeepConv1d(input_dim, output_dim, activations=nn.LeakyReLU(0.1), kernel_sizes=1)
        self.decoder = DeepConv1dTrans(output_dim, input_dim, activations=None, kernel_sizes=1)


# a wrapper to a simple linear decoder to use sigmoid to emphasize the "one-hotness" of the values (forcing them to be a one hot value)
class CategoricalLinearDecoder(nn.Module):
    def __init__(self, raw_decoding):
        super().__init__()
        self.raw_decoder = raw_decoding

    def forward(self, x):
        decoder_result = self.raw_decoder(x)
        return torch.softmax(decoder_result, dim=-1)


# a wrapper to the Linear embedder to wrap the decoder with sigmoid to match the values more to "one-hotness"
class OneHotLinearEmbedding(LinearEmbedding):
    def __init__(self, input_dim, output_dim):
        super().__init__(input_dim, output_dim)
        #   self.encoder = CategoricalLinearDecoder(self.encoder)
        self.decoder = CategoricalLinearDecoder(self.decoder)
