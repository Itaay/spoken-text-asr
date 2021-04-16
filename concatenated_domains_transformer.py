import numpy as np
import torch
import torch.nn as nn


class ConcatenatedDomainsTransformer(nn.Module):
    def __init__(self, config):
        super(ConcatenatedDomainsTransformer, self).__init__()
        self.config = config
        self.input_embeddings = nn.ModuleList()
        self.positional_embeddings = nn.ParameterList()
        self.number_of_domains = len(self.config['domains'])
        current_domain_dims = []
        for domain in self.config['domains']:
            ctx_size, input_dims, embed_dims = domain
            self.input_embeddings.append(nn.Linear(input_dims, embed_dims))
            self.positional_embeddings.append(nn.Parameter(torch.zeros([ctx_size, embed_dims], dtype=torch.float32)))
            current_domain_dims.append(embed_dims)

        self.attention_blocks = []

        self.keys_weights = nn.ModuleList()
        self.query_weights = nn.ModuleList()
        self.value_weights = nn.ModuleList()
        self.attention_blocks_output_weights = nn.ModuleList()

        for attention_dims in self.config['attention_blocks']:
            domain_key_weights = nn.ModuleList()
            domain_query_weights = nn.ModuleList()
            domain_value_weights = nn.ModuleList()
            domain_output_weights = nn.ModuleList()

            for i, domain_dim in enumerate(current_domain_dims):
                domain_key_weights.append(nn.Linear(domain_dim, attention_dims))
                domain_query_weights.append(nn.Linear(domain_dim, attention_dims))
                domain_value_weights.append(nn.Linear(domain_dim, attention_dims))
                domain_output_weights.append(nn.Linear(attention_dims, domain_dim))
                current_domain_dims[i] = domain_dim

            self.keys_weights.append(domain_key_weights)
            self.query_weights.append(domain_query_weights)
            self.value_weights.append(domain_value_weights)
            self.attention_blocks_output_weights.append(domain_output_weights)

            scale_factor = np.sqrt(attention_dims)
            self.attention_blocks.append(
                (domain_key_weights, domain_query_weights, domain_value_weights, domain_output_weights, scale_factor))

        self.output_weights = nn.ModuleList()
        for domain_dim, current_domain_dim in zip(self.config['output_dims'], current_domain_dims):
            self.output_weights.append(nn.Linear(current_domain_dim, domain_dim))

    def forward(self, x):
        # get context sizes for each domain
        context_sizes = [domain.size()[-2] for domain in x]
        #   concatenated_indices = [0] + [sum(context_sizes[:i + 1]) for i in range(len(context_sizes))]

        # embed domains
        x = [embed_layer(domain) for domain, embed_layer in zip(x, self.input_embeddings)]
        # positionally embed domains
        x = [domain + positional_embed for domain, positional_embed in zip(x, self.positional_embeddings)]
        # for each attention block(layer)
        for attention_block in self.attention_blocks:
            #   calculate keys, queries and values for each domain
            key_weights, query_weights, value_weights, domain_output_weights, scale_factor = attention_block
            keys = [domain_key_weights(domain) for domain, domain_key_weights in zip(x, key_weights)]
            queries = [domain_query_weights(domain) for domain, domain_query_weights in zip(x, query_weights)]
            values = [domain_values_weights(domain) for domain, domain_values_weights in zip(x, value_weights)]
            #   afterwards, concatenate all keys, queries and values
            concat_keys = torch.cat(keys, dim=-2)
            concat_queries = torch.cat(queries, dim=-2)
            concat_values = torch.cat(values, dim=-2)
            #   dot product the key matrix with the query matrix
            match_matrix = torch.bmm(concat_queries, torch.transpose(concat_keys, -2, -1))
            scaled_match_matrix = match_matrix / scale_factor
            fixed_match_matrix = torch.softmax(scaled_match_matrix, dim=-2)

            # multiply the match matrix by the values matrix
            evaluated_match_matrix = torch.bmm(fixed_match_matrix, concat_values)

            #   split the matching matrix along the query axis for each domain (The axis where the dimensions of the query matrix were the context size)
            domain_specific_match_matrix = torch.split(evaluated_match_matrix, context_sizes, dim=-2)

            block_output = [domain_output_weight(domain) for domain, domain_output_weight in
                            zip(domain_specific_match_matrix, domain_output_weights)]

            new_x = [nn.ReLU()(domain) for domain in block_output]

            x = [old_domain + new_domain for old_domain, new_domain in zip(x, new_x)]

        x = [domain_output_layer(domain) for domain, domain_output_layer in zip(x, self.output_weights)]
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
