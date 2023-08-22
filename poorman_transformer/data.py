import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset


class TokenManager:
    def __init__(self, token_lib):
        self.token_lib      = token_lib

        # Create token to integer mapping...
        self.token_to_int = { token : enum_idx for enum_idx, token in enumerate(self.token_lib) }
        self.int_to_token = { enum_idx : token for enum_idx, token in enumerate(self.token_lib) }


    def encode(self, data):
        return [ self.token_to_int[each_data] for each_data in data ]


    def decode(self, data):
        return ''.join([ self.int_to_token[each_data] for each_data in data ])




class TinyShakespearDataset(Dataset):
    def __init__(self, data_source, context_length, sample_size):
        self.data_source    = data_source
        self.context_length = context_length
        self.sample_size    = sample_size

        self.idx_list = []
        self.update_random_dataset()


    def update_random_dataset(self):
        data_source    = self.data_source
        context_length = self.context_length
        sample_size    = self.sample_size

        # Create the dataset...
        # The rightmost context window is used as a target
        self.idx_list = random.choices(list(range(len(data_source) - context_length)), k = sample_size)


    def __len__(self):
        return self.sample_size


    def __getitem__(self, idx):
        data_source      = self.data_source
        context_length   = self.context_length

        sample_idx = self.idx_list[idx]

        # Get the context...
        context_min = sample_idx
        context_max = sample_idx + context_length
        context = data_source[context_min:context_max]

        # Get the target (context offset by 1)...
        target_min = sample_idx + 1
        target_max = sample_idx + context_length + 1
        target = data_source[target_min:target_max]

        return context, target
