import os

import numpy as np


def get_context_slices_from_array(arr, ctx, stride=1, pad_value=None):
    arr_shape = list(arr.shape)
    sample_shape = arr_shape[:]
    sample_shape[0] = ctx
    stride_range = range(0, arr.shape[0] + 1 - ctx, stride)
    number_of_steps = max(len(stride_range), 1)
    samples = np.zeros([number_of_steps] + sample_shape)
    if pad_value is not None:
        samples = np.array([[pad_value]*ctx] * number_of_steps)
    for i, step in enumerate(stride_range):
        samples[i, -ctx:] = arr[step:step+ctx]
    return samples


def get_all_paths_of_type(root_path, file_type):
    paths = []
    for root, dirs, files in os.walk(root_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_path.endswith('.'+file_type):
                paths.append(file_path)
    return paths


def assure_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
