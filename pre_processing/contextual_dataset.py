from pre_processing.cached_dataset import CachedDataSet
import numpy as np

from pre_processing.utils import get_context_slices_from_array


class ContextualDataSet(CachedDataSet):
    def __init__(self, sample_paths, cache_path=None):
        super().__init__(cache_path)
        self.sample_paths = sample_paths

    def get_samples_with_context_size(self, ctx, stride=1):
        return np.concatenate(self.get_samples_with_context_size_per_path(ctx, stride), axis=0)

    def get_samples_with_context_size_per_path(self, ctx, stride=1, pad_value=None):
        samples = []
        for path in self.sample_paths:
            backup = self.load_backup(path, ctx, stride)
            if backup is not None:
                context_slices = backup
            else:
                sample = self.get_path_sample(path)
                context_slices = get_context_slices_from_array(sample, ctx, stride=stride, pad_value=pad_value)
                self.save_backup(context_slices, path, ctx, stride)
            samples.append(context_slices)
        return samples

    def get_path_sample(self, path):
        raise NotImplementedError()
