import numpy as np
import os

from pre_processing.utils import assure_directory


class CachedDataSet:
    def __init__(self, cache_path=None):
        self.cache_path = cache_path
        if self.cache_path is not None:
            assure_directory(self.cache_path)

    def does_cache_match_requirement(self, wanted_path, ctx, stride):
        # cache_path is a folder which contains the data for each path.
        # for each path, there will be a file in the cache, containing it's context_sliced_spectogram.
        # the file_cache path will be of the format: [ctx]_[stride]_[file_path].bk
        if self.cache_path is None:
            return False
        return os.path.exists(self.get_cache_path_for_file(wanted_path, ctx, stride))

    def get_cache_path_for_file(self, file_path, ctx, stride):
        path_format = file_path.replace('/', '-').replace('\\', '=')
        return os.path.join(self.cache_path, '{}_{}_{}.bk.npy'.format(ctx, stride, path_format))

    def load_backup(self, path, ctx, stride):
        if self.does_cache_match_requirement(path, ctx, stride):
            cache_path = self.get_cache_path_for_file(path, ctx, stride)
            return np.load(cache_path)
        return None

    def save_backup(self, res, path, ctx, stride):
        if self.cache_path is not None:
            backup_path = self.get_cache_path_for_file(path, ctx, stride)
            np.save(backup_path, res)
