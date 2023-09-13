from math import ceil
import tensorflow as tf
import numpy as np
from .utils import ensure_size

INCOMPLETE_LAST_BATCH_MODE = ["KEEP", "CONSTANT_SIZE", "REMOVE"]
class IndexArrayIterator(tf.keras.preprocessing.image.Iterator):
    def __init__(self, n, batch_size, shuffle, seed, incomplete_last_batch_mode=INCOMPLETE_LAST_BATCH_MODE[1], step_number:int=0):
        super().__init__(n, batch_size, shuffle, seed)
        self.allowed_indexes=np.arange(self.n)
        self.incomplete_last_batch_mode = INCOMPLETE_LAST_BATCH_MODE.index(incomplete_last_batch_mode)
        self.step_number=step_number

    def set_allowed_indexes(self, indexes):
        if isinstance(indexes, int):
            self.n = indexes
            self.allowed_indexes=np.arange(self.n)
        else:
            self.allowed_indexes=indexes
            self.n=len(indexes)
        self.index_array=None

    def __len__(self):
        if self.step_number > 0:
            return self.step_number
        if self.incomplete_last_batch_mode==2:
            return self.n // self.batch_size
        else:
            return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def __getitem__(self, idx):
        length = len(self)
        if idx >= length:
            raise ValueError('Asked to retrieve element {idx}, but the Sequence has length {length}'.format(idx=idx,length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        if idx==length-1 and self.incomplete_last_batch_mode==1:
            index_array = self.index_array[-self.batch_size:]
        else:
            index_array = self.index_array[self.batch_size * idx:self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        pass

    def _set_index_array(self):
        pass

    def _ensure_step_number(self):
        step_number = self.step_number
        if self.step_number <= 0:
            if len(self.index_array) < self.batch_size :
                step_number = 1
        self.index_array = ensure_size(self.index_array, step_number * self.batch_size, shuffle=self.shuffle)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self._ensure_step_number()