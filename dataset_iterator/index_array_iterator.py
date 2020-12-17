from tensorflow.keras.preprocessing.image import Iterator
import numpy as np
import types

IMCOMPLETE_LAST_BATCH_MODE = ["KEEP", "CONSTANT_SIZE", "REMOVE"]
class IndexArrayIterator(Iterator):
    def __init__(self, n, batch_size, shuffle, seed, incomplete_last_batch_mode=IMCOMPLETE_LAST_BATCH_MODE[0]):
        super().__init__(n, batch_size, shuffle, seed)
        self.allowed_indexes=np.arange(self.n)
        self.incomplete_last_batch_mode = IMCOMPLETE_LAST_BATCH_MODE.index(incomplete_last_batch_mode)

    def set_allowed_indexes(self, indexes):
        self.allowed_indexes=indexes
        self.n=len(indexes)
        self.index_array=None

    def __len__(self):
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
