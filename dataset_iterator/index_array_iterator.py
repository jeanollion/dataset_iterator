#from keras_preprocessing.image import Iterator
from tensorflow.keras.preprocessing.image import Iterator
import numpy as np

class IndexArrayIterator(Iterator):
    def __init__(self, n, batch_size, shuffle, seed):
        super().__init__(n, batch_size, shuffle, seed)
        self.allowed_indexes=np.arange(self.n)

    def set_allowed_indexes(self, indexes):
        self.allowed_indexes=indexes
        self.n=len(indexes)
        self.index_array=None

    def _set_index_array(self):
        if self.shuffle:
            self.index_array = np.random.permutation(self.allowed_indexes)
        else:
            self.index_array = np.copy(self.allowed_indexes)

# class ConcatenatedCategoryIterator(Iterator):
#     def __init__(self, *iterators):
#         self.iterators = iterators
#         batch_size = sum([it.batch_size for it in iterators])
#         n = min([it.n for it in iterators])
#         super().__init__(n, batch_size, False, 0)
#
#     def _set_index_array(self):
#         for it in self.iterators:
#             it._set_index_array()
#
#     def __getitem__(self, idx):
#         if idx >= len(self):
#             raise ValueError('Asked to retrieve element {idx}, '
#                              'but the Sequence '
#                              'has length {length}'.format(idx=idx,
#                                                           length=len(self)))
#         if self.seed is not None:
#             np.random.seed(self.seed + self.total_batches_seen)
#         self.total_batches_seen += 1
#         if self.index_array is None:
#             self._set_index_array()
#         index_array = self.index_array[self.batch_size * idx:
#                                        self.batch_size * (idx + 1)]
#         return self._get_batches_of_transformed_samples(index_array)
#
#     def __len__(self):
#         return min([len(it) for it in iterators])
#
#     def on_epoch_end(self):
#         for it in self.iterators:
#             it.on_epoch_end()
#
#     def reset(self):
#         for it in self.iterators:
#             it.reset()
#
#     def _flow_index(self): #TODO
#         # Ensure self.batch_index is 0.
#         self.reset()
#         while 1:
#             if self.seed is not None:
#                 np.random.seed(self.seed + self.total_batches_seen)
#             if self.batch_index == 0:
#                 self._set_index_array()
#
#             if self.n == 0:
#                 # Avoiding modulo by zero error
#                 current_index = 0
#             else:
#                 current_index = (self.batch_index * self.batch_size) % self.n
#             if self.n > current_index + self.batch_size:
#                 self.batch_index += 1
#             else:
#                 self.batch_index = 0
#             self.total_batches_seen += 1
#             yield self.index_array[current_index:
#                                    current_index + self.batch_size]
#
#     def __iter__(self): #TODO
#         # Needed if we want to do something like:
#         # for x, y in data_gen.flow(...):
#         return self
#
#     def __next__(self, *args, **kwargs): #TODO
#         return self.next(*args, **kwargs)
#
#     def next(self): #TODO
#         """For python 2.x.
#         # Returns
#             The next batch.
#         """
#         with self.lock:
#             index_array = next(self.index_generator)
#         # The transformation of images is not under thread lock
#         # so it can be done in parallel
#         return self._get_batches_of_transformed_samples(index_array)
#
#     def _get_batches_of_transformed_samples(self, index_array): #TODOp0mp3tt3!
#
#         """Gets a batch of transformed samples.
#         # Arguments
#             index_array: Array of sample indices to include in batch.
#         # Returns
#             A batch of transformed samples.
#         """
#         raise NotImplementedError
