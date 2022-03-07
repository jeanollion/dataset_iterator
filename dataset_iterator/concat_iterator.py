import numpy as np
from .index_array_iterator import IndexArrayIterator, IMCOMPLETE_LAST_BATCH_MODE
from .utils import ensure_multiplicity, flatten_list

class ConcatIterator(IndexArrayIterator):
    def __init__(self,
            iterators:list,
            batch_size:int,
            proportion:list=None,
            shuffle:bool=True,
            seed = None,
            incomplete_last_batch_mode:str=IMCOMPLETE_LAST_BATCH_MODE[0]):
        assert isinstance(iterators, (list, tuple)), "iterators must be either list or tuple"
        self.iterators = iterators
        if proportion is None:
            self.proportion = [1./len(iterators) for i in iterators]
        else:
            self.proportion = ensure_multiplicity(len(iterators), proportion)

        it_len = np.array([len(it) for it in self.iterators])
        for i in range(1, len(it_len)):
            it_len[i]=it_len[i-1]+it_len[i]
        self.it_cumlen=it_len
        self.it_off=np.insert(self.it_cumlen[:-1], 0, 0)
        super().__init__(-1, batch_size, shuffle, seed, incomplete_last_batch_mode)

    def _set_index_array(self):
        indices_per_iterator = [np.random.randint(low=self.it_off[i], high=self.it_cumlen[i], size=int((self.it_cumlen[i] - self.it_off[i])*self.proportion[i]+0.5) ) for i in range(len(self.iterators))]
        index_a = np.concatenate(indices_per_iterator)
        if self.shuffle:
            self.index_array = np.random.permutation(index_a)
        else:
            self.index_array = index_a
        self.n = len(index_a)

    def __len__(self):
        if self.n<0:
            self._set_index_array() # also set self.n
        return super().__len__()

    def _get_batches_of_transformed_samples(self, index_array):
        index_array = np.copy(index_array) # so that main index array is not modified
        index_it = self._get_it_idx(index_array) # modifies index_array

        batches = [self.iterators[it][i] for i, it in zip(index_array, index_it)]
        for i in range(1, len(batches)):
            assert len(batches[i])==len(batches[0]), f"Iterators have different outputs: batch from iterator {index_it[0]} has length {len(batches[0])} whereas batch from iterator {index_it[i]} has length {batches[i]}"
        # concatenate batches
        if len(batches[0]) == 2:
            inputs = [b[0] for b in batches]
            outputs = [b[1] for b in batches]
            return (concat_numpy_arrays(inputs), concat_numpy_arrays(outputs))
        else:
            return concat_numpy_arrays(batches)

    def _get_it_idx(self, index_array): # !! modifies index_array
        it_idx = np.searchsorted(self.it_cumlen, index_array, side='right')
        index_array -= self.it_off[it_idx] # remove ds offset to each index
        return it_idx

    def set_allowed_indexes(self, indexes):
        raise NotImplementedError("Not supported yet")

    def _close_datasetIO(self):
        for it in self.iterators:
            it._close_datasetIO()
            
def concat_numpy_arrays(arrays):
    if isinstance(arrays[0], (list, tuple)):
        n = len(arrays[0])
        for i in range(1, len(arrays)):
            assert len(arrays[i])==n, "Iterators have different outputs"
        return [np.concatenate([ a[i] for a in arrays], 0) for i in range(n)]
    else:
        return np.concatenate(arrays, 0)
