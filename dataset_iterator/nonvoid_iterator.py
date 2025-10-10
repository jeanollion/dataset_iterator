from math import isclose
import numpy as np
from .index_array_iterator import IndexArrayIterator, INCOMPLETE_LAST_BATCH_MODE
from .utils import ensure_multiplicity, ensure_size

# iterator that iterates until all batch items contain at least a segmented pixel
class NonVoidIterator(IndexArrayIterator):
    def __init__(self,
                 iterator:IndexArrayIterator,
                 mask_channel_idx:int, # index in the input or output batch
                 mask_is_input:bool = False,
                 pix_thld:int = 10
                 ):
        self.iterator = iterator
        self.mask_channel_idx = mask_channel_idx
        self.mask_is_input = mask_is_input
        self.pix_thld = pix_thld
        super().__init__(iterator.n, iterator.batch_size, iterator.shuffle, iterator.seed, iterator.incomplete_last_batch_mode, step_number=iterator.step_number)

    def set_allowed_indexes(self, indexes):
        self.iterator.set_allowed_indexes(indexes)

    def _get_index_array(self, choice: bool = True):
        return self.iterator._get_index_array(choice)

    def get_sample_number(self):
        return self.iterator.get_sample_number()

    def get_batch_size(self):
        return self.iterator.get_batch_size()

    def set_index_probability(self, value, n_tiles=1):
        self.iterator.set_index_probability(value, n_tiles=n_tiles)

    def open(self):
        self.iterator.open()

    def close(self, force: bool = False):
        self.iterator.close()

    def enqueuer_init(self):
        self.iterator.enqueuer_init()

    def enqueuer_end(self, params):
        self.iterator.enqueuer_end(params)

    def __len__(self):
        return self.iterator.__len__()

    def __getitem__(self, idx):
        ## LOGIC : get batch, and pick from next batches if necessary ##
        batch = self.iterator[idx]
        batch, (nn, tot) = self._select_non_void(batch)
        if nn == tot:
            return batch
        b_list = [batch] if nn>0 else []
        while nn < tot:
            idx = (idx + 1) % len(self.iterator)
            batch = self.iterator[idx]
            batch, (nn2, _) = self._select_non_void(batch)
            if nn2 > 0:
                b_list.append(batch)
                nn += nn2

        # concat
        b_in = self._concat_batch([b_item[0] for b_item in b_list])
        b_in = b_in[:tot]
        if len(batch) == 2: # input & output
            b_out = self._concat_batch([b_item[1] for b_item in b_list])
            b_out = b_out[:tot]
            return b_in, b_out
        else:
            return b_in,

    @staticmethod
    def _concat_batch(batch_list):
        if isinstance(batch_list[0], (tuple, list)):
            res = []
            for i in range(len(batch_list[0])):
                res.append(np.concatenate( [b_item[i] for b_item in batch_list], axis=0 ))
            return tuple(res) if isinstance(batch_list[0], tuple) else res
        else:
            return np.concatenate( batch_list , axis=0 )

    def _select_non_void(self, batch):
        if not self.mask_is_input and len(batch)==1:
            raise ValueError("Mask channel is output but there is no output batch")
        if len(batch)==2:
            b_in, b_out = batch
            if self.mask_is_input:
                if isinstance(b_in, (list, tuple)):
                    bmask = b_in[self.mask_channel_idx]
                else:
                    bmask = b_in
                    assert self.mask_channel_idx == 0, "invalid mask channel idx: there is only one input"
            else:
                if isinstance(b_out, (list, tuple)):
                    bmask = b_out[self.mask_channel_idx]
                else:
                    bmask = b_out
                    assert self.mask_channel_idx == 0, "invalid mask channel idx: there is only one output"
            idxs = self.get_non_void_indices(bmask, self.pix_thld)
            if idxs.shape[0] == 0:
                return (None, None), (0, bmask.shape[0])
            return (self.subset_batch(b_in, idxs), self.subset_batch(b_out, idxs)), (idxs.shape[0], bmask.shape[0])
        else:
            b_in,  = batch
            assert self.mask_is_input, "no output batch"
            if isinstance(b_in, (list, tuple)):
                bmask = b_in[self.mask_channel_idx]
            else:
                bmask = b_in
                assert self.mask_channel_idx == 0, "invalid mask channel idx: there is only one input"
            idxs = self.get_non_void_indices(bmask, self.pix_thld)
            if idxs.shape[0] == 0:
                return (None,), (0, bmask.shape[0])
            return (self.subset_batch(b_in, idxs),), (idxs.shape[0], bmask.shape[0])


    @staticmethod
    def subset_batch(batch, idxs):
        if isinstance(batch, (list, tuple)):
            bsub = [b[idxs] for b in batch]
            return tuple(bsub) if isinstance(batch, tuple) else bsub
        else:
            return batch[idxs]

    @staticmethod
    def get_non_void_indices(tensor, thld):
        # Sum along all axes except the first (batch dimension)
        sum_along_batches = np.sum(tensor, axis=tuple(range(1, tensor.ndim)))
        # Get indices where sum is > 0
        non_zero_batch_indices = np.where(sum_along_batches > thld)[0]
        #if non_zero_batch_indices.shape[0] > 0:
        #    print(f"counts {sum_along_batches[non_zero_batch_indices]}")
        return non_zero_batch_indices

    def _get_batches_of_transformed_samples(self, index_array):
        self.iterator._get_batches_of_transformed_samples(index_array)

    def _set_index_array(self):
        self.iterator._set_index_array()

    def _ensure_step_number(self):
        self.iterator._ensure_step_number()

    def disable_random_transforms(self, data_augmentation: bool = True, channels_postprocessing: bool = False):
        return self.iterator.disable_random_transforms(data_augmentation, channels_postprocessing)

    def enable_random_transforms(self, parameters):
        self.iterator.enable_random_transforms(parameters)

    def on_epoch_end(self):
        self.iterator.on_epoch_end()

    def reset(self):
        self.iterator.reset()

    @property
    def batch_size(self):
        return self.iterator._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.iterator.batch_size = value

    @property
    def step_number(self):
        return self.iterator._step_number

    @step_number.setter
    def step_number(self, value):
        self.iterator.step_number = value

    @property
    def n(self):
        return self.iterator.n

    @n.setter
    def n(self, value):
        assert value == self.iterator.n, "cannot set n"

    @property
    def index_probability(self):
        return self.iterator._index_probability

    @index_probability.setter
    def index_probability(self, value):
        self.iterator.index_probability = value
