import numpy as np
from dataset_iterator import MultiChannelIterator
from random import random, randint

class TrackingIterator(MultiChannelIterator):
    def __init__(self,
                *args,
                channels_prev:list,
                channels_next:list,
                n_frames:int = 1,
                aug_all_frames:bool=True,
                aug_remove_prob:float = 0,
                frame_subsampling:int = 1, # either integer -> constant subsampling, callable (called at each mini batch and returns the subsampling), or interval with breaks included
                verbose:bool = False,
                **kwargs):
        """

        Parameters
        ----------
        dataset : dataset IO or convertible
        channels_prev: return previous frames for which channel
        channels_next return next frames for which channel
        n_frames: number of previous/next frames added to central frame
        aug_all_frames: whether perform data augmentation on all frames or only on central and edges
        aug_remove_prob: probability that previous/next frames equals current frame (simulate static)
        frame_subsampling : callable / list / integer: random spacing bewteen frames
        verbose
        kwargs
        """
        super().__init__(*args, **kwargs)
        if len(channels_next)!=len(self.channel_keywords):
            raise ValueError("length of channels_next differs from channel_keywords")
        if len(channels_prev)!=len(self.channel_keywords):
            raise ValueError("length of channels_prev differs from channel_keywords")
        #if mask_channels is not None and len(mask_channels)>0:
        #    if any(channels_prev) and not channels_prev[mask_channels[0]]:
        #        raise ValueError("Previous time point of first mask channel should be returned if previous time point from another channel is returned")
        #    if any(channels_next) and not channels_next[mask_channels[0]]:
        #        raise ValueError("Next time point of first mask channel should be returned if next time point from another channel is returned")
        self.verbose=verbose
        self.channels_prev=channels_prev
        self.channels_next=channels_next
        self.aug_remove_prob = aug_remove_prob # set current image as prev / next
        self.def_n_frames=n_frames
        self.aug_all_frames=aug_all_frames
        if callable(frame_subsampling):
            self.frame_subsampling = frame_subsampling
        elif isinstance(frame_subsampling, (list, tuple)):
            assert len(frame_subsampling) == 2, "if tuple/list frame_subsampling should be of length 2"
            assert frame_subsampling[0]<=frame_subsampling[1] and frame_subsampling[0]>=0, "invalid interval for frame_subsampling"
            def fs(): # no lambda for pickling
                return randint(frame_subsampling[0], frame_subsampling[1])
            self.frame_subsampling = fs
        elif frame_subsampling is None or frame_subsampling<=1:
            def fs():
                return 1
            self.frame_subsampling = fs
        else:
            def fs():
                return frame_subsampling
            self.frame_subsampling = fs


    def _get_batch_by_channel(self, index_array, perform_augmentation, input_only=False, perform_elasticdeform=True, perform_tiling=True, **kwargs):
        if "n_frames" not in kwargs:
            if self.aug_remove_prob>0 and random() < self.aug_remove_prob:
                kwargs.update({"n_frames":0}) # flag aug remove
        if "frame_subsampling" not in kwargs:
            kwargs.update({"frame_subsampling":self.frame_subsampling()})
        res = super()._get_batch_by_channel(index_array, perform_augmentation, input_only, perform_elasticdeform, perform_tiling, **kwargs)
        return res

    def _get_frames_to_augment(self, img, chan_idx, aug_params):
        if self.aug_all_frames:
            return list(range(img.shape[-1]))
        n_frames = (img.shape[-1]-1)//2 if self.channels_prev[chan_idx] and self.channels_next[chan_idx] else img.shape[-1]-1
        if self.channels_prev[chan_idx] and self.channels_next[chan_idx]:
            return [0, n_frames, img.shape[-1]-1]
        elif self.channels_prev[chan_idx] or self.channels_next[chan_idx]:
            return [0, img.shape[-1]-1]
        else:
            return [0]

    def _apply_augmentation(self, img, chan_idx, aug_params): # apply separately for prev / cur / next
        frames_to_augment = self._get_frames_to_augment(img, chan_idx, aug_params)
        if self.aug_all_frames:
            return super()._apply_augmentation(img, chan_idx, aug_params)
        else:
            img[..., frames_to_augment] = super()._apply_augmentation(img[..., frames_to_augment], chan_idx, aug_params)
            return img

    def _read_image_batch(self, index_ds, index_array, chan_idx, ref_chan_idx, aug_param_array, is_array=False, **kwargs):
        batch, index_a = super()._read_image_batch(index_ds, index_array, chan_idx, ref_chan_idx, aug_param_array, is_array=is_array, **kwargs)
        batch_list= []
        index_array_list = []
        n_frames = kwargs.get("n_frames", self.def_n_frames if self.def_n_frames>0 else 1)
        subsampling = kwargs.get("frame_subsampling", 1)
        aug_remove = n_frames<=0
        if n_frames<=0:
            n_frames = self.def_n_frames

        batch_list.append(batch)
        index_array_list.append(index_a)
        if self.verbose:
            print(f"read n_frames={n_frames} (aug remove: {aug_remove}) -> {index_array}")
        if (not is_array and self.channels_prev[chan_idx]) or (is_array and self.channels_prev[ref_chan_idx]):
            for increment in range(1, n_frames+1):
                neigh = self._read_image_batch_neigh(index_ds, index_array, chan_idx, ref_chan_idx, True, aug_param_array, increment * subsampling, aug_remove, is_array, **kwargs)
                if neigh is None: # repeat previous one
                    neigh = batch_list[-1]
                    index_array_neigh = index_array_list[-1]
                else:
                    neigh, index_array_neigh = neigh
                batch_list.append(neigh)
                index_array_list.append(index_array_neigh)
            batch_list = batch_list[::-1] # reverse order -> previous frames first
            index_array_list = index_array_list[::-1]

        if (not is_array and self.channels_next[chan_idx]) or (is_array and self.channels_next[ref_chan_idx]):
            for increment in range(1, n_frames+1):
                neigh = self._read_image_batch_neigh(index_ds, index_array, chan_idx, ref_chan_idx, False, aug_param_array, increment * subsampling, aug_remove, is_array=is_array, **kwargs)
                if neigh is None: # repeat previous one
                    neigh = batch_list[-1]
                    index_array_neigh = index_array_list[-1]
                else:
                    neigh, index_array_neigh = neigh
                batch_list.append(neigh)
                index_array_list.append(index_array_neigh)
        if len(batch_list)>1:
            if is_array:
                return np.stack(batch_list, axis=-1), None
            index_a = np.concatenate(index_array_list, axis=-1) if self.return_image_index else None
            return np.concatenate(batch_list, axis=-1), index_a
        else:
            return batch, index_a

    def _get_max_increment(self, ds_idx, im_idx, c_idx, prev, increment):
        oob=False
        if prev:
            if im_idx<increment:
                increment = im_idx
                oob=True
        else:
            if im_idx+increment>=len(self.ds_array[c_idx][ds_idx]):
                increment = len(self.ds_array[c_idx][ds_idx]) - 1 - im_idx
                oob = True
        if increment==0:
            return 0,oob
        if self.labels is not None: # in this case, actual frame number can be deduced from label, and we can allow non-consecutive frames in a single dataset
            while increment>0:
                inc = -increment if prev else increment
                if get_neighbor_label(self.labels[ds_idx][im_idx], increment=inc)!=self.labels[ds_idx][im_idx+inc]:
                    increment -= 1
                    oob=True
                else:
                    return increment,oob
        return increment,oob

    def _read_image_batch_neigh(self, index_ds, index_array, chan_idx, ref_chan_idx, prev, aug_param_array, increment = 1, aug_remove = False, is_array=False, **kwargs):
        inc_kw = ('prev_inc_{}' if prev else 'next_inc_{}').format(increment)
        if chan_idx==ref_chan_idx: # record actual increment in aug_param_array so that same increment is used for all channels
            for i, (ds_idx, im_idx) in enumerate(zip(index_ds, index_array)):
                inc,oob = self._get_max_increment(ds_idx, im_idx, ref_chan_idx, prev, increment)
                if aug_remove: # neighbor image is replaced by current image as part of data augmentation + signal in order to set constant displacement map in further steps
                    aug_param_array[i][ref_chan_idx][inc_kw] = 0
                else:
                    aug_param_array[i][ref_chan_idx][inc_kw] = inc
                if oob:
                    aug_param_array[i][ref_chan_idx]['oob_inc'] = inc # flag out-of-bound
        if aug_remove:
            return None # image is replaced by same batch
        index_array = np.copy(index_array)
        inc_array = [aug_param_array[i][ref_chan_idx][inc_kw] for i in range(len(index_ds))]
        if prev:
            index_array -= inc_array
        else:
            index_array += inc_array
        if self.verbose:
            print(f"read inc={increment} -> {index_array}")
        return super()._read_image_batch(index_ds, index_array, chan_idx, ref_chan_idx, aug_param_array, is_array=is_array, **kwargs)

    def train_test_split(self, **options):
        train_iterator, test_iterator = super().train_test_split(**options)
        train_idx = train_iterator.allowed_indexes
        test_idx = test_iterator.allowed_indexes
        # remove neighboring time points that are seen by the network. only in terms of ground truth, ie depends on returned values:  previous and next frames or next frame only (displacement)
        if any(self.channels_prev): # an index visited in train_idx implies the previous one is also seen during training. to avoind that previous index being in test_idx, next indices of test_idx should remove from train_idx
            train_idx = np.setdiff1d(train_idx, self._get_neighbor_indices(test_idx, prev=False))
        if any(self.channels_next): # an index visited in train_idx implies the next one is also seen during training. to avoin that next index being in test_idx, previous indices of test_idx should remove from train_idx
            train_idx = np.setdiff1d(train_idx, self._get_neighbor_indices(test_idx, prev=True))

        train_iterator.set_allowed_indexes(train_idx)

        return train_iterator, test_iterator

    # for train test split
    def _get_neighbor_indices(self, index_array, prev):
        index_array_local = np.copy(index_array)
        ds_idx_array = self._get_ds_idx(index_array_local)
        res = []
        inc = -1 if prev else 1
        for i, (ds_idx, im_idx) in enumerate(zip(ds_idx_array, index_array_local)):
            neigh_lab = get_neighbor_label(self.labels[ds_idx][im_idx], increment=inc)
            bound_idx = 0 if prev else len(self.labels[ds_idx])-1
            if im_idx!=bound_idx and neigh_lab==self.labels[ds_idx][im_idx+inc]:
                res.append(index_array[i]+inc)
        return res

# class util methods
def get_neighbor_label(label, increment):
    frame = int(label[-5:])
    if increment<0 and frame<-increment:
        return None
    return label[:-5]+str(frame+increment).zfill(5)
