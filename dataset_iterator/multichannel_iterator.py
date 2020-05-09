import numpy as np
from dataset_iterator import IndexArrayIterator
from .utils import remove_duplicates
from sklearn.model_selection import train_test_split
from math import ceil
import time
import copy
from math import copysign
from .datasetIO import DatasetIO, get_datasetIO
from .utils import ensure_multiplicity

class MultiChannelIterator(IndexArrayIterator):
    def __init__(self,
                dataset,
                channel_keywords=['/raw'],
                input_channels=[0],
                output_channels=[0],
                weight_map_functions=None,
                output_postprocessing_functions=None,
                extract_tile_function=None,
                mask_channels=[],
                output_multiplicity = 1,
                channel_scaling_param=None, #[{'level':1, 'qmin':5, 'qmax':95}],
                group_keyword=None,
                image_data_generators=None,
                singleton_channels=[],
                batch_size=32,
                shuffle=True,
                perform_data_augmentation=True,
                seed=None,
                dtype='float32'):
        self.dataset = dataset
        self.group_keyword=group_keyword
        self.channel_keywords=channel_keywords
        self.channel_scaling_param = channel_scaling_param
        self.dtype = dtype
        self.perform_data_augmentation=perform_data_augmentation
        self.singleton_channels=[] if singleton_channels is None else singleton_channels
        assert isinstance(self.singleton_channels, list), "singleton_channels must be a list"
        if len(self.singleton_channels)>0:
            assert max(self.singleton_channels)<len(channel_keywords), "invalid singleton_channels index (outside channel range)"
            assert min(self.singleton_channels)>0, "invalid singleton_channels index (first channel cannot be singleton)"
        if mask_channels is None:
            mask_channels = []
        self.mask_channels = mask_channels
        self.output_multiplicity=output_multiplicity
        if len(mask_channels)>0:
            assert min(mask_channels)>=0, "invalid mask channel: should be >=0"
            assert max(mask_channels)<len(channel_keywords), "invalid mask channel: should be in range [0-{})".format(len(channel_keywords))
        if output_channels is None:
            output_channels = []
        if input_channels is None or len(input_channels)==0:
            raise ValueError("No input channels set")
        if (len(input_channels) != len(set(input_channels))):
            raise ValueError("Duplicated channels in input_channels")
        self.input_channels=input_channels
        self.output_channels=output_channels # duplicated output channels allowed because can be modified by a postprocessing function
        if image_data_generators!=None and len(channel_keywords)!=len(image_data_generators):
            raise ValueError('image_data_generators argument should be either None or an array of same length as channel_keywords')
        self.image_data_generators=image_data_generators
        if weight_map_functions is not None:
            assert len(weight_map_functions)==len(output_channels), "weight map should have same length as output channels"
        self.weight_map_functions=weight_map_functions
        if output_postprocessing_functions is not None:
            assert len(output_postprocessing_functions)==len(output_postprocessing_functions), "output postprocessing functions should have same length as output channels"
        self.output_postprocessing_functions = output_postprocessing_functions
        self.extract_tile_function=extract_tile_function
        self.paths=None
        self._open_datasetIO()
        # check that all ds have compatible length between input and output
        indexes = np.array([ds.shape[0] for ds in self.ds_array[0]])
        if len(channel_keywords)>1:
            for c, ds_l in enumerate(self.ds_array):
                singleton = c in self.singleton_channels
                if len(self.ds_array[0])!=len(ds_l):
                    raise ValueError('Channels {}({}) has #{} datasets whereas first channel has #{} datasets'.format(c, channel_keywords[c], len(ds_l), len(self.ds_array[0])))
                for ds_idx, ds in enumerate(ds_l):
                    if singleton:
                        if ds.shape[0]!=1:
                            raise ValueError("Channel {} is set as singleton but one dataset has more that one image".format(c))
                    elif indexes[ds_idx] != ds.shape[0]:
                        raise ValueError('Channel {}({}) has at least one dataset with number of elements that differ from Channel 0'.format(c, channel_keywords[c]))

        # get offset for each dataset
        for i in range(1, len(indexes)):
            indexes[i]=indexes[i-1]+indexes[i]
        self.ds_len=indexes
        self.ds_off=np.insert(self.ds_len[:-1], 0, 0)

        # check that all datasets have same image shape within each channel
        self.channel_image_shapes = [ds_l[0].shape[1:] for ds_l in self.ds_array]
        for c, ds_l in enumerate(self.ds_array):
            for ds_idx, ds in enumerate(ds_l):
                if ds.shape[1:] != self.channel_image_shapes[c]:
                    raise ValueError('Dataset {dsi} with path {dspath} from channel {chan}({chank}) has shape {dsshape} that differs from first dataset with path {ds1path} with shape {ds1shape}'.format(dsi=ds_idx, dspath=self._get_dataset_path(c, ds_idx), chan=c, chank=self.channel_keywords[c], dsshape=ds.shape[1:], ds1path=self._get_dataset_path(c, 0), ds1shape=self.channel_image_shapes[c] ))

        # labels
        try:
            self.labels = [self.datasetIO.get_dataset(path.replace(self.channel_keywords[0], '/labels')) for path in self.paths]
            for i, ds in enumerate(self.labels):
                self.labels[i] = np.char.asarray(ds[()].astype('unicode')) # todo: check if necessary to convert to char array ? unicode is necessary
            if len(self.labels)!=len(self.ds_array[0]):
                raise ValueError('Invalid input file: number of label array differ from dataset number')
            if any(len(self.labels[i].shape)==0 or self.labels[i].shape[0]!=self.ds_array[0][i].shape[0] for i in range(len(self.labels))):
                raise ValueError('Invalid input file: at least one dataset has element numbers that differ from corresponding label array')
        except:
            self.labels = None
        # set scaling information for each dataset
        self.channel_scaling = [None]*len(channel_keywords)
        if self.channel_scaling_param!=None:
            percentile_x = np.arange(0, 101)
            for c, scaling_info in enumerate(self.channel_scaling_param):
                if scaling_info!=None:
                    self.channel_scaling[c]=[None]*len(self.paths)
                    for ds_idx, path in enumerate(self.paths):
                        group = self.datasetIO.get_parent_path(path)
                        for i in range(scaling_info.get('level', 1)):
                            group = self.datasetIO.get_parent_path(group)
                            if group==None:
                                raise ValueError("scaling group level too high for channel {}({}) group path: {}".format(c, channel_keywords[c]), self.datasetIO.get_parent_path(path))
                        # percentiles are located in attributes of group
                        # TODO test if this raise an error when not present
                        percentiles = self.datasetIO.get_attribute(group, channel_keywords[c].replace('/', '')+'_percentiles')
                        if percentiles is None:
                            raise ValueError("No percentile array found in group {} for channel: {}({})".format(group, c, channel_keywords[c]))

                        # get IQR and median
                        minv, med, maxv = np.interp([scaling_info.get('qmin', 5), 50, scaling_info.get('qmax', 95)], percentile_x, percentiles)
                        self.channel_scaling[c][ds_idx] = [med, maxv-minv]

        self._close_datasetIO()
        self.void_mask_max_proportion = -1
        if len(mask_channels)>0:
            self.void_mask_chan = mask_channels[0]
        else:
            self.void_mask_chan=-1
        super().__init__(indexes[-1], batch_size, shuffle, seed)

    def _open_datasetIO(self):
        self.datasetIO = get_datasetIO(self.dataset, 'r')
        if self.paths is None:
            # get all dataset paths
            self.paths = self.datasetIO.get_dataset_paths(self.channel_keywords[0], self.group_keyword)
            if (len(self.paths)==0):
                raise ValueError('No datasets found ending by {} {}'.format(self.channel_keywords[0], "and containing {}".format(self.group_keyword) if self.group_keyword!=None else "" ))
        # get all matching dataset
        self.ds_array = [[self.datasetIO.get_dataset(self._get_dataset_path(c, ds_idx)) for ds_idx in range(len(self.paths))] for c in range(len(self.channel_keywords))]
        getAttribute = lambda a, def_val : def_val if a is None else (a[0] if isinstance(a, list) else a)
        self.ds_scaling_center = [[getAttribute(self.datasetIO.get_attribute(self._get_dataset_path(c, ds_idx), "scaling_center"), 0) for ds_idx in range(len(self.paths))] for c in range(len(self.channel_keywords))]
        self.ds_scaling_factor = [[getAttribute(self.datasetIO.get_attribute(self._get_dataset_path(c, ds_idx), "scaling_factor"), 1) for ds_idx in range(len(self.paths))] for c in range(len(self.channel_keywords))]

    def _close_datasetIO(self):
        if self.datasetIO is not None:
            self.datasetIO.close()
            self.datasetIO = None
            self.ds_array = None

    def train_test_split(self, **options):
        """Split this iterator in two distinct iterators

        Parameters
        ----------

        **options : dictionary
            options passed to train_test_split method of scikit-learn package
            this dictionary can also contain 3 arguments passed to the constructor of the test iterator. if absent, values of the current instance will be passed to the constructor.

            suffle_test : Boolean
                whether indexes should be shuffled in test iterator
            perform_data_augmentation_test : Boolean
                wether data augmentation should be performed by the test iterator
            seed_test : Integer
                seed for test iterator

        Returns
        -------
        tuple of train and test iterators of same type as instance, that access two distinct partitions of the whole dataset.
            train iterator has the same parameters as current instance
            test iterator has the same parameters as current instance except those defined in the argument of this method
        """
        shuffle_test=options.pop('shuffle_test', self.shuffle)
        perform_data_augmentation_test=options.pop('perform_data_augmentation_test', self.perform_data_augmentation)
        seed_test=options.pop('seed_test', self.seed)
        train_idx, test_idx = train_test_split(self.allowed_indexes, **options)
        self._close_datasetIO()
        train_iterator = copy.copy(self)
        train_iterator.set_allowed_indexes(train_idx)

        test_iterator = copy.copy(self)
        test_iterator.shuffle=shuffle_test
        test_iterator.perform_data_augmentation=perform_data_augmentation_test
        test_iterator.seed=seed_test
        test_iterator.set_allowed_indexes(test_idx)

        return train_iterator, test_iterator

    def _get_ds_idx(self, index_array):
        ds_idx = np.searchsorted(self.ds_len, index_array, side='right')
        index_array -= self.ds_off[ds_idx] # remove ds offset to each index
        return ds_idx

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples (tuple of input and output if output_keyword is specified).
        """
        batch_by_channel, aug_param_array, ref_chan_idx = self._get_batch_by_channel(index_array, self.perform_data_augmentation)

        if self.output_channels is None or len(self.output_channels)==0:
            return self._get_input_batch(batch_by_channel, ref_chan_idx, aug_param_array)
        else:
            input = self._get_input_batch(batch_by_channel, ref_chan_idx, aug_param_array)
            output = self._get_output_batch(batch_by_channel, ref_chan_idx, aug_param_array)
            if self.output_multiplicity>1:
                if not isinstance(output, list):
                    output = [output] * self.output_multiplicity
                else:
                    output = output * self.output_multiplicity
            return (input, output)

    def _get_batch_by_channel(self, index_array, perform_augmentation, input_only=False):
        if self.datasetIO is None: # for concurency issues: file is open lazyly by each worker
            self._open_datasetIO()
        index_array = np.copy(index_array) # so that main index array is not modified
        index_ds = self._get_ds_idx(index_array) # modifies index_array

        batch_by_channel = dict()
        channels = self.input_channels if input_only else remove_duplicates(self.input_channels+self.output_channels)
        if len(self.mask_channels)>0 and self.mask_channels[0] in channels: # put mask channel first so it can be used for determining some data augmentation parameters
            channels.insert(0, channels.pop(channels.index(self.mask_channels[0])))
        aug_param_array = [[dict()]*len(self.channel_keywords) for i in range(len(index_array))]
        for chan_idx in channels:
            batch_by_channel[chan_idx] = self._get_batches_of_transformed_samples_by_channel(index_ds, index_array, chan_idx, channels[0], aug_param_array, perform_augmentation=perform_augmentation)

        if self.extract_tile_function is not None:
            numpy_rand_state = np.random.get_state()
            for chan_idx in channels:
                np.random.set_state(numpy_rand_state) # ensure same tile + aug if tile fun implies randomness
                batch_by_channel[chan_idx] = self.extract_tile_function(batch_by_channel[chan_idx])

        return batch_by_channel, aug_param_array, channels[0]

    def _get_input_batch(self, batch_by_channel, ref_chan_idx, aug_param_array):
        if len(self.input_channels)==1:
            return batch_by_channel[self.input_channels[0]]
        else:
            return [batch_by_channel[chan_idx] for chan_idx in self.input_channels]

    def _apply_postprocessing_and_concat_weight_map(self, batch, output_chan_idx):
        if self.weight_map_functions is not None and self.weight_map_functions[output_chan_idx] is not None:
            wm = self.weight_map_functions[output_chan_idx](batch)
        else:
            wm = None
        if self.output_postprocessing_functions is not None and self.output_postprocessing_functions[output_chan_idx] is not None:
            batch = self.output_postprocessing_functions[output_chan_idx](batch)
        if wm is not None:
            batch = np.concatenate([batch, wm], -1)
        return batch

    def _get_output_batch(self, batch_by_channel, ref_chan_idx, aug_param_array):
        if len(self.output_channels)==1:
            return self._apply_postprocessing_and_concat_weight_map(batch_by_channel[self.output_channels[0]], 0)
        else:
            return [self._apply_postprocessing_and_concat_weight_map(batch_by_channel[chan_idx], i) for i, chan_idx in enumerate(self.output_channels)]

    def _get_batches_of_transformed_samples_by_channel(self, index_ds, index_array, chan_idx, ref_chan_idx, aug_param_array=None, perform_augmentation=True, transfer_aug_param_function=lambda source, dest:copy_geom_tranform_parameters(source, dest)):
        """Generate a batch of transformed sample for a given channel

        Parameters
        ----------
        index_ds : int array
            dataset index for each image
        index_array : int array
            image index within dataset
        chan_idx : int
            index of the channel
        ref_chan_idx : int
            chanel on which aug param are initiated.
        aug_param_array : dict array
            parameters generated by the ImageDataGenerator of the input channel.
            Affine transformation parameters are generated for ref channel and shared with output batch so that same affine transform are applied to output batch

        Returns
        -------
        type
            batch of image for the channel of index chan_idx

        """

        batch = self._read_image_batch(index_ds, index_array, chan_idx, ref_chan_idx, aug_param_array)
        # apply augmentation
        image_data_generator = self.image_data_generators[chan_idx] if self.perform_data_augmentation and perform_augmentation and self.image_data_generators!=None else None
        for i in range(batch.shape[0]):
            if image_data_generator!=None:
                params = self._get_data_augmentation_parameters(chan_idx, ref_chan_idx, batch, i, index_ds, index_array)
                if aug_param_array is not None:
                    if chan_idx!=ref_chan_idx:
                        transfer_aug_param_function(aug_param_array[i][ref_chan_idx], params)
                    for k,v in params.items():
                        aug_param_array[i][chan_idx][k]=v
                batch[i] = self._apply_augmentation(batch[i], chan_idx, params)
        return batch

    def _apply_augmentation(self, img, chan_idx, aug_params):
        image_data_generator = self.image_data_generators[chan_idx]
        if image_data_generator is not None:
            img = image_data_generator.apply_transform(img, aug_params)
            img = image_data_generator.standardize(img)
        return img

    def _read_image_batch(self, index_ds, index_array, chan_idx, ref_chan_idx, aug_param_array):
        # read all images
        images = [self._read_image(chan_idx, ds_idx, im_idx) for i, (ds_idx, im_idx) in enumerate(zip(index_ds, index_array))]
        batch = np.stack(images)
        return batch

    def _get_data_augmentation_parameters(self, chan_idx, ref_chan_idx, batch, idx, index_ds, index_array):
        if self.image_data_generators is None or self.image_data_generators[chan_idx] is None:
            return None
        params = self.image_data_generators[chan_idx].get_random_transform(batch.shape[1:])
        if  chan_idx==ref_chan_idx and chan_idx in self.mask_channels:
            try:
                self.image_data_generators[chan_idx].adjust_augmentation_param_from_mask(params, batch[idx,...,0])
            except AttributeError: # data generator does not have this method
                pass
        return params

    def _read_image(self, chan_idx, ds_idx, im_idx):
        ds = self.ds_array[chan_idx][ds_idx]
        if chan_idx in self.singleton_channels:
            im_idx=0
        im = ds[im_idx]
        if len(self.channel_image_shapes[chan_idx])==2 and len(im.shape)==2:
            im = np.expand_dims(im, -1)
        im = im.astype(self.dtype)
        # apply dataset-wise scaling if information is present in attributes
        off = self.ds_scaling_center[chan_idx][ds_idx]
        factor = self.ds_scaling_factor[chan_idx][ds_idx]
        if off!=0 or factor!=1:
            im = (im - off)/factor

        # apply group-wise scaling
        off, factor = self._get_scaling(chan_idx, ds_idx)
        if off!=0 or factor!=1:
            im = (im - off) / factor
        if chan_idx in self.mask_channels:
            im[np.abs(im) < 1e-10] = 0
        return im

    def _get_scaling(self, chan_idx, ds_idx):
        if self.channel_scaling==None or self.channel_scaling[chan_idx]==None:
            return (0, 1)
        else:
            return self.channel_scaling[chan_idx][ds_idx]

    def _get_dataset_path(self, channel_idx, ds_idx):
        if channel_idx==0:
            return self.paths[ds_idx]
        else:
            return self.paths[ds_idx].replace(self.channel_keywords[0], self.channel_keywords[channel_idx])

    def inspect_indices(self, index_array):
        a = np.array(index_array, dtype=np.int)
        i = self._get_ds_idx(a)
        p = [self.paths[ii] for ii in i]
        return(a, i, p)

    def predict(self, output, model, output_keys, write_every_n_batches = 100, n_output_channels=1, output_image_shapes = None, prediction_function=None, apply_to_prediction=None, close_outputIO=True, **create_dataset_options):
        of = get_datasetIO(output, 'a')
        if output_image_shapes is None:
            output_image_shapes = self.channel_image_shapes[0]
        if not isinstance(output_keys, list):
            output_keys = [output_keys]
        if not isinstance(output_image_shapes, list):
            output_image_shapes = [output_image_shapes]
        output_image_shapes = ensure_multiplicity(len(output_keys), output_image_shapes)
        n_output_channels = ensure_multiplicity(len(output_keys), n_output_channels)
        output_shapes = [ois+(nc,) for ois, nc in zip(output_image_shapes, n_output_channels)]
        # set iterators parameters for prediction & record them
        batch_index = self.batch_index
        self.batch_index=0
        shuffle = self.shuffle
        self.shuffle=False
        perform_aug = self.perform_data_augmentation
        self.perform_data_augmentation=False
        self.reset()
        self._set_index_array() # if shuffle was true

        if np.any(self.index_array[1:] < self.index_array[:-1]):
            raise ValueError('Index array should be monotonically increasing')

        buffer = [np.zeros(shape = (write_every_n_batches*self.batch_size,)+output_shapes[oidx], dtype=self.dtype) for oidx,k in enumerate(output_keys)]
        if prediction_function is None:
            pred_fun = lambda model, input : model.predict(input)
        else:
            pred_fun = prediction_function

        for ds_i, ds_i_i, ds_i_len in zip(*np.unique(self._get_ds_idx(self.index_array), return_index=True, return_counts=True)):
            self._ensure_dataset(of, output_shapes, output_keys, ds_i, **create_dataset_options)
            paths = [self.paths[ds_i].replace(self.channel_keywords[0], output_key) for output_key in output_keys]
            index_arrays = np.array_split(self.index_array[ds_i_i:(ds_i_i+ds_i_len)], ceil(ds_i_len/self.batch_size))
            print("predictions for dataset:", self.paths[ds_i])
            unsaved_batches = 0
            buffer_idx = 0
            output_idx = 0
            #current_indices=[]
            start_pred = time.time()
            for i, index_array in enumerate(index_arrays):
                batch_by_channel, aug_param_array, ref_chan_idx = self._get_batch_by_channel(index_array, False, input_only=True)
                input = self._get_input_batch(batch_by_channel, ref_chan_idx, aug_param_array)
                cur_pred = pred_fun(model, input)
                if apply_to_prediction is not None:
                    cur_pred = apply_to_prediction(cur_pred)
                if not isinstance(cur_pred, list):
                    cur_pred = [cur_pred]
                assert len(cur_pred)==len(output_keys), 'prediction should have as many output as output_keys argument'
                for oidx in range(len(output_keys)):
                    assert cur_pred[oidx].shape[1:] == output_shapes[oidx], "prediction shape differs from output shape for output idx={} : prediction: {} target: {}".format(oidx, cur_pred[oidx].shape[1:], output_shapes[oidx])
                for oidx in range(len(output_keys)):
                    buffer[oidx][buffer_idx:(buffer_idx+input.shape[0])] = cur_pred
                buffer_idx+=input.shape[0]
                unsaved_batches +=1
                #current_indices.append(index_array)
                if unsaved_batches==write_every_n_batches or i==len(index_arrays)-1:
                    start_save = time.time()
                    #idx_o = list(np.concatenate(current_indices))
                    print("dest sel: {} -> {}".format(output_idx, output_idx+buffer_idx))
                    for oidx in range(len(output_keys)):
                        of.write_direct(paths[oidx], buffer[oidx], source_sel=np.s_[0:buffer_idx], dest_sel=np.s_[output_idx:(output_idx+buffer_idx)])
                    end_save = time.time()
                    print("#{} batches ({} images) computed in {}s and saved in {}s".format(unsaved_batches, buffer_idx, start_save-start_pred, end_save-start_save))
                    unsaved_batches=0
                    output_idx+=buffer_idx
                    buffer_idx=0
                    #current_indices = []
                    start_pred = time.time()
        if close_outputIO:
            of.close()

        # reset iterators parameters
        self.shuffle = shuffle
        self.batch_index = batch_index
        self.perform_data_augmentation = perform_aug

    def _ensure_dataset(self, output_file, output_shapes, output_keys, ds_i, **create_dataset_options):
        if self.datasetIO is None:
            self._open_datasetIO()
        if self.labels is not None:
            label_path = self.paths[ds_i].replace(self.channel_keywords[0], '/labels')
            if label_path not in output_file:
                output_file.create_dataset(label_path, data=np.asarray(self.labels[ds_i], dtype=np.string_))
        dim_path = self.paths[ds_i].replace(self.channel_keywords[0], '/originalDimensions')
        if dim_path not in output_file and dim_path in self.datasetIO:
            output_file.create_dataset(dim_path, data=self.datasetIO.get_dataset(dim_path))
        for oidx, output_key in enumerate(output_keys):
            ds_path = self.paths[ds_i].replace(self.channel_keywords[0], output_key)
            if ds_path not in output_file:
                output_file.create_dataset(ds_path, shape=(self.ds_array[0][ds_i].shape[0],)+output_shapes[oidx], dtype=self.dtype, **create_dataset_options) #, compression="gzip" # no compression for compatibility with java driver

    def _has_object_at_y_borders(self, mask_channel_idx, ds_idx, im_idx):
        ds = self.ds_array[mask_channel_idx][ds_idx]
        off = self.ds_scaling_center[mask_channel_idx][ds_idx] # supposes there are no other scaling for mask channel
        return np.any(ds[im_idx, [-1,0], :] - off, 1) # np.flip()

    def _get_void_masks(self):
        assert len(self.mask_channels)>0, "cannot compute void mask if no mask channel is defined"
        if self.datasetIO is None: # for concurency issues: file is open lazyly by each worker
            self._open_datasetIO()
        mask_channel = self.void_mask_chan
        index_array = np.copy(self.allowed_indexes)
        index_ds = self._get_ds_idx(index_array)
        void_masks = np.array([not np.any(self._read_image(mask_channel, ds_idx, im_idx)) for i, (ds_idx, im_idx) in enumerate(zip(index_ds, index_array))])
        return void_masks

    def set_allowed_indexes(self, indexes):
        super().set_allowed_indexes(indexes)
        try: # reset void masks
            del self.void_masks
        except AttributeError:
            pass

    def __len__(self):
        if self.void_mask_max_proportion>=0 and not hasattr(self, "void_masks"):
            self._set_index_array() # redefines n
        return super().__len__()

    def _set_index_array(self):
        if self.void_mask_max_proportion>=0: # in case there are too many void masks -> some are randomly removed
            try:
                void_masks = self.void_masks
            except AttributeError:
                self.void_masks = self._get_void_masks()
                void_masks = self.void_masks
            bins = np.bincount(void_masks) #[not void ; void]
            if len(bins)==2:
                prop = bins[1] / (bins[0]+bins[1])
                target_void_count = int( (self.void_mask_max_proportion / (1 - self.void_mask_max_proportion) ) * bins[0] )
                n_rem  = bins[1] - target_void_count
                #print("void mask bins: {}, prop: {}, n_rem: {}".format(bins, prop, n_rem))
                if n_rem>0:
                    idx_void = np.flatnonzero(void_masks)
                    to_rem = np.random.choice(idx_void, n_rem, replace=0)
                    index_a = np.delete(self.allowed_indexes, to_rem)
                    self.n = len(index_a)
                else:
                    index_a = self.allowed_indexes
            else:  # only void or only not void
                #print("void mask bins: ", bins)
                index_a = self.allowed_indexes
        else:
            index_a = self.allowed_indexes
        if self.shuffle:
            self.index_array = np.random.permutation(index_a)
        else:
            self.index_array = np.copy(index_a)

    def evaluate(self, model, metrics, progress_callback=None):
        if len(metrics) != len(self.output_channels)*self.output_multiplicity:
            raise ValueError("metrics should be an array of length equal to output number ({})".format(len(self.output_channels)*self.output_multiplicity))

        shuffle = self.shuffle
        perform_aug = self.perform_data_augmentation
        self.shuffle=False
        self.perform_data_augmentation=False
        self.reset()
        self._set_index_array() # in case shuffle was true.

        metrics = [l if isinstance(l, list) else ([] if l is None else [l]) for l in metrics]
        # count metrics
        count = 0
        for m in metrics:
            count+=len(m)
        values = np.zeros(shape=(len(self.allowed_indexes), count+2))
        i=0
        for step in range(len(self)):
            x, y = self.next()
            y_pred = model.predict(x=x, verbose=0)
            n = y_pred[0].shape[0]
            j=2
            for oi, ms in enumerate(metrics):
                for m in ms:
                    res = m(y[oi], y_pred[oi])
                    n_axis = len(res.shape)
                    if n_axis>1:
                        res = np.mean(res, axis=tuple(range(1,n_axis)))
                    values[i:(i+n), j] = res
                    j=j+1
            i+=n

            if progress_callback is not None:
                progress_callback()

        self.shuffle = shuffle
        self.perform_data_augmentation = perform_aug
        # also return dataset dir , index and labels (if availables)

        idx = np.copy(self.index_array)
        ds_idx = self._get_ds_idx(idx)
        values[:,0] = idx
        values[:,1] = ds_idx
        path = [self.paths[i] for i in ds_idx]
        labels = [self.labels[i][j] for i,j in zip(ds_idx, idx)] if self.labels is not None else None
        indices = [str(int(s[1]))+"-"+s[0].split('-')[1] for s in [l.split("_f") for l in labels]] #caveat: s[0].split('-')[1] is not the parent idx but the parentTrackHead idx, same in most case but ...
        return values, path, labels, indices

# class util methods
def copy_geom_tranform_parameters(aug_param_source, aug_param_dest): # TODO : parametrizable
    aug_param_dest['theta'] = aug_param_source.get('theta', 0)
    aug_param_dest['tx'] = aug_param_source.get('tx', 0)
    aug_param_dest['ty'] = aug_param_source.get('ty', 0)
    aug_param_dest['shear'] = aug_param_source.get('shear', 0)
    aug_param_dest['zx'] = aug_param_source.get('zx', 1)
    aug_param_dest['zy'] = aug_param_source.get('zy', 1)
    aug_param_dest['flip_horizontal'] = aug_param_source.get('flip_horizontal', False)
    aug_param_dest['flip_vertical'] = aug_param_source.get('flip_vertical', 0)
    if 'bacteria_swim' in aug_param_source:
        aug_param_dest['bacteria_swim'] = copy.deepcopy(aug_param_source['bacteria_swim'])
    if 'rotate90' in aug_param_source:
        aug_param_dest['rotate90'] = aug_param_dest['rotate90']
