import warnings
import numpy as np
from dataset_iterator import IndexArrayIterator
from .utils import remove_duplicates, pick_from_array
from sklearn.model_selection import train_test_split
import time
import copy
from math import copysign, ceil
from .datasetIO import DatasetIO, get_datasetIO, MemoryIO
from .utils import ensure_multiplicity, flatten_list
from itertools import groupby
from .index_array_iterator import IMCOMPLETE_LAST_BATCH_MODE

class MultiChannelIterator(IndexArrayIterator):
    """Flexible Image iterator allowing on-the-fly pre-processing / augmentation of data of massive multichannel datasets.
    Each element returned by the iterator is a tuple (input, output), where input and output represent one or several tensors.
    Each tensor's first axis is the batch, and last axis is the channel.
    Dataset is accessed through the class DatasetIO, that includes .h5 files and PILLOW-compatible images.
    Datasets can contain several sub-datasets, that will be grouped according to their path (see channel_keywords and group_keyword descriptions)

    Parameters
    ----------
    dataset : DatasetIO or string
        if string : will be processed by the method DatasetIO.get_datasetIO. Currently it should be a .h5 file.
    channel_keywords : list of strings
        keywords defining each channel.
        The path of each channel file within dataset should only differ by this keyword
        This list defines the indices of all channels: e.g if channel_keywords = ["/raw", "/labels"] : all datasets containing "/raw" in their path will be associated to the channel of index 0 and all datasets which path contains "/labels" instead of "/raw" will be associated to the channel of index 1.
    input_channels : list of ints
        index of returned input channels within channel_keywords
    output_channels : list of ints
        index of returned output channels within channel_keywords
    weight_map_functions : list of callable
        should be None or of same length as output_channels.
        For each output channel, if not None, the corresponding function is applied to the output batch of same index, and concatenated along the last axis
        Applied before output_postprocessing_functions if any and after channels_postprocessing_function.
    output_postprocessing_functions : list of callable
        should be None or of same length as output_channels.
        For each output channel, if not None, the corresponding function is applied to the output batch of same index.
        Applied after channels_postprocessing_function.
    channels_postprocessing_function : callable
        input is a dict mapping channel index to batch
        applied after image_data_generators and before weight_map_functions / output_postprocessing_functions if any.
    extract_tile_function : callable
        function that inputs a batch, splits it into tiles and concatenate along batch axis.
        Applied before channels_postprocessing_function and after image_data_generators
    mask_channels : list of ints
        index of channels that contain binary (or labeled) images. Used to detect the presence/abscence of segmented objects at image borders
    output_multiplicity : int
        output = (output) * output_multiplicity
    input_multiplicity : int
        input = (input) * input_multiplicity
    channel_scaling_param : list of dict
        channel scaling parameter: each element should be as: {'level':1, 'qmin':5, 'qmax':95}, where:
        qmin is the min percentile, qmax the max percentile, Level represent the hierachical level at which computing the percentile.
        scaling is: I -> ( I - qmin ) / ( qmax - qmin )
    group_keyword : string or list of strings
        string contained in the path of all channels -> allows to limit iteration to a subset of the dataset.
    image_data_generators : list of image ImageDataGenerator as defined in keras pre-processing, for data augmentation.
        should be of same size as channel_keywords.
        augmentation parameters are computed on the first channel, and applied on each channels using the ImageDataGenerator of corresponding index.
        if a mask_channels is not empty, then the first mask channel is used as reference for parameters computation instead of the first channel.
    singleton_channels : list of ints
        list of channel that contains a single image.
    n_spatial_dims : int (2 or 3)
        number of spatial dimensions. if 2, 4D tensor are returned, if 3 5D tensor are returned.
    batch_size : int
        size of the batch (first axis of returned tensors)
    shuffle : boolean
        if true, image indices are shuffled
    perform_data_augmentation : boolean
        if false, image_data_generators are ignored
    seed : int
        random seed
    dtype : numpy datatype

    Attributes
    ----------
    paths : list of string
        list of paths of all the reference channel sub-datasets
    group_map_paths : dict
        dict mapping each group_keyword to a list of paths
    ds_array : list of list of sub-dataset
        first axis is the channel, second the index of the sub-dataset (same length as paths)
    ds_len : list of ints
        cumulative index of the last element of each sub-dataset
    ds_off : list of ints
        cumulative index of the first element of each sub-dataset
    grp_len : list of ints
        cumulative index of the last element of each sub-dataset group
    grp_off : list of ints
        cumulative index of the first element of each sub-dataset group
    channel_image_shapes : list of tuple
        tensor shape of each sub-dataset without batch axis.
    labels : list of string
        list of label vector corresponding to each sub-dataset. initialized if the dataset contains sub-datasets corresponding to the channel keyword "/labels"
        each element of labels is a vector of length equal to the batch axis of the corresponding sub-datasets
        each label is in the format: <barcode>_fXXXXX where <barcode> is a unique identifier of a time-series withing the sub-dataset, and XXXXX an int that correspond to the time step of the imageself.
        Labels are used to return the index of the previous/next time-step in the TrackingIterator. In that case sub-datasets should correpond one or several time-series (a time-series correspond to successive images)
    void_mask_max_proportion : float
        maximum proportion of void images in each batch
        an image is considered as void when the corresponding image in the channel defined in void_mask_chan contains only zeros.
    void_mask_chan : int
        index of the mask channel used to determine whether and image is void or not.
        see: void_mask_max_proportion
    group_proportion : list of floats
        should be of same length as group_keyword
        proportion of image of each group in each batch
    incomplete_last_batch_mode : one of ["KEEP", "CONSTANT_SIZE", "REMOVE"]
        behavior for last batch in case number of element is not a multiple of batch_size
        "KEEP" : last batch will be smaller than other batches
        "CONSTANT_SIZE" : last batch will have same size as other batches, some of his elements will overlap with previous batch
        "REMOVE" : last batch is simply removed
    dataset
    n_spatial_dims
    group_keyword
    channel_keywords
    channel_scaling_param
    dtype
    perform_data_augmentation
    singleton_channels
    mask_channels
    output_multiplicity
    input_multiplicity
    input_channels
    output_channels
    image_data_generators
    weight_map_functions
    output_postprocessing_functions
    channels_postprocessing_function
    extract_tile_function

    """
    def __init__(self,
                dataset,
                channel_keywords=['/raw'],
                input_channels=[0],
                output_channels=[0],
                weight_map_functions=None,
                output_postprocessing_functions=None,
                channels_postprocessing_function=None,
                extract_tile_function=None,
                mask_channels=[],
                output_multiplicity = 1,
                input_multiplicity = 1,
                channel_scaling_param=None,
                group_keyword=None,
                group_proportion=None,
                image_data_generators=None,
                singleton_channels=[],
                n_spatial_dims=2,
                batch_size=32,
                shuffle=True,
                perform_data_augmentation=True,
                seed=None,
                dtype='float32',
                memory_persistant=False,
                incomplete_last_batch_mode=IMCOMPLETE_LAST_BATCH_MODE[0]):
        self.dataset = dataset
        self.memory_persistant=memory_persistant
        self.n_spatial_dims=n_spatial_dims
        self.group_keyword=group_keyword
        self.group_proportion=group_proportion
        if group_proportion is not None:
            assert group_keyword is not None and isinstance(group_keyword, (tuple, list)) and isinstance(group_proportion, (tuple, list)) and len(group_proportion)==len(group_keyword), "when group_proportion is not None, group_keyword should be a list/tuple group_proportion should be of same length as group_keyword"
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
        self.input_multiplicity=input_multiplicity
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
        self.channels_postprocessing_function=channels_postprocessing_function
        self.extract_tile_function=extract_tile_function
        self.paths=None
        self.group_map_paths=None
        self._open_datasetIO()
        # check that all ds have compatible length between input and output
        indexes = np.array([ds.shape[0] for ds in self.ds_array[0]])
        if len(channel_keywords)>1:
            for c, ds_l in enumerate(self.ds_array):
                if self.channel_keywords[c] is not None:
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
        # get offset for each group of dataset
        self.grp_len = []
        off = 0
        for paths in self.group_map_paths.values():
            off += len(paths)
            self.grp_len.append(self.ds_len[off-1])
        self.grp_off=np.insert(self.grp_len[:-1], 0, 0)
        # check that all datasets have same image shape within each channel. rank should be n_spatial_dims (if n channel = 1 or spatial dims + 1)
        self.channel_image_shapes = [ds_l[0].shape[1:] if ds_l is not None else None for ds_l in self.ds_array]
        assert np.all(len(s) == self.n_spatial_dims or len(s) == self.n_spatial_dims+1 for s in self.channel_image_shapes if s is not None), "invalid image rank, current spatial dims number is {}, image rank should be in [{}, {}]".format(self.n_spatial_dims, self.n_spatial_dims, self.n_spatial_dims+1)
        # check that all dataset have same image shape
        for c, ds_l in enumerate(self.ds_array):
            if self.channel_keywords[c] is not None:
                for ds_idx, ds in enumerate(ds_l):
                    if ds.shape[1:] != self.channel_image_shapes[c]:
                        warnings.warn('Dataset {dsi} with path {dspath} from channel {chan}({chank}) has shape {dsshape} that differs from first dataset with path {ds1path} with shape {ds1shape}. Batch size is set to 1'.format(dsi=ds_idx, dspath=self._get_dataset_path(c, ds_idx), chan=c, chank=self.channel_keywords[c], dsshape=ds.shape[1:], ds1path=self._get_dataset_path(c, 0), ds1shape=self.channel_image_shapes[c] ))
                        self.batch_size=1
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
                if scaling_info is not None:
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
        if not memory_persistant:
            self._close_datasetIO()
        self.void_mask_max_proportion = -1
        if len(mask_channels)>0:
            self.void_mask_chan = mask_channels[0]
        else:
            self.void_mask_chan=-1
        super().__init__(indexes[-1], batch_size, shuffle, seed, incomplete_last_batch_mode)

    def _open_datasetIO(self):
        self.datasetIO = get_datasetIO(self.dataset, 'r')
        if self.memory_persistant:
            self.datasetIO = MemoryIO(self.datasetIO)
        if self.paths is None:
            self.group_map_paths = dict()
            group_list = self.group_keyword if isinstance(self.group_keyword, (list, tuple)) else [self.group_keyword]
            for k in group_list:
                # get all dataset paths
                paths = self.datasetIO.get_dataset_paths(self.channel_keywords[0], k)
                if (len(paths)==0):
                    raise ValueError('No datasets found ending by {} {}'.format(self.channel_keywords[0], "and containing {}".format(k) if k is not None else "" ))
                self.group_map_paths[k] = paths
            self.paths = flatten_list(self.group_map_paths.values())
        # get all matching dataset
        self.ds_array = [ [self.datasetIO.get_dataset(self._get_dataset_path(c, ds_idx)) for ds_idx in range(len(self.paths))] if self.channel_keywords[c] is not None else None for c in range(len(self.channel_keywords))]
        getAttribute = lambda a, def_val : def_val if a is None else (a[0] if isinstance(a, list) else a)
        self.ds_scaling_center = [[getAttribute(self.datasetIO.get_attribute(self._get_dataset_path(c, ds_idx), "scaling_center"), 0) for ds_idx in range(len(self.paths))]  if self.channel_keywords[c] is not None else None for c in range(len(self.channel_keywords))]
        self.ds_scaling_factor = [[getAttribute(self.datasetIO.get_attribute(self._get_dataset_path(c, ds_idx), "scaling_factor"), 1) for ds_idx in range(len(self.paths))]  if self.channel_keywords[c] is not None else None for c in range(len(self.channel_keywords))]

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

    def _get_grp_idx(self, index_array):
        grp_idx = np.searchsorted(self.grp_len, index_array, side='right')
        index_array -= self.grp_off[grp_idx] # remove ds offset to each index
        return grp_idx

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples (tuple of input and output if output_keyword is specified).
        """
        batch_by_channel, aug_param_array, ref_chan_idx = self._get_batch_by_channel(index_array, self.perform_data_augmentation)

        if self.output_channels is None or len(self.output_channels)==0:
            input = self._get_input_batch(batch_by_channel, ref_chan_idx, aug_param_array)
            return _apply_multiplicity(input, self.input_multiplicity) # removes None batches
        else:
            input = self._get_input_batch(batch_by_channel, ref_chan_idx, aug_param_array)
            output = self._get_output_batch(batch_by_channel, ref_chan_idx, aug_param_array)
            if isinstance(self.output_multiplicity, dict):
                output_len = len(output) if isinstance(output, (list, tuple)) else 1
                assert max(self.output_multiplicity.keys())<output_len, "invalid output_multiplicity : keys should be included in output_channels list range"
                for ocidx in self.output_channels:
                    if output[ocidx] is not None:
                        assert ocidx in self.output_multiplicity, "output: #"+ocidx+" is not None and not in output_multiplicity"
            if isinstance(self.input_multiplicity, dict):
                input_len = len(input) if isinstance(input, (list, tuple)) else 1
                assert max(self.input_multiplicity.keys())<input_len, "invalid input_multiplicity : keys should be included in input_channels list range"
                for icidx in self.input_channels:
                    if output[icidx] is not None:
                        assert icidx in self.input_multiplicity, "input: #"+icidx+" is not None and not in input_multiplicity"

            input = _apply_multiplicity(input, self.input_multiplicity) # removes None batches
            output = _apply_multiplicity(output, self.output_multiplicity) # removes None batches
            return (input, output)

    def _get_batch_by_channel(self, index_array, perform_augmentation, input_only=False):
        if self.datasetIO is None: # for concurency issues: file is open lazyly by each worker
            self._open_datasetIO()
        index_array = np.copy(index_array) # so that main index array is not modified
        index_ds = self._get_ds_idx(index_array) # modifies index_array

        batch_by_channel = dict()
        channels = self.input_channels if input_only else [c_idx for c_idx, key in enumerate(self.channel_keywords) if key is not None] #remove_duplicates(self.input_channels+self.output_channels)
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

        if self.channels_postprocessing_function is not None:
            self.channels_postprocessing_function(batch_by_channel)

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
            if batch_by_channel[self.output_channels[0]] is None:
                return
            else:
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
                if aug_param_array is not None and params is not None:
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
        # read all images # TODO read all image per ds at once.
        images = [self._read_image(chan_idx, ds_idx, im_idx) for i, (ds_idx, im_idx) in enumerate(zip(index_ds, index_array))]
        batch = np.stack(images)
        return batch

    def _get_data_augmentation_parameters(self, chan_idx, ref_chan_idx, batch, idx, index_ds, index_array):
        if self.image_data_generators is None or self.image_data_generators[chan_idx] is None:
            return None
        params = self.image_data_generators[chan_idx].get_random_transform(batch.shape[1:])
        if  params is not None and chan_idx==ref_chan_idx and chan_idx in self.mask_channels:
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
        if len(im.shape)==self.n_spatial_dims: # add channel axis
            im = np.expand_dims(im, -1)
        im = im.astype(self.dtype, copy=False) # copy
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
            output_image_shapes = self.channel_image_shapes[0] if len(self.channel_image_shapes[0])==self.n_spatial_dims else self.channel_image_shapes[0][:-1]
        if not isinstance(output_keys, (list, tuple)):
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
        self.reset()
        self._set_index_array() # if shuffle was true

        if np.any(self.index_array[1:] < self.index_array[:-1]):
            raise ValueError('Index array should be monotonically increasing')

        buffer = [np.zeros(shape = (min(len(self) * self.batch_size, write_every_n_batches*self.batch_size),)+output_shapes[oidx], dtype=self.dtype) for oidx,k in enumerate(output_keys)]
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
            start_pred = time.time()
            for i, index_array in enumerate(index_arrays):
                batch_by_channel, aug_param_array, ref_chan_idx = self._get_batch_by_channel(index_array, perform_augmentation=self.perform_data_augmentation, input_only=True)
                input = self._get_input_batch(batch_by_channel, ref_chan_idx, aug_param_array)
                cur_pred = pred_fun(model, input)
                if apply_to_prediction is not None:
                    cur_pred = apply_to_prediction(cur_pred)
                if not isinstance(cur_pred, (list, tuple)):
                    cur_pred = [cur_pred]
                assert len(cur_pred)==len(output_keys), 'prediction should have as many output as output_keys argument. # output keys: {} # predictions: {}'.format(len(output_keys), len(cur_pred))
                for oidx in range(len(output_keys)):
                    assert cur_pred[oidx].shape[1:] == output_shapes[oidx], "prediction shape differs from output shape for output idx={} : prediction: {} target: {}".format(oidx, cur_pred[oidx].shape[1:], output_shapes[oidx])
                #print("predicted: {}->{}".format(output_idx, cur_pred[0].shape[0]))
                for oidx in range(len(output_keys)):
                    buffer[oidx][buffer_idx:(buffer_idx+cur_pred[oidx].shape[0])] = cur_pred[oidx]
                buffer_idx+=cur_pred[0].shape[0] # assumes all outputs have same batch size
                unsaved_batches +=1
                if unsaved_batches==write_every_n_batches or i==len(index_arrays)-1:
                    start_save = time.time()
                    #print("dest sel: {} -> {}".format(output_idx, output_idx+buffer_idx))
                    for oidx in range(len(output_keys)):
                        of.write_direct(paths[oidx], buffer[oidx], source_sel=np.s_[0:buffer_idx], dest_sel=np.s_[output_idx:(output_idx+buffer_idx)])
                    end_save = time.time()
                    print("#{} batches ({} images) computed in {}s and saved in {}s".format(unsaved_batches, buffer_idx, start_save-start_pred, end_save-start_save))
                    unsaved_batches=0
                    output_idx+=buffer_idx
                    buffer_idx=0
                    start_pred = time.time()
        if close_outputIO:
            of.close()

        # reset iterators parameters
        self.shuffle = shuffle
        self.batch_index = batch_index

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
        for ocidx, output_key in enumerate(output_keys):
            ds_path = self.paths[ds_i].replace(self.channel_keywords[0], output_key)
            if ds_path not in output_file:
                output_file.create_dataset(ds_path, shape=(self.ds_array[0][ds_i].shape[0],)+output_shapes[ocidx], dtype=self.dtype, **create_dataset_options) #, compression="gzip" # no compression for compatibility with java driver

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
        if self.void_mask_max_proportion>=0 and not hasattr(self, "void_masks") or self.void_mask_max_proportion<0 and self.group_proportion is not None:
            self._set_index_array() # redefines n in either cases
        return super().__len__()

    def _set_index_array(self):
        if self.void_mask_max_proportion>=0: # if void_mask_max_proportion is set. Use case: in case there are too many void masks -> some are randomly removed
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
        elif self.group_proportion is not None: # generate a batch with user-defined proportion in each group
            # pick indices for each group
            if self.allowed_indexes is None:
                indices_per_group = [np.random.randint(low=self.grp_off[i], high=self.grp_len[i], size=int((self.grp_len[i] - self.grp_off[i])*self.group_proportion[i]+0.5) ) for i in range(len(self.group_map_paths))]
                index_a = flatten_list(indices_per_group)
            else:
                index_array = np.copy(self.allowed_indexes) # index within group
                index_grp = self._get_grp_idx(index_array) # group index
                allowed_indexes_per_group = [self.allowed_indexes[index_grp==i] for i in range(len(self.group_map_paths))]
                indexes_per_group = [ pick_from_array(allowed_indexes_per_group[i], self.group_proportion[i]) for i in range(len(self.group_map_paths)) ]
                index_a = np.concatenate(indexes_per_group)
            self.n = len(index_a)
        else:
            index_a = self.allowed_indexes
        if self.shuffle:
            self.index_array = np.random.permutation(index_a)
        else:
            self.index_array = np.copy(index_a)

    def evaluate(self, model, metrics, progress_callback=None):
        output_number = sum(self.output_multiplicity.values()) if isinstance(self.output_multiplicity, dict) else len(self.output_channels)*self.output_multiplicity
        if len(metrics) != output_number:
            raise ValueError("metrics should be an array of length equal to output number ({})".format(output_number))
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
def _apply_multiplicity(batch, multiplicity):
    if batch is None:
        return
    if multiplicity==1:
        return batch
    if isinstance(batch, tuple):
        batch = list(batch)
    elif not isinstance(batch, list):
        batch = [batch]
    if isinstance(multiplicity, dict):
        batch = [ [batch[oidx]]*n for oidx, n in multiplicity.items() if batch[oidx] is not None ]
        return flatten_list(batch)
    elif multiplicity>1:
        batch = [b for b in batch if b is not None]
        return batch * multiplicity

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
