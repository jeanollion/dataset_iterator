import os

class DatasetIO:

    def close(self):
        raise NotImplementedError

    def get_dataset_paths(self, channel_keyword, group_keyword):
        raise NotImplementedError

    def get_dataset(self, path):
        raise NotImplementedError

    def get_attribute(self, path, attribute_name):
        return None

    def create_dataset(self, path, **create_dataset_kwargs):
        raise NotImplementedError

    def write_direct(self, path, data, source_sel, dest_sel):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def get_parent_path(self, path):
        raise NotImplementedError

def get_datasetIO(dataset, mode='r'):
    #print("type: {}, subclass: {}".format(type(dataset), issubclass(type(dataset), DatasetIO)))
    if issubclass(type(dataset), DatasetIO):
        return dataset
    elif isinstance(dataset, str):
        if dataset.endswith(".h5") or dataset.endswith(".hdf5"):
            from .h5pyIO import H5pyIO
            return H5pyIO(dataset, mode)
        elif os.path.isdir(dataset):
            from .multiple_fileIO import MultipleFileIO
            return MultipleFileIO(dataset, 1) # consider that directory contains files, each file corresponds to a single image with same dimensions
    elif isinstance(dataset, (tuple, list)):
        from .concatenate_datasetIO import ConcatenateDatasetIO
        return ConcatenateDatasetIO(dataset) if len(dataset)>1 else get_datasetIO(dataset[0])
    raise ValueError("File type not supported (yet)")
