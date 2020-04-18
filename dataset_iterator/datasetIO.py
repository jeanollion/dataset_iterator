class DatasetIO:
    def __init__(self, path):
        self.path=path

    def close(self):
        raise NotImplementedError

    def get_dataset_paths(self, channel_keyword, group_keyword):
        raise NotImplementedError

    def get_dataset(self, path):
        raise NotImplementedError

    def get_attribute(self, path, attribute_name):
        return None

    def create_dataset(self, path, data, **create_dataset_options):
        raise NotImplementedError

    def write_direct(self, path, data, dest_sel):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    @staticmethod
    def get_parent_path(path):
        raise NotImplementedError

def get_datasetIO(dataset, mode):
    if isinstance(dataset, DatasetIO):
        return dataset
    elif dataset.endswith(".h5") or dataset.endswith(".hdf5"):
        from .h5pyIO import H5pyIO
        return H5pyIO(dataset, mode)
    else:
        raise ValueError("File type not supported (yet)")
