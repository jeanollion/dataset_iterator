import h5py
from .atomic_file_handler import AtomicFileHandler
from .datasetIO import DatasetIO

class H5pyIO(DatasetIO):
    def __init__(self, h5py_file, mode):
        super().__init__(h5py_file)
        self.h5py_file = h5py.File(AtomicFileHandler(h5py_file), mode) # this does work with version 1.14 and 2.9 of h5py but not with version 2.8

    def close(self):
        self.h5py_file.close()

    def get_dataset_paths(self, channel_keyword, group_keyword):
        return get_dataset_paths(self.h5py_file, channel_keyword, group_keyword)

    def get_dataset(self, path):
        return self.h5py_file[path]

    def get_attribute(self, path, attribute_name):
        return self.h5py_file[path].attrs.get(attribute_name)

    def create_dataset(self, path, data, **create_dataset_options):
        self.h5py_file.create_dataset(path, data, **create_dataset_options)

    def __contains__(self, key):
        return key in self.h5py_file

    def write_direct(self, path, data, dest_sel):
        self.h5py_file[path].write_direct(data, dest_sel)

    @staticmethod
    def get_parent_path(path):
        idx = path.rfind('/')
        if idx>0:
            return path[:idx]
        else:
            return None

def h5py_dataset_iterator(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group): # test for group (go down)
            yield from h5py_dataset_iterator(item, path)

def get_dataset_paths(h5py_file, suffix, group_keyword=None):
    return [path for (path, ds) in h5py_dataset_iterator(h5py_file) if path.endswith(suffix) and (group_keyword==None or group_keyword in path)]

def get_datasets(h5py_file, suffix, group_keyword=None):
    return [ds for (path, ds) in h5py_dataset_iterator(h5py_file) if path.endswith(suffix) and (group_keyword==None or group_keyword in path)]
