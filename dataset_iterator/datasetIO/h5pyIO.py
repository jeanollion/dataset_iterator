import h5py
from .atomic_file_handler import AtomicFileHandler
from .datasetIO import DatasetIO
import threading
import re

class H5pyIO(DatasetIO):
    def __init__(self, h5py_file_path, mode, atomic=False):
        super().__init__()
        self.path = h5py_file_path
        self.mode = mode
        self.atomic = atomic
        self.__lock__ = threading.Lock()
        self.h5py_file=None

    def _get_file(self):
        if self.h5py_file is None:
            with self.__lock__:
                if self.h5py_file is None:
                    file = AtomicFileHandler(self.path) if self.atomic else self.path # this does work with version 1.14 and 2.9 of h5py but not with version 2.8
                    self.h5py_file = h5py.File(file, self.mode)
        return self.h5py_file

    def close(self):
        if self.h5py_file is not None:
            self.h5py_file.close()
            self.h5py_file = None

    def get_dataset_paths(self, channel_keyword, group_keyword):
        return get_dataset_paths(self._get_file(), channel_keyword, group_keyword)

    def get_dataset(self, path):
        return self._get_file()[path]

    def get_attribute(self, path, attribute_name):
        return self._get_file()[path].attrs.get(attribute_name)

    def create_dataset(self, path, **create_dataset_kwargs):
        self._get_file().create_dataset(path, **create_dataset_kwargs)

    def __contains__(self, key):
        return key in self._get_file()

    def write_direct(self, path, data, source_sel, dest_sel):
        self._get_file()[path].write_direct(data, source_sel, dest_sel)

    def get_parent_path(self, path):
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
    if group_keyword is not None and '/.+/' in group_keyword: # common pattern
        group_keyword = re.compile(group_keyword)
    return [path for (path, ds) in h5py_dataset_iterator(h5py_file) if path.endswith(suffix) and _contains_group(path, group_keyword)]

def get_datasets(h5py_file, suffix, group_keyword=None):
    if group_keyword is not None and '/.+/' in group_keyword: # common pattern
        group_keyword = re.compile(group_keyword)
    return [ds for (path, ds) in h5py_dataset_iterator(h5py_file) if path.endswith(suffix) and _contains_group(path, group_keyword)]

def _contains_group(path, group_keyword):
    if group_keyword is None:
        return True
    elif hasattr(group_keyword, "search"):
        return group_keyword.search(path) is not None
    else:
        return group_keyword in path
