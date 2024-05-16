from .datasetIO import DatasetIO
import threading
import numpy as np
from ..shared_memory import to_shm, get_idx_from_shm, unlink_shm_ref


class MemoryIO(DatasetIO):
    def __init__(self, datasetIO: DatasetIO, use_shm: bool = True):
        super().__init__()
        self.datasetIO = datasetIO
        self.__lock__ = threading.Lock()
        self.datasets = dict()
        self.use_shm = use_shm

    def close(self):
        self.datasets.clear()  # if shm : del will unlink
        self.datasetIO.close()

    def __del__(self):
        self.datasets.clear() # if shm : del will unlink

    def get_dataset_paths(self, channel_keyword, group_keyword):
        return self.datasetIO.get_dataset_paths(channel_keyword, group_keyword)

    def get_dataset(self, path):
        if path not in self.datasets:
            with self.__lock__:
                if path not in self.datasets:
                    if self.use_shm:
                        self.datasets[path] = ShmArrayWrapper(*_to_shm(self.datasetIO.get_dataset(path)[:]))
                    else:
                        self.datasets[path] = ArrayWrapper(self.datasetIO.get_dataset(path)[:]) # load into memory
        return self.datasets[path]

    def get_attribute(self, path, attribute_name):
        return self.datasetIO.get_attribute(path, attribute_name)

    def create_dataset(self, path, **create_dataset_kwargs):
        self.datasetIO.create_dataset(path, **create_dataset_kwargs)

    def write_direct(self, path, data, source_sel, dest_sel):
        self.datasetIO.write_direct(path, data, source_sel, dest_sel)

    def __contains__(self, key):
        self.datasetIO.__contains__(key)

    def get_parent_path(self, path):
        self.datasetIO.get_parent_path(path)


def _to_shm(array):
    shapes, dtypes, shm_name, _ = to_shm(array)
    return shapes[0], dtypes[0], shm_name


class ArrayWrapper:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def __getitem__(self, item):
        return np.copy(self.array[item])

    def __len__(self):
        return self.shape[0]


class ShmArrayWrapper:
    def __init__(self, shape, dtype, shm_name):
        self.shape = shape
        self.dtype = dtype
        self.shm_name = shm_name

    def __getitem__(self, item):
        assert isinstance(item, (int, np.integer)), f"only integer index supported: recieved: {item} of type: {type(item)}"
        return get_idx_from_shm(item, (self.shape,), (self.dtype,), self.shm_name, array_idx=0)

    def __len__(self):
        return self.shape[0]

    def __del__(self):
        self.unlink()

    def unlink(self):
        unlink_shm_ref(self.shm_name)
