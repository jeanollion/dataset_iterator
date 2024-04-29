from .datasetIO import DatasetIO
import threading
import numpy as np
from multiprocessing import managers, shared_memory, Value
from ..shared_memory import to_shm, get_idx_from_shm

_MEMORYIO_SHM_MANAGER = {}
_MEMORYIO_UID = None

class MemoryIO(DatasetIO):
    def __init__(self, datasetIO: DatasetIO, use_shm: bool = True):
        super().__init__()
        self.datasetIO = datasetIO
        self.__lock__ = threading.Lock()
        self.datasets = dict()
        self.use_shm = use_shm
        global _MEMORYIO_UID
        if _MEMORYIO_UID is None:
            try:
                _MEMORYIO_UID = Value("i", 0)
            except OSError: # In this case the OS does not allow us to use multiprocessing. We resort to an int for indexing.
                _MEMORYIO_UID = 0

        if isinstance(_MEMORYIO_UID, int):
            self.uid = _MEMORYIO_UID
            _MEMORYIO_UID += 1
        else:
            # Doing Multiprocessing.Value += x is not process-safe.
            with _MEMORYIO_UID.get_lock():
                self.uid = _MEMORYIO_UID.value
                _MEMORYIO_UID.value += 1
        if use_shm:
            self._start_shm_manager()

    def _start_shm_manager(self):
        global _MEMORYIO_SHM_MANAGER
        _MEMORYIO_SHM_MANAGER[self.uid] = managers.SharedMemoryManager()
        _MEMORYIO_SHM_MANAGER[self.uid].start()
        self.shm_manager_on = True

    def _stop_shm_manager(self):
        global _MEMORYIO_SHM_MANAGER
        if _MEMORYIO_SHM_MANAGER[self.uid] is not None:
            _MEMORYIO_SHM_MANAGER[self.uid].shutdown()
            _MEMORYIO_SHM_MANAGER[self.uid].join()
            _MEMORYIO_SHM_MANAGER[self.uid] = None
        self.shm_manager_on = False

    def _to_shm(self, array):
        global _MEMORYIO_SHM_MANAGER
        shapes, dtypes, shm_name, _ = to_shm(_MEMORYIO_SHM_MANAGER[self.uid], array)
        return shapes[0], dtypes[0], shm_name

    def close(self):
        if self.use_shm:
            for shma in self.datasets.values():
                shma.unlink()
        self.datasets.clear()
        self.datasetIO.close()
        if self.use_shm and self.shm_manager_on:
            self._stop_shm_manager()

    def get_dataset_paths(self, channel_keyword, group_keyword):
        return self.datasetIO.get_dataset_paths(channel_keyword, group_keyword)

    def get_dataset(self, path):
        if path not in self.datasets:
            with self.__lock__:
                if self.use_shm and not self.shm_manager_on:
                    self._start_shm_manager()
                if path not in self.datasets:
                    if self.use_shm:
                        self.datasets[path] = ShmArrayWrapper(*self._to_shm(self.datasetIO.get_dataset(path)[:]))
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

    def unlink(self):
        try:
            existing_shm = shared_memory.SharedMemory(self.shm_name)
            existing_shm.unlink()
        except Exception:
            pass