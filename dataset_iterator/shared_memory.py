import numpy as np
from multiprocessing import shared_memory


def to_shm(tensors, shm_manager=None):
    flatten_tensor_list, nested_structure = get_flatten_list(tensors)
    size = np.sum([a.nbytes for a in flatten_tensor_list])
    shm = shm_manager.SharedMemory(size=size) if shm_manager is not None else shared_memory.SharedMemory(create=True, size=size)
    shapes = []
    dtypes = []
    offset = 0
    for a in flatten_tensor_list:
        shm_a = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf, offset=offset)
        shm_a[:] = a[:]
        shapes.append(a.shape)
        dtypes.append(a.dtype)
        offset += a.nbytes
    tensor_ref = (shapes, dtypes, shm.name, nested_structure)
    shm.close()
    del shm
    return tensor_ref


def from_shm(shapes, dtypes, shm_name, nested_structure):
    existing_shm = ErasingSharedMemory(shm_name)
    offset = 0
    tensor_list = []
    for shape, dtype in zip(shapes, dtypes):
        a = ShmArray(shape, dtype=dtype, buffer=existing_shm.buf, offset=offset, shm=existing_shm)
        tensor_list.append(a)
        offset += a.nbytes
    return get_nested_structure(tensor_list, nested_structure)


def get_idx_from_shm(idx, shapes, dtypes, shm_name, array_idx=0):
    existing_shm = shared_memory.SharedMemory(shm_name)
    offset = 0
    for i in range(0, array_idx):
        offset += np.prod(shapes[i]) * dtypes[i].itemsize
    shape = shapes[array_idx][1:] if len(shapes[array_idx]) > 1 else (1,)
    offset += idx * np.prod(shape) * dtypes[array_idx].itemsize
    array = np.copy(np.ndarray(shape, dtype=dtypes[array_idx], buffer=existing_shm.buf, offset=offset))
    existing_shm.close()
    del existing_shm
    return array


def multiple(item):
    return isinstance(item, (list, tuple))


def get_flatten_list(item):
    flatten_list = []
    nested_structure = []
    _flatten(item, 0, flatten_list, nested_structure)
    return flatten_list, nested_structure[0]


def _flatten(item, offset, flatten_list, nested_structure):
    if multiple(item):
        nested_structure.append([])
        for sub_item in item:
            offset = _flatten(sub_item, offset, flatten_list, nested_structure[-1])
        return offset
    else:
        nested_structure.append(offset)
        flatten_list.append(item)
        return offset + 1


def get_nested_structure(flatten_list, nested_structure):
    if multiple(nested_structure):
        result = []
        _get_nested(flatten_list, nested_structure, 0, result)
        return result[0]
    else:
        return flatten_list[0]


def _get_nested(flatten_list, nested_structure, offset, result):
    if multiple(nested_structure):
        result.append([])
        for sub_nested in nested_structure:
            offset = _get_nested(flatten_list, sub_nested, offset, result[-1])
        return offset
    else:
        result.append(flatten_list[offset])
        return offset + 1


def unlink_tensor_ref(shapes, dtypes, shm_name, nested_structure):
    unlink_shm_ref(shm_name)


def unlink_shm_ref(shm_name):
    try:
        existing_shm = shared_memory.SharedMemory(shm_name)
        existing_shm.unlink()
    except (FileExistsError, FileNotFoundError):
        pass


# code from: https://muditb.medium.com/speed-up-your-keras-sequence-pipeline-f5d158359f46
class ShmArray(np.ndarray):
    def __new__(cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, shm=None):
        obj = super(ShmArray, cls).__new__(cls, shape, dtype, buffer, offset, strides,  order)
        obj.shm = shm
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.shm = getattr(obj, 'shm', None)


class ErasingSharedMemory(shared_memory.SharedMemory):
    def __del__(self):
        super(ErasingSharedMemory, self).__del__()
        try:
            self.unlink()
        except (FileExistsError, FileNotFoundError):  # manager can delete the file before array is finalized
            pass
