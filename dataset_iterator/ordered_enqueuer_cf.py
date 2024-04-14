import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import queue
import random
import numpy as np
import threading
import time
from multiprocessing import Queue, managers, shared_memory

# Global variables to be shared across processes
_SHARED_SEQUENCES = {}
_SHARED_SHM_MANAGER = {}
# We use a Value to provide unique id to different processes.
_SEQUENCE_COUNTER = None
# adapted from https://github.com/keras-team/keras/blob/v2.13.1/keras/utils/data_utils.py#L651-L776
# uses cncurrent.futures  + shared memory + overcome a memory leak in case of hard sample mining run as callback
class OrderedEnqueuerCF():
    def __init__(self, sequence, shuffle=False, single_epoch:bool=False, use_shm:bool=True, wait_for_me=None):
        self.sequence = sequence
        self.shuffle = shuffle
        self.single_epoch=single_epoch
        self.use_shm=use_shm
        self.wait_for_me = wait_for_me
        global _SEQUENCE_COUNTER
        if _SEQUENCE_COUNTER is None:
            try:
                _SEQUENCE_COUNTER = multiprocessing.Value("i", 0)
            except OSError:
                # In this case the OS does not allow us to use
                # multiprocessing. We resort to an int
                # for enqueuer indexing.
                _SEQUENCE_COUNTER = 0

        if isinstance(_SEQUENCE_COUNTER, int):
            self.uid = _SEQUENCE_COUNTER
            _SEQUENCE_COUNTER += 1
        else:
            # Doing Multiprocessing.Value += x is not process-safe.
            with _SEQUENCE_COUNTER.get_lock():
                self.uid = _SEQUENCE_COUNTER.value
                _SEQUENCE_COUNTER.value += 1

        self.workers = 0
        self.queue = None
        self.run_thread = None
        self.stop_signal = None
        self.shm_manager = None

    def is_running(self):
        return self.stop_signal is not None and not self.stop_signal.is_set()

    def start(self, workers=1, max_queue_size=10):
        """Starts the handler's workers.

        Args:
            workers: Number of workers.
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        """
        self.workers = workers
        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        if self.use_shm:
            self.shm_manager = managers.SharedMemoryManager()
            self.shm_manager.start()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _wait_queue(self):
        """Wait for the queue to be empty."""
        while True:
            time.sleep(0.1)
            if self.queue.unfinished_tasks == 0 or self.stop_signal.is_set():
                return

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""

        sequence = list(range(len(self.sequence)))
        self._send_sequence()  # Share the initial sequence
        while True:
            if self.shuffle:
                random.shuffle(sequence)
            task = get_item_shm if self.use_shm else get_item
            with ProcessPoolExecutor(max_workers=self.workers, initializer=init_pool_generator, initargs=(self.sequence, self.uid, self.shm_manager)) as executor:
                for i in sequence:
                    if self.stop_signal.is_set():
                        return
                    self.queue.put(executor.submit(task, self.uid, i), block=True)
                # Done with the current epoch, waiting for the final batches
                self._wait_queue()

            if self.stop_signal.is_set() or self.single_epoch:
                # We're done
                return
            if self.wait_for_me is not None:
                self.wait_for_me.wait()
            # Call the internal on epoch end.
            self.sequence.on_epoch_end()
            self._send_sequence()  # Update the pool

    def _send_sequence(self):
        """Sends current Iterable to all workers."""
        # For new processes that may spawn
        global _SHARED_SEQUENCES
        _SHARED_SEQUENCES[self.uid] = self.sequence
        global _SHARED_SHM_MANAGER
        _SHARED_SHM_MANAGER[self.uid] = self.shm_manager

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        Yields:
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        while self.is_running():
            try:
                inputs = self.queue.get(block=True, timeout=5).result()
                if self.is_running():
                    self.queue.task_done()
                if inputs is not None:
                    if self.use_shm:
                        inputs = from_shm(*inputs)
                    yield inputs
            except queue.Empty:
                pass
            except Exception as e:
                self.stop()
                raise e

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        Args:
            timeout: maximum time to wait on `thread.join()`
        """
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.run_thread.join(timeout)
        if self.use_shm is not None:
            self.shm_manager.shutdown()
            self.shm_manager.join()
        global _SHARED_SHM_MANAGER
        _SHARED_SHM_MANAGER[self.uid] = None
        global _SHARED_SEQUENCES
        _SHARED_SEQUENCES[self.uid] = None


def init_pool_generator(gen, uid, shm_manager):
    global _SHARED_SEQUENCES
    _SHARED_SEQUENCES[uid] = gen
    global _SHARED_SHM_MANAGER
    _SHARED_SHM_MANAGER[uid] = shm_manager


def get_item_shm(uid, i):
    tensors = _SHARED_SEQUENCES[uid][i]
    return to_shm(_SHARED_SHM_MANAGER[uid], tensors)


def get_item(uid, i):
    return _SHARED_SEQUENCES[uid][i]


def to_shm(shm_manager, tensors):
    flatten_tensor_list, nested_structure = get_flatten_list(tensors)
    size = np.sum([a.nbytes for a in flatten_tensor_list])
    shm = shm_manager.SharedMemory(size=size)
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


# code from: https://muditb.medium.com/speed-up-your-keras-sequence-pipeline-f5d158359f46
class ShmArray(np.ndarray):
    def __new__(cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, shm=None):
        obj = super(ShmArray, cls).__new__(cls, shape, dtype, buffer, offset, strides,  order)
        obj.shm = shm
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.shm = getattr(obj, 'shm', None)

class ErasingSharedMemory(shared_memory.SharedMemory):
    def __del__(self):
        super(ErasingSharedMemory, self).__del__()
        self.unlink()