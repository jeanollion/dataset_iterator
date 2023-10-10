import multiprocessing

import numpy as np
from multiprocessing import shared_memory, managers
import queue
from keras.utils import data_utils


# adapted from https://muditb.medium.com/speed-up-your-keras-sequence-pipeline-f5d158359f46
class ShmArray(np.ndarray):
    def __new__(cls, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, shm=None):
        obj = super(ShmArray, cls).__new__(cls, shape, dtype, buffer, offset, strides, order)
        obj.shm = shm
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.shm = getattr(obj, 'shm', None)

def decode_queue_item(i_shapes, i_dtypes, o_shapes, o_dtypes, shm_name):
    existing_shm = shared_memory.SharedMemory(create=False, name=shm_name)
    offset = 0
    inputs = []
    outputs = []
    for shape, dtype in zip(i_shapes, i_dtypes):
        a = ShmArray(shape, dtype=dtype, buffer=existing_shm.buf, offset=offset, shm=existing_shm)
        inputs.append(a)
        offset += a.nbytes
    for shape, dtype in zip(o_shapes, o_dtypes):
        a = ShmArray(shape, dtype=dtype, buffer=existing_shm.buf, offset=offset, shm=existing_shm)
        outputs.append(a)
        offset += a.nbytes
    existing_shm.close()
    existing_shm.unlink()
    del existing_shm
    if len(inputs) == 1:
        inputs = inputs[0]
    elif len(inputs) > 1:
        inputs = tuple(inputs)
    if len(outputs) == 0:
        return inputs, None
    elif len(outputs) == 1:
        outputs = outputs[0]
    elif len(outputs) > 1:
        outputs = tuple(outputs)
    return inputs, outputs

def get_index(uid, idx, epoch, use_shm):
    #sequences, sequence_indices = data_utils._SHARED_SEQUENCES[uid]
    #seq_idx = sequence_indices.get()
    #print(f"getting {idx} from sequence: {seq_idx}")
    #item = sequences[seq_idx][idx]
    #sequence_indices.put(seq_idx)
    w = multiprocessing.current_process()
    w.set_epoch(epoch)
    item = w.sequence[idx]
    if multiple(item):
        inputs, outputs = item
    else:
        inputs = item
        outputs = []
    if not multiple(inputs):
        inputs = [inputs]
    if not multiple(outputs):
        outputs = [outputs]
    size = np.sum([a.nbytes for a in inputs] + [a.nbytes for a in outputs])
    #print( f"calling get_index {idx} by pid {multiprocessing.current_process().ident} inputs {len(inputs)} outputs: {len(outputs)} size={size}")
    if use_shm:
        shm = shared_memory.SharedMemory(create=True, size=size)
        i_shapes = []
        i_dtypes = []
        o_shapes = []
        o_dtypes = []
        offset = 0
        for a in inputs:
            shm_a = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf, offset=offset)
            shm_a[:] = a[:]
            i_shapes.append(a.shape)
            i_dtypes.append(a.dtype)
            offset += a.nbytes
        for a in outputs:
            shm_a = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf, offset=offset)
            shm_a[:] = a[:]
            o_shapes.append(a.shape)
            o_dtypes.append(a.dtype)
            offset += a.nbytes
        result = (i_shapes, i_dtypes, o_shapes, o_dtypes, shm.name)
        shm.close()
        del shm
        return result
    else:
        return inputs, outputs

class SharedMemEnqueuer(data_utils.OrderedEnqueuer): # adapted from tf.keras.utils.data_utils.OrderedEnqueuer
    """Builds an Enqueuer from a Sequence.

    Args:
        sequence: A `tf.keras.utils.data_utils.Sequence` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        shuffle: whether to shuffle the data at the beginning of each epoch
    """

    def __init__(self, sequence_len, sequence_fun, use_shm:bool = True, shuffle=True):
        super().__init__(sequence_fun, True, shuffle)
        self.shuffle = shuffle
        self.use_shm = use_shm
        if use_shm:
            self.shared_mem_manager = managers.SharedMemoryManager()
        self.sequence_len = sequence_len
        context = multiprocessing.get_context()
        self.context = context

        class Worker(context.Process):
            def __init__(self, *args, **kwargs):
                """ Constructor.
                :return void
                """
                super().__init__(*args, **kwargs)
                self.sequence = sequence_fun()
                self.epoch = 0

            def __del__(self):
                try:
                    self.sequence.close()
                finally:
                    super().__del__()

            def set_epoch(self, epoch):
                if epoch != self.epoch:
                    self.sequence.on_epoch_end()
                    self.epoch = epoch
            @classmethod
            def register(cls, ctx):
                """ Ensure this worker is used to process
                the tasks.
                """
                ctx.Process = cls
        Worker.register(context)

    def _get_executor_init(self, workers):
        """Gets the Pool initializer for multiprocessing.

        Args:
            workers: Number of workers.

        Returns:
            Function, a Function to initialize the pool
        """

        def pool_fn():
            pool = self.context.Pool(
                workers, initializer=data_utils.init_pool_generator,
                initargs=(None, None, data_utils.get_worker_id_queue()))
            data_utils._DATA_POOLS.add(pool)
            return pool
        return pool_fn

    def start(self, workers=1, max_queue_size=10):
        if self.use_shm:
            self.shared_mem_manager.start()
        super().start(workers, max_queue_size)

    def stop(self, timeout=None):
        if self.use_shm:
            self.shared_mem_manager.shutdown()
            self.shared_mem_manager.join()
        super().stop(timeout)

    def _send_sequence(self):
        pass
        #data_utils._SHARED_SEQUENCES[self.uid] = (self.sequences, self.sequence_indices)

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        sequence = list(range(self.sequence_len))
        #self._send_sequence()
        with data_utils.closing(self.executor_fn()) as executor:
            epoch = 0
            while True:
                if self.shuffle:
                    np.random.shuffle(sequence)
                for i in sequence:
                    if self.stop_signal.is_set():
                        return

                    self.queue.put(
                        executor.apply_async(get_index, (self.uid, i, epoch, self.use_shm)),
                        block=True,
                    )
                if self.stop_signal.is_set():
                    return
                epoch += 1

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
                item = self.queue.get(block=True, timeout=5).get()
                if self.is_running():
                    self.queue.task_done()
                if item is not None:
                    if self.use_shm:
                        inputs, outputs = decode_queue_item(*item)
                    else:
                        inputs, outputs = item
                        if len(outputs) == 0:
                            outputs = None
                    if outputs is None:
                        yield inputs
                    else:
                        yield inputs, outputs
            except queue.Empty:
                pass
            except Exception as e:
                self.stop()
                raise e

def multiple(item):
    return isinstance(item, (list, tuple))