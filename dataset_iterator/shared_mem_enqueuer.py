import multiprocessing
import tensorflow as tf
import numpy as np
from multiprocessing import shared_memory, managers
import queue
from contextlib import closing
from tensorflow.python.keras.utils import data_utils
import random
import SharedArray as sa
import uuid


class ShmSequenceAdapter(object):
    def __init__(self, sequence):
        self.sequence = sequence
    def on_epoch_end(self):
        try:
            self.sequence.on_epoch_end()
        except AttributeError:
            pass

    def __len__(self):
        return self.sequence.__len__()

    def __getitem__(self, item):
        result = self.sequence.__getitem__(item)
        if multiple(result):
            inputs, outputs = result
        else:
            inputs = result
            outputs = []
        if not multiple(inputs):
            inputs = [inputs]
        if not multiple(outputs):
            outputs = [outputs]
        size = np.sum([a.nbytes for a in inputs] + [a.nbytes for a in outputs])
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

    @staticmethod
    def decode(i_shapes, i_dtypes, o_shapes, o_dtypes, shm_name):
        existing_shm = shared_memory.SharedMemory(name=shm_name, create=False)
        offset = 0
        inputs = []
        outputs = []
        for shape, dtype in zip(i_shapes, i_dtypes):
            a = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf, offset=offset)
            inputs.append(a)
            offset += a.nbytes
        for shape, dtype in zip(o_shapes, o_dtypes):
            a = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf, offset=offset)
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
            return inputs
        elif len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) > 1:
            outputs = tuple(outputs)
        return inputs, outputs

class SharedArraySequenceAdapter(object):
    def __init__(self, sequence):
        self.sequence = sequence
    def on_epoch_end(self):
        try:
            self.sequence.on_epoch_end()
        except AttributeError:
            pass

    def __len__(self):
        return self.sequence.__len__()

    def __getitem__(self, item):
        result = self.sequence.__getitem__(item)
        if multiple(result):
            inputs, outputs = result
        else:
            inputs = result
            outputs = []
        if not multiple(inputs):
            inputs = [inputs]
        if not multiple(outputs):
            outputs = [outputs]
        i_names = []
        o_names = []
        for a in inputs:
            name = f"shm://shmnp{uuid.uuid4().hex}"
            shm_a = sa.create(name=name, shape=a.shape, dtype=a.dtype)
            shm_a[:] = a[:]
            i_names.append(name)
        for a in outputs:
            name = f"shm://shmnp{uuid.uuid4().hex}"
            shm_a = sa.create(name=name, shape=a.shape, dtype=a.dtype)
            shm_a[:] = a[:]
            o_names.append(name)
        result = (i_names, o_names)
        return result

    @staticmethod
    def decode(i_names, o_names):
        inputs = []
        outputs = []
        for name in i_names:
            a = sa.attach(name=name)
            sa.delete(name)
            inputs.append(a)
        for name in o_names:
            a = sa.attach(name=name)
            sa.delete(name)
            outputs.append(a)
        if len(inputs) == 1:
            inputs = inputs[0]
        elif len(inputs) > 1:
            inputs = tuple(inputs)
        if len(outputs) == 0:
            return inputs
        elif len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) > 1:
            outputs = tuple(outputs)
        return inputs, outputs

class DummySequenceAdapter(object):
    def __init__(self, sequence):
        self.sequence = sequence
    def on_epoch_end(self):
        try:
            self.sequence.on_epoch_end()
        except AttributeError:
            pass

    def __len__(self):
        return self.sequence.__len__()

    def __getitem__(self, item):
        result = self.sequence.__getitem__(item)
        if multiple(result):
            inputs, outputs = result
        else:
            inputs = result
            outputs = []
        if not multiple(inputs):
            inputs = [inputs]
        if not multiple(outputs):
            outputs = [outputs]
        return inputs, outputs

    @staticmethod
    def decode(inputs, outputs):
        if len(inputs) == 1:
            inputs = inputs[0]
        elif len(inputs) > 1:
            inputs = tuple(inputs)
        if len(outputs) == 0:
            return inputs
        elif len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) > 1:
            outputs = tuple(outputs)
        return inputs, outputs



class SharedMemEnqueuer(tf.keras.utils.OrderedEnqueuer): # adapted from tf.keras.utils.data_utils.OrderedEnqueuer
    """Builds an Enqueuer from a Sequence.

    Args:
        sequence_fun: A callable that generates a sequence `tf.keras.utils.data_utils.Sequence` object.
        shuffle: whether to shuffle the data at the beginning of each epoch
    """

    def __init__(self, sequence_len, sequence_fun, use_shm:bool = True, shuffle=True, shm_adaptor_cls=SharedArraySequenceAdapter):
        super().__init__(sequence_fun, True, shuffle)
        self.shuffle = shuffle
        self.use_shm = use_shm
        self.sequence_len = sequence_len
        context = multiprocessing.get_context("fork")
        self.context = context
        self.shm_adaptor_cls=shm_adaptor_cls
        class Worker(context.Process):
            def __init__(self, *args, **kwargs):
                """ Constructor.
                :return void
                """
                super().__init__(*args, **kwargs)
                self.sequence = shm_adaptor_cls(sequence_fun()) if use_shm else sequence_fun()
                self.epoch = 0

            def close(self):
                try:
                    self.sequence.close()
                except AttributeError:
                    pass
                super().close()

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
        super().start(workers, max_queue_size)

    def stop(self, timeout=None):
        super().stop(timeout)
        if self.shm_adaptor_cls == SharedArraySequenceAdapter: # erase ref in shared mem (only works with linux).
            try:
                for ad in sa.list():
                    name = ad.name.decode()
                    if name.startswith("shmnp") and len(name) == 37:
                        sa.delete(name)
            finally:
                pass

    def _send_sequence(self):
        pass
        #data_utils._SHARED_SEQUENCES[self.uid] = (self.sequences, self.sequence_indices)

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        sequence = list(range(self.sequence_len))
        with closing(self.executor_fn()) as executor:
            epoch = 0
            while True:
                if self.shuffle:
                    np.random.shuffle(sequence)
                for i in sequence:
                    if self.stop_signal.is_set():
                        return
                    self.queue.put(
                        executor.apply_async(get_index, (i, epoch)),
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
                        inputs, outputs = self.shm_adaptor_cls.decode(*item)
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

class OrderedEnqueuerShm(data_utils.OrderedEnqueuer):
    def __init__(self, sequence, use_multiprocessing=False, shuffle=False, use_shm=True, continuous=False, shm_adaptor_cls=SharedArraySequenceAdapter):
        super().__init__(shm_adaptor_cls(sequence) if use_shm else sequence, use_multiprocessing, shuffle)
        self.use_shm = use_shm
        self.continuous = continuous
        self.shm_adaptor_cls = shm_adaptor_cls

    def _run(self):
        if not self.continuous:
            super()._run()
        else:
            """Submits request to the executor and queue the `Future` objects."""
            sequence = list(range(len(self.sequence)))
            self._send_sequence()  # Share the initial sequence
            with closing(self.executor_fn(data_utils._SHARED_SEQUENCES)) as executor:
                while True:
                    if self.shuffle:
                        random.shuffle(sequence)
                    for i in sequence:
                        if self.stop_signal.is_set():
                            return
                        self.queue.put(executor.apply_async(data_utils.get_index, (self.uid, i)), block=True)

                    if self.stop_signal.is_set():
                        # We're done
                        return
                    self._wait_queue() # wait for final batches
                    # Call the internal on epoch end.
                    self.sequence.on_epoch_end()
                    self._send_sequence()  # Update the pool

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
                        inputs, outputs = self.shm_adaptor_cls.decode(*item)
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

    def stop(self, timeout=None):
        super().stop(timeout)
        if self.shm_adaptor_cls == SharedArraySequenceAdapter: # erase ref in shared mem (only works with linux).
            try:
                for ad in sa.list():
                    name = ad.name.decode()
                    if name.startswith("shmnp") and len(name) == 37:
                        sa.delete(name)
            finally:
                pass

def multiple(item):
    return isinstance(item, (list, tuple))

def get_index(idx, epoch):
    w = multiprocessing.current_process()
    w.set_epoch(epoch)
    return w.sequence[idx]