from tensorflow.keras.utils import OrderedEnqueuer
from multiprocessing import Pool, Queue, current_process
from keras.src.utils import data_utils
import queue
import random
from contextlib import closing
import numpy as np
# Global variables to be shared across processes
_SHARED_SEQUENCES = {}
# We use a Value to provide unique id to different processes.
_SEQUENCE_COUNTER = None


class OrderedEnqueuer2(OrderedEnqueuer):
    def __init__(self, sequence, maxtasksperchild=None, shuffle=False, single_epoch:bool=False):
        super(OrderedEnqueuer2, self).__init__(sequence, True, shuffle)
        self.maxtasksperchild=maxtasksperchild
        self.current_pool = None
        self.id_queue = None
        self.single_epoch=single_epoch
        self.shared_sequence = {}

    def _get_executor_init(self, workers):
        """Gets the Pool initializer for multiprocessing.

        Args:
            workers: Number of workers.

        Returns:
            Function, a Function to initialize the pool
        """

        def pool_fn(seqs):
            if self.current_pool is not None:
                self.current_pool.terminate()
            if self.id_queue is not None:
                self.id_queue._reset()
                self.id_queue.close()
            self.id_queue = Queue()
            self.current_pool = Pool(
                workers, initializer=init_pool_generator,
                initargs=(seqs, None, self.id_queue),
                maxtasksperchild=self.maxtasksperchild)
            return self.current_pool

        return pool_fn

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        sequence = list(range(len(self.sequence)))
        self._send_sequence()  # Share the initial sequence
        while True:
            if self.shuffle:
                random.shuffle(sequence)

            with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
                for i in sequence:
                    if self.stop_signal.is_set():
                        return

                    self.queue.put(
                        executor.apply_async(get_index, (self.uid, i)),
                        block=True,
                    )

                # Done with the current epoch, waiting for the final batches
                self._wait_queue()

                if self.stop_signal.is_set() or self.single_epoch:
                    # We're done
                    return

            # Call the internal on epoch end.
            self.sequence.on_epoch_end()
            self._send_sequence()  # Update the pool

    def _send_sequence(self):
        """Sends current Iterable to all workers."""
        # For new processes that may spawn
        _SHARED_SEQUENCES[self.uid] = self.sequence

    def stop(self, timeout=None):
        super().stop(timeout=timeout)
        if self.current_pool is not None:
            #l = len(data_utils._DATA_POOLS)
            #data_utils._DATA_POOLS.remove(self.current_pool)
            print(f"enqueuer: {self.uid} terminating pool")
            self.current_pool.terminate()
            self.current_pool = None
            #print(f"number of pools: {l} after remove: {len(data_utils._DATA_POOLS)}")
        if self.id_queue is not None:
            self.id_queue._reset()
            self.id_queue.close()
        _SHARED_SEQUENCES[self.uid] = None

def init_pool_generator(gens, random_seed=None, id_queue=None):
    """Initializer function for pool workers.

    Args:
      gens: State which should be made available to worker processes.
      random_seed: An optional value with which to seed child processes.
      id_queue: A multiprocessing Queue of worker ids. This is used to indicate
        that a worker process was created by Keras and can be terminated using
        the cleanup_all_keras_forkpools utility.
    """
    global _SHARED_SEQUENCES
    _SHARED_SEQUENCES = gens

    worker_proc = current_process()

    # name isn't used for anything, but setting a more descriptive name is
    # helpful when diagnosing orphaned processes.
    worker_proc.name = f"Keras_worker_{worker_proc.name}"

    if random_seed is not None:
        np.random.seed(random_seed + worker_proc.ident)

    if id_queue is not None:
        # If a worker dies during init, the pool will just create a replacement.
        id_queue.put(worker_proc.ident, block=True, timeout=0.1)


def next_sample(uid):
    """Gets the next value from the generator `uid`.

    To allow multiple generators to be used at the same time, we use `uid` to
    get a specific one. A single generator would cause the validation to
    overwrite the training generator.

    Args:
        uid: int, generator identifier

    Returns:
        The next value of generator `uid`.
    """
    return next(_SHARED_SEQUENCES[uid])
def get_index(uid, i):
    """Get the value from the Sequence `uid` at index `i`.

    To allow multiple Sequences to be used at the same time, we use `uid` to
    get a specific one. A single Sequence would cause the validation to
    overwrite the training Sequence.

    Args:
        uid: int, Sequence identifier
        i: index

    Returns:
        The value at index `i`.
    """
    return _SHARED_SEQUENCES[uid][i]