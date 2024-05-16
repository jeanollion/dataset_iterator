import os
import gc
import traceback
from concurrent.futures import ProcessPoolExecutor, CancelledError, TimeoutError
import multiprocessing
import random
import threading
import time
from threading import BoundedSemaphore
from .shared_memory import to_shm, from_shm, unlink_tensor_ref

# adapted from https://github.com/keras-team/keras/blob/v2.13.1/keras/utils/data_utils.py#L651-L776
# uses concurrent.futures, solves a memory leak in case of hard sample mining run as callback with regular orderedEnqueur. Option to pass tensors through shared memory
# Global variables to be shared across processes
_SHARED_SEQUENCES = {}
# We use a Value to provide unique id to different processes.
_SEQUENCE_COUNTER = None

class OrderedEnqueuerCF():
    def __init__(self, sequence, shuffle=False, single_epoch:bool=False, use_shm:bool=True, wait_for_me:threading.Event=None):
        self.sequence = sequence
        self.shuffle = shuffle
        self.single_epoch = single_epoch
        self.use_shm = use_shm
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
        self.stop_signal = None
        self.semaphore = None

    def is_running(self):
        return self.stop_signal is not None and not self.stop_signal.is_set()

    def start(self, workers=1, max_queue_size=10):
        """Starts the handler's workers.

        Args:
            workers: Number of workers.
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        """
        if self.use_shm:  # load in shared memory before spawning threads otherwise each thread will load in memory
            try:
                self.sequence.open()
            except AttributeError:
                pass
        self.workers = workers
        if max_queue_size <= 0:
            max_queue_size = self.workers
        self.semaphore = BoundedSemaphore(max_queue_size)
        self.queue = []
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _wait_queue(self, empty:bool):
        """Wait for the queue to be empty or not empty."""
        while True:
            if (empty and len(self.queue) == 0) or (not empty and len(self.queue) > 0) or self.stop_signal.is_set():
                return
            time.sleep(0.1)

    def _task_done(self, _):
        """Called once task is done, releases the queue if blocked."""
        self.semaphore.release()

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        task = get_item_shm if self.use_shm else get_item
        sequence = list(range(len(self.sequence)))
        self._send_sequence()  # Share the initial sequence
        while True:
            if self.shuffle:
                random.shuffle(sequence)
            executor = ProcessPoolExecutor(max_workers=self.workers)
            for idx, i in enumerate(sequence):
                if self.stop_signal.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    return
                self.semaphore.acquire()
                future = executor.submit(task, self.uid, i)
                self.queue.append((future, i))
            # Done with the current epoch, waiting for the final batches
            self._wait_queue(True) # safer to wait before calling shutdown than calling directly shutdown with wait=True
            time.sleep(0.1)
            executor.shutdown(wait=False, cancel_futures=True)
            del executor
            gc.collect()
            if self.stop_signal.is_set() or self.single_epoch:
                return
            if self.wait_for_me is not None:
                self.wait_for_me.wait()
            self._send_sequence()  # Update the pool

    def _send_sequence(self):
        """Sends current Iterable to all workers."""
        # For new processes that may spawn
        global _SHARED_SEQUENCES
        try:
            self.sequence.on_epoch_end()
        except AttributeError:
            pass
        _SHARED_SEQUENCES[self.uid] = self.sequence

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        Yields:
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        while self.is_running():
            self._wait_queue(False)
            if len(self.queue) > 0:
                future, i = self.queue[0]
                #print(f"processing task: {i}")
                ex = future.exception()
                if ex is None:
                    inputs = future.result()
                    if self.use_shm:
                        inputs = from_shm(*inputs)
                else:
                    traceback.print_exception(ex)
                    print(f"Exception raised while getting future result from task: {i}. Task will be re-computed.", flush=True)
                    inputs = get_item(self.uid, i)
                self.queue.pop(0)  # only remove after result() is called to avoid terminating pool while a process is still running
                self.semaphore.release()  # release is done here and not as a future callback to limit effective number of samples in memory
                future.cancel()
                del future
                yield inputs

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        Args:
            timeout: maximum time to wait on `thread.join()`
        """
        self.stop_signal.set()
        self.run_thread.join(timeout)
        if self.use_shm and self.queue is not None and len(self.queue) > 0:  # clean shm
            for (future, _) in self.queue:
                if future.exception() is None:
                    try:
                        unlink_tensor_ref(*future.result(timeout=0.1))
                    except CancelledError | TimeoutError:  # save to shm is the last step, if task was not finished it is likely not saved to shm
                        pass
        self.queue = None
        self.semaphore = None
        global _SHARED_SEQUENCES
        _SHARED_SEQUENCES[self.uid] = None

    def __del__(self):
        self.stop()


def get_item_shm(uid, i):
    tensors = _SHARED_SEQUENCES[uid][i]
    #print(f"item {i} -> {_SHARED_SEQUENCES[uid].index_array[i]} process: {os.getpid()}", flush=True)
    return to_shm(tensors)


def get_item(uid, i):
    return _SHARED_SEQUENCES[uid][i]

