import gc
import os
import traceback
from .process_utils import kill_processes, log_used_mem  # this import needs to be before any import related to concurrent futures to pathc
from concurrent.futures import ProcessPoolExecutor, CancelledError, TimeoutError, as_completed
import multiprocessing
import random
import threading
import time
from threading import BoundedSemaphore
from .shared_memory import to_shm, from_shm, unlink_tensor_ref

# adapted from https://github.com/keras-team/keras/blob/v2.13.1/keras/utils/data_utils.py#L651-L776
# uses concurrent.futures, solves a memory leak in case of hard sample mining run as callback with regular orderedEnqueur. Option to pass tensors through shared memory
# Global variables to be shared across processes
_SHARED_ITERATOR = {}
# We use a Value to provide unique id to different processes.
_COUNTER = None


class OrderedEnqueuerCF():
    def __init__(self, iterator, shuffle=False, single_epoch:bool=False, use_shm:bool=False, use_shared_array:bool=True):
        self.iterator = iterator
        self.shuffle = shuffle
        self.single_epoch = single_epoch
        self.use_shm = use_shm
        self.use_shared_array=use_shared_array
        assert not self.use_shm and not self.use_shared_array or self.use_shm != self.use_shared_array, "either shm or shared_array or none of the 2"
        self.wait_for_me_supplier = None
        self.wait_for_me_consumer = None
        global _COUNTER
        if _COUNTER is None:
            try:
                _COUNTER = multiprocessing.Value("i", 0)
            except OSError:
                # In this case the OS does not allow us to use
                # multiprocessing. We resort to an int
                # for enqueuer indexing.
                _COUNTER = 0

        if isinstance(_COUNTER, int):
            self.uid = _COUNTER
            _COUNTER += 1
        else:
            # Doing Multiprocessing.Value += x is not process-safe.
            with _COUNTER.get_lock():
                self.uid = _COUNTER.value
                _COUNTER.value += 1

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
        try:
            self.iterator.open()  # load in shared memory before spawning threads otherwise each thread will load in memory
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
        if self.wait_for_me_supplier is not None:
            self.wait_for_me_supplier.wait()
        log_used_mem()
        if self.use_shm:
            task = get_item_shm
        elif self.use_shared_array:
            task = get_item_shared_array
        else:
            task = get_item
        indices = list(range(len(self.iterator)))
        self._send_iterator()  # Share the initial sequence
        while True:
            if self.shuffle:
                random.shuffle(indices)
            executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context('fork'), initializer=init_pool_generator, initargs=(self.uid, self.iterator))
            for idx, i in enumerate(indices):
                if self.stop_signal.is_set():
                    processes = list(executor._processes.keys())
                    executor.shutdown(wait=True, cancel_futures=True)
                    del executor
                    self.leaked_processes = kill_processes(processes, timeout=1, verbose=True)
                    self._clear_iterator()
                    return
                self.semaphore.acquire()
                future = executor.submit(task, self.uid, i)
                self.queue.append((future, i))
            # Done with the current epoch, waiting for the final batches
            self._wait_queue(True)  # safer to wait before calling shutdown than calling directly shutdown with wait=True

            futures = [executor.submit(close_iterator, self.uid) for _ in range(self.workers)]  #  close iterator in each process's memory TODO necesary ?
            for _ in as_completed(futures, timeout=5):
                pass
            del futures
            processes = list(executor._processes.keys())
            executor.shutdown(wait=True, cancel_futures=True)  # wait=True often hangs because no timeout is set to Process.join().
            del executor
            kill_processes(processes, timeout=3, verbose=True)
            self._clear_iterator()
            gc.collect()
            if self.stop_signal.is_set() or self.single_epoch:
                return
            if self.wait_for_me_supplier is not None:
                self.wait_for_me_supplier.wait()
            log_used_mem()
            indices = list(range(len(self.iterator)))
            self._send_iterator()  # Update the pool

    def _send_iterator(self):
        """Sends current Iterable to all workers."""
        # For new processes that may spawn
        global _SHARED_ITERATOR
        try:
            self.iterator.on_epoch_end()
        except AttributeError:
            pass
        _SHARED_ITERATOR[self.uid] = self.iterator

    def _clear_iterator(self):
        """Sends current Iterable to all workers."""
        # For new processes that may spawn
        global _SHARED_ITERATOR
        _SHARED_ITERATOR[self.uid] = None

    def get(self):
        return self.get_wfm(self.wait_for_me_consumer)

    def get_wfm(self, wait_for_me:threading.Event):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        Yields:
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        while self.is_running():
            self._wait_queue(False)
            if wait_for_me is not None:
                wait_for_me.wait()
                self._wait_queue(False)

            if len(self.queue) > 0:
                future, i = self.queue[0]
                #print(f"processing task: {i}")
                ex = future.exception()
                if ex is None:
                    inputs = future.result()
                    if self.use_shm or self.use_shared_array:
                        inputs = from_shm(*inputs)
                else:
                    print(f"Exception raised while getting future result from task: {i}. Task will be re-computed.", flush=True)
                    traceback.print_exception(ex)
                    try:
                        inputs = get_item(self.uid, i)
                        print(f"Task {i} successfully re-computed.", flush=True)
                    except Exception as e:
                        print(f"Exception raised while trying to re-compute task {i}. Stopping the pool.", flush=True)
                        traceback.print_exception(e)
                        self.stop()
                        return
                self.queue.pop(0)  # only remove after result() is called to avoid terminating pool while a process is still running
                self.semaphore.release()  # release is done here and not as a future callback to limit effective number of samples in memory
                future.cancel()
                del future
                yield inputs

    def stop(self, timeout=5):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        Args:
            timeout: maximum time to wait on `thread.join()`
        """
        if self.run_thread is None:  # has not been started
            return
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
        self._clear_iterator()

    def __del__(self):
        self.stop()


def get_item_shm(uid, i):
    tensors = _SHARED_ITERATOR[uid][i]
    #print(f"item {i} -> {_SHARED_SEQUENCES[uid].index_array[i]} process: {os.getpid()}", flush=True)
    return to_shm(tensors)


def get_item_shared_array(uid, i):
    tensors = _SHARED_ITERATOR[uid][i]
    #print(f"item {i} -> {_SHARED_SEQUENCES[uid].index_array[i]} process: {os.getpid()}", flush=True)
    return to_shm(tensors, use_shared_array=True)


def get_item(uid, i):
    return _SHARED_ITERATOR[uid][i]


def close_iterator(uid):  # method intended to be called by each process to free memory related to iterator
    if _SHARED_ITERATOR[uid] is not None:
        _SHARED_ITERATOR[uid].close(force=True)
        _SHARED_ITERATOR[uid] = None
        time.sleep(0.5)


def init_pool_generator(uid, seq):
    global _SHARED_ITERATOR
    _SHARED_ITERATOR = {uid:seq}
