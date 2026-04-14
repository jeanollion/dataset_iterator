import gc
import os
import traceback
import dill
from .process_utils import kill_processes, log_used_mem, \
    get_num_fds  # this import needs to be before any import related to concurrent futures to patch
from concurrent.futures import ProcessPoolExecutor, CancelledError, TimeoutError, as_completed
from collections import deque
import multiprocessing
import random
import threading
import time
from threading import BoundedSemaphore
from .shared_memory import to_shm, from_shm, unlink_tensor_ref, unlink_shared_array

# adapted from https://github.com/keras-team/keras/blob/v2.13.1/keras/utils/data_utils.py#L651-L776
# uses concurrent.futures, solves a memory leak in case of hard sample mining run as callback with regular orderedEnqueur. Option to pass tensors through shared memory
# Global variables to be shared across processes
_SHARED_ITERATOR = {}
# We use a Value to provide unique id to different processes.
_COUNTER = None

class OrderedEnqueuerCF:
    def __init__(self, iterator, shuffle=False, single_epoch:bool=False, use_shm:bool=False, use_shared_array:bool=True, max_restarts:int=10, max_steps=0, step_timeout=10, name="enqueuer"):
        self.iterator = iterator
        self.shuffle = shuffle
        self.single_epoch = single_epoch
        self.use_shm = use_shm
        self.use_shared_array=use_shared_array
        assert not self.use_shm and not self.use_shared_array or self.use_shm != self.use_shared_array, "either shm or shared_array or none of the 2"
        self.wait_for_me_supplier = threading.Event() # wait to start the epoch
        self.wait_for_me_supplier.set()
        self.request_lock_list = []
        self.supplying_signal = threading.Event()
        self.supplying_signal.clear() # wait -> until is supplying
        self.supplying_end_signal = threading.Event()
        self.supplying_end_signal.set()  # wait -> until end of epoch
        self.wait_for_me_consumer = threading.Event()
        self.wait_for_me_consumer.set()
        assert max_restarts > 0
        self.max_restarts=max_restarts
        self.max_steps=max_steps
        self.task_timeout=step_timeout
        self.name=name
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
        self.semaphore = None
        self._orphaned_futures = []  # futures that timed out but may still hold shm references
        #if isinstance(self.iterator.datasetIO, MemoryIO):
        #    print(f"{self.name}({self.uid}) iterator is shm {self.iterator.datasetIO.use_shm} sa {self.iterator.datasetIO.use_shared_array} open datasets: {len(self.iterator.datasetIO.datasets)}", flush=True)

    def request_lock(self):
        return True in self.request_lock_list

    def append_request_lock(self, request_lock):
        self.request_lock_list.append(request_lock)
        return len(self.request_lock_list) - 1

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
            self.iterator_params = self.iterator.enqueuer_init()
        except AttributeError:
            self.iterator_params = None
        self.workers = workers
        if max_queue_size <= 0:
            max_queue_size = self.workers
        self.semaphore = BoundedSemaphore(max_queue_size)
        self.queue = deque()
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def wait_queue(self, empty:bool):
        """Wait for the queue to be empty or not empty."""
        while True:
            if (empty and len(self.queue) == 0) or (not empty and len(self.queue) > 0) or self.stop_signal.is_set():
                return
            time.sleep(0.1)

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        if self.wait_for_me_supplier is not None:
            #was_locked = not self.wait_for_me_supplier.is_set()
            #if was_locked:
            #    print(f"{self.name}({self.uid}) S waiting supplier...", flush=True)
            self.wait_for_me_supplier.wait()
            #if was_locked:
            #    print(f"{self.name}({self.uid}) S waiting supplier done", flush=True)
        if self.use_shm:
            task = get_item_shm
        elif self.use_shared_array:
            task = get_item_shared_array
        else:
            task = get_item
        indices = list(range(len(self.iterator)))
        self._send_iterator()  # Share the initial sequence
        mp_context_method = "fork"
        try:
            mp_context = multiprocessing.get_context(mp_context_method)
        except ValueError:  # method not available
            mp_context_method = "spawn"
            mp_context = multiprocessing.get_context(mp_context_method)
        def get_init_pool_args(iterator):
            return self.uid, iterator if mp_context_method == "fork" else dill.dumps(iterator), mp_context_method != "fork"

        while True:
            #print(f"{self.name}({self.uid}) epoch start: open fds: {get_num_fds()}", flush=True)
            self.supplying_signal.set()
            self.supplying_end_signal.clear()
            #print(f"{self.name}({self.uid}) enqueuer start epoch. semaphore: {self.semaphore._value}", flush=True)
            if self.shuffle:
                random.shuffle(indices)
            executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp_context, initializer=init_pool_generator, initargs=get_init_pool_args(self.iterator))
            step_number = min(self.max_steps, len(indices)) if self.max_steps > 0 else len(indices)
            for idx in range(step_number):
                i = indices[idx]
                restarts = 0
                self.semaphore.acquire()
                #print(f"{self.name}({self.uid}) supply task: {i} semaphore: {self.semaphore._value} queue: {len(self.queue)}", flush=True)
                while restarts < self.max_restarts:
                    if self.stop_signal.is_set():
                        shutdown_executor(executor)
                        self._clear_iterator()
                        return
                    try:
                        future = executor.submit(task, self.uid, i)
                        self.queue.append((future, i))
                        break  # Task submitted successfully, move to next task
                    except Exception as e:
                        if restarts == self.max_restarts:
                            raise ValueError(f"Failed to submit task for index {i} after {self.max_restarts} attempts. {e}")
                        print(f"Executor {self.name}({self.uid}) error for index {i} (attempt {restarts + 1}/{self.max_restarts}): {e}. Restarting executor...", flush=True)
                        self.wait_queue(True)
                        #with _EXECUTOR_LOCK:
                        shutdown_executor(executor)
                        executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp_context, initializer=init_pool_generator, initargs=get_init_pool_args(self.iterator))
                        print(f"Executor {self.name}({self.uid}) restarted! ", flush=True)
                        restarts += 1

            # Done with the current epoch, waiting for the final batches
            self.wait_queue(True)  # safer to wait before calling shutdown than calling directly shutdown with wait=True
            self.supplying_signal.clear()
            shutdown_executor(executor)
            self._clear_iterator()
            self._cleanup_orphaned_futures()
            del executor
            gc.collect()
            self.supplying_end_signal.set()
            #print(f"{self.name}({self.uid}) Supplying signal off", flush=True)

            if self.wait_for_me_supplier is not None:
                if self.request_lock() and self.wait_for_me_supplier.is_set():
                    #print(f"{self.name}({self.uid}) lock requested", flush=True)
                    self.wait_for_me_supplier.clear()
                #was_locked = not self.wait_for_me_supplier.is_set()
                #if was_locked:
                #    print(f"{self.name}({self.uid}) waiting supplier...", flush=True)
                self.wait_for_me_supplier.wait()
                #if was_locked:
                #    print(f"{self.name}({self.uid}) supplier waiting done", flush=True)
            #log_used_mem()
            #print(f"{self.name}({self.uid}) sending iterator")
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

    def _cleanup_future_shm(self, future):
        """Try to clean up shm from a failed/timed-out future. If not ready yet, track it for later cleanup."""
        try:
            result = future.result(timeout=0.1)
            if self.use_shm:
                unlink_tensor_ref(*result)
            else:
                unlink_shared_array(*result[0])
        except (CancelledError, TimeoutError):
            # Worker may still be running — track for later cleanup
            self._orphaned_futures.append(future)
        except Exception:
            pass

    def _cleanup_orphaned_futures(self):
        """Clean up shm from futures that previously timed out but may have completed since."""
        remaining = []
        for future in self._orphaned_futures:
            if future.done():
                if future.exception() is None:
                    try:
                        result = future.result(timeout=0.1)
                        if self.use_shm:
                            unlink_tensor_ref(*result)
                        else:
                            unlink_shared_array(*result[0])
                    except Exception:
                        pass
            else:
                remaining.append(future)
        self._orphaned_futures = remaining

    def get(self, block:bool=True, name="main"):
        return self.get_wfm(self.wait_for_me_consumer, block=block, name=name)

    def get_wfm(self, wait_for_me:threading.Event, block:bool=True, name:str="main"):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        Yields:
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        while self.is_running():
            if block:
                self.wait_queue(False)
            if wait_for_me is not None:
                #was_locked = not wait_for_me.is_set()
                #if was_locked:
                #    print(f"{name}({self.uid}) waiting consumer...", flush=True)
                wait_for_me.wait()
                #if was_locked:
                #    print(f"{name}({self.uid}) waiting consumer done", flush=True)
                if block:
                    self.wait_queue(False)
            if len(self.queue) > 0:
                future, i = self.queue[0]
                #print(f"{name}({self.uid}) is processing task: {i} (queue: {len(self.queue)})", flush=True)
                try:
                    ex = future.exception(timeout=self.task_timeout)
                except TimeoutError as toex: # this happens often when validation is enabled. TODO: findout why
                    ex = toex
                if ex is None:
                    inputs = future.result()
                    if self.use_shm or self.use_shared_array:
                        inputs = from_shm(*inputs)
                else:
                    #print(f"Exception raised while getting future result from task: {i}. Task will be re-computed.", flush=True)
                    #traceback.print_exception(ex)
                    # Clean up shared memory that may have been allocated by the worker
                    if self.use_shm or self.use_shared_array:
                        self._cleanup_future_shm(future)
                    try:
                        inputs = get_item(self.uid, i)
                        #print(f"Task {i} successfully re-computed.", flush=True)
                    except Exception as e:
                        print(f"Exception raised while trying to re-compute task {i}. Stopping the pool.", flush=True)
                        traceback.print_exception(e)
                        self.stop()
                        return
                self.queue.popleft()  # only remove after result() is called to avoid terminating pool while a process is still running
                self.semaphore.release()  # release is done here and not as a future callback to limit effective number of samples in memory
                future.cancel()
                del future
                #print(f"{name}({self.uid}) has processed task: {i} (semaphore: {self.semaphore._value} queue: {len(self.queue)})", flush=True)
                yield inputs
            elif not block and not self.supplying_signal.is_set() and _SHARED_ITERATOR.get(self.uid) is not None:
                #print(f"{name}({self.uid}) yield item 0 to avoid blocking")
                yield get_item(self.uid, 0)
            else:
                time.sleep(0.01)

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
        if (self.use_shm or self.use_shared_array) and self.queue is not None and len(self.queue) > 0:  # clean shm
            for (future, _) in self.queue:
                try:
                    ex = future.exception(timeout=1)
                except (CancelledError, TimeoutError):
                    continue
                if ex is None:
                    try:
                        if self.use_shm:
                            unlink_tensor_ref(*future.result(timeout=0.1))
                        else:
                            unlink_shared_array(*future.result(timeout=0.1)[0])
                    except (CancelledError, TimeoutError):
                        pass
        if self.use_shm or self.use_shared_array:
            self._cleanup_orphaned_futures()
            if self._orphaned_futures:
                # Last resort: wait a bit for remaining orphans then clean
                time.sleep(1)
                self._cleanup_orphaned_futures()
                if self._orphaned_futures:
                    print(f"Warning: {self.name}({self.uid}) {len(self._orphaned_futures)} orphaned shm futures could not be cleaned up", flush=True)
        self.queue = None
        self.semaphore = None
        self._clear_iterator()
        if self.iterator_params is not None:
            self.iterator.enqueuer_end(self.iterator_params)

    def __del__(self):
        self.stop()


def get_item_shm(uid, i):
    tensors = _SHARED_ITERATOR[uid][i]
    #print(f"item {i} -> {_SHARED_SEQUENCES[uid].index_array[i]} process: {os.getpid()}", flush=True)
    return to_shm(tensors, use_shared_array=False)


def get_item_shared_array(uid, i):
    tensors = _SHARED_ITERATOR[uid][i]
    #print(f"item {i} -> {_SHARED_SEQUENCES[uid].index_array[i]} process: {os.getpid()}", flush=True)
    return to_shm(tensors, use_shared_array=True)


def get_item(uid, i):
    return _SHARED_ITERATOR[uid][i]


def close_iterator(uid):  # method intended to be called by each process to free memory related to iterator
    if _SHARED_ITERATOR[uid] is not None:
        _SHARED_ITERATOR[uid].close()
        _SHARED_ITERATOR[uid] = None
        time.sleep(0.5)


def init_pool_generator(uid, seq, unpickle):
    global _SHARED_ITERATOR
    _SHARED_ITERATOR = {uid:dill.loads(seq) if unpickle else seq}


def shutdown_executor(executor, timeout=10):
    processes = list(executor._processes.keys()) if executor._processes is not None else None
    # Run shutdown in a thread with a timeout to avoid hanging indefinitely
    shutdown_thread = threading.Thread(target=lambda: executor.shutdown(wait=True, cancel_futures=True))
    shutdown_thread.start()
    shutdown_thread.join(timeout=timeout)
    if shutdown_thread.is_alive():
        print(f"Warning: executor.shutdown() did not complete within {timeout}s, force-killing workers", flush=True)
    del executor
    if processes is not None:
        kill_processes(processes, timeout=timeout, verbose=True)
    time.sleep(0.1)