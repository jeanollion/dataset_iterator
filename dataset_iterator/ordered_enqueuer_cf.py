import gc
import os
import statistics
import traceback
import dill
from .process_utils import kill_processes, log_used_mem, \
    get_num_fds, malloc_trim, log_resources  # this import needs to be before any import related to concurrent futures to patch
from concurrent.futures import ProcessPoolExecutor, CancelledError, TimeoutError, as_completed
from collections import deque
import multiprocessing
import random
import threading
import time
from threading import BoundedSemaphore
from .shared_memory import to_shm, from_shm, unlink_tensor_ref, unlink_shared_array

def _poll_event(event, poll_interval=0.05):
    """Fork-safe alternative to Event.wait().

    Event.wait() internally acquires a Condition lock. If fork() happens
    while another thread is inside Event.wait(), the child inherits the
    lock in a held state with no thread to release it — causing a deadlock
    in the forked worker. Polling with is_set() + sleep avoids this:
    is_set() is a plain attribute read (no lock), and sleep holds no
    Python locks.
    """
    while not event.is_set():
        time.sleep(poll_interval)


# adapted from https://github.com/keras-team/keras/blob/v2.13.1/keras/utils/data_utils.py#L651-L776
# uses concurrent.futures, solves a memory leak in case of hard sample mining run as callback with regular orderedEnqueur. Option to pass tensors through shared memory
# Global variables to be shared across processes
_SHARED_ITERATOR = {}
# Graceful window when shutting a pool down because stop() was requested: short,
# since any in-flight results are discarded anyway -> stop returns promptly.
_STOP_SHUTDOWN_GRACE = 2.
# We use a Value to provide unique id to different processes.
_COUNTER = None

class OrderedEnqueuerCF:
    def __init__(self, iterator, shuffle=False, single_epoch:bool=False, use_shm:bool=False, use_shared_array:bool=False, max_restarts:int=10, max_steps=0, step_duration=60, name="enqueuer", log_resources:bool=False):
        self.iterator = iterator
        self.shuffle = shuffle
        self.log_resources = log_resources  # per-epoch RSS / thread / child-process diagnostics
        self._epoch = 0
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
        self.step_duration=step_duration # estimate used for timeout, will be measured after 1st epoch
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
        self._step_durations = [] # step duration for the epoch
        self.record_step_duration = True
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

    def _shutdown_timeout(self):
        # Graceful-shutdown grace period, scaled to the measured step duration so a
        # worker that is legitimately mid-batch can finish (a too-short window is
        # what SIGKILL-ed busy workers and stranded their results in the parent).
        # executor.shutdown(wait=True) returns as soon as the (already drained)
        # workers exit, so in the common case this is not actually waited out; it
        # only bounds the rare stuck-worker case before we hand off to the reaper.
        return max(10., 1.5 * self.step_duration)

    def _shutdown_executor_bounded(self, executor, timeout=None):
        """Synchronously shut `executor` down within a step-scaled grace period,
        then hand any survivors to the background reaper.

        Synchronous on purpose: the next epoch must not fork a new pool while this
        executor's manager thread is still alive — forking over a half-torn-down
        pool corrupts the child and yields the 'NoneType has no attribute poll'
        submit errors. So we wait (bounded) for graceful shutdown; if workers
        overrun the grace period we SIGKILL them so the manager thread can exit
        promptly, and any process that survives even SIGKILL (e.g. stuck in
        uninterruptible IO) is queued to the reaper daemon, which keeps hunting it
        off the training critical path. Net effect: bounded stall, no stranded
        zombies accumulating across epochs."""
        if timeout is None:
            timeout = self._shutdown_timeout()
        processes = list(executor._processes.keys()) if executor._processes is not None else None
        done = threading.Thread(target=lambda: executor.shutdown(wait=True, cancel_futures=True),
                                name=f"{self.name}({self.uid})-shutdown", daemon=True)
        done.start()
        done.join(timeout)
        if done.is_alive():
            # Grace period elapsed: force-kill so the manager thread can join, then
            # give it a brief moment to finish. Survivors go to the reaper.
            if processes:
                survivors = kill_processes(processes, timeout=2, verbose=True)
                _REAPER.enqueue(survivors)
            done.join(timeout=min(10., timeout))
            if done.is_alive():
                # Rare: manager thread still not exited after the workers were killed.
                # The reaper keeps mopping up; at an epoch boundary the upcoming fork
                # could race it (on the stop path there is no further fork).
                print(f"{self.name}({self.uid}) WARNING: executor manager thread still alive after "
                      f"force-kill of its workers", flush=True)
        # Whichever path we took, release the executor's parent-side fds explicitly.
        # This is what prevents the steady "Too many open files" fd leak: on the
        # force path join_executor_internals could only Process.terminate() the
        # stuck workers (Process.close() raises while they are alive), leaving each
        # worker's sentinel fd — plus the result_queue pipe the patch never closes —
        # open, ~2 fds per stuck worker per epoch until the process hits its ulimit.
        self._release_executor_resources(executor)

    def _release_executor_resources(self, executor):
        """Close the parent-side fds an executor holds: each worker Popen's
        sentinel fd and the call/result/thread-wakeup pipes. Safe to call once the
        workers are dead (clean shutdown finished, or reaper SIGKILL); every close
        is guarded so re-closing already-closed objects (clean path) is a no-op."""
        procs = getattr(executor, "_processes", None)
        if procs:
            for p in list(procs.values()):
                try:
                    p.join(0)  # reap if it just died, so close() won't raise
                except Exception:
                    pass
                try:
                    p.close()  # releases the sentinel fd terminate() leaves open
                except Exception:
                    pass
        for attr in ("_call_queue", "_result_queue"):
            q = getattr(executor, attr, None)
            if q is None:
                continue
            try:
                q.close()
            except Exception:
                pass
            jt = getattr(q, "join_thread", None)  # SimpleQueue has no feeder thread
            if jt is not None:
                try:
                    jt()
                except Exception:
                    pass
        tw = getattr(executor, "_thread_wakeup", None)
        if tw is not None:
            try:
                tw.close()  # closes the self-pipe reader+writer
            except Exception:
                pass

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        if self.wait_for_me_supplier is not None:
            #was_locked = not self.wait_for_me_supplier.is_set()
            #if was_locked:
            #    print(f"{self.name}({self.uid}) S waiting supplier...", flush=True)
            #self.wait_for_me_supplier.wait()
            _poll_event(self.wait_for_me_supplier)
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
            if self.stop_signal.is_set():  # stop requested between epochs: exit before forking a new pool
                self._clear_iterator()
                return
            self._step_durations.clear()
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
                # interruptible acquire: on stop() the consumer stops releasing the
                # semaphore, so block in short slices and bail out instead of hanging
                while not self.semaphore.acquire(timeout=0.5):
                    if self.stop_signal.is_set():
                        break
                if self.stop_signal.is_set():
                    self._shutdown_executor_bounded(executor, timeout=_STOP_SHUTDOWN_GRACE)
                    self._clear_iterator()
                    return
                #print(f"{self.name}({self.uid}) supply task: {i} semaphore: {self.semaphore._value} queue: {len(self.queue)}", flush=True)
                while restarts < self.max_restarts:
                    if self.stop_signal.is_set():
                        self._shutdown_executor_bounded(executor, timeout=_STOP_SHUTDOWN_GRACE)
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
                        self._shutdown_executor_bounded(executor)
                        executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp_context, initializer=init_pool_generator, initargs=get_init_pool_args(self.iterator))
                        print(f"Executor {self.name}({self.uid}) restarted! ", flush=True)
                        restarts += 1

            # Done with the current epoch, waiting for the final batches
            self.wait_queue(True)  # safer to wait before calling shutdown than calling directly shutdown with wait=True
            self.supplying_signal.clear()
            # Bounded synchronous shutdown so the manager thread is gone before the
            # next epoch forks; stragglers are reaped in the background.
            self._shutdown_executor_bounded(executor)
            self._clear_iterator()
            self._cleanup_orphaned_futures()
            del executor
            gc.collect()
            if self.record_step_duration and len(self._step_durations) > 0:
                step_duration = statistics.median(self._step_durations)
                #print(f"{self.name}({self.uid}) step duration: median={step_duration} range: [{min(self._step_durations)}, {max(self._step_durations)}] timeout: {self.step_duration} -> {step_duration * 1.5}")
                self.step_duration = step_duration
            self._epoch += 1
            if self.log_resources:
                log_resources(f"{self.name}({self.uid}) end-epoch {self._epoch} step_dur={self.step_duration:.2f}s reaper_pending={_REAPER.pending_count()}")
            self.supplying_end_signal.set()
            #print(f"{self.name}({self.uid}) Supplying signal off", flush=True)

            if self.wait_for_me_supplier is not None:
                if self.request_lock() and self.wait_for_me_supplier.is_set():
                    #print(f"{self.name}({self.uid}) lock requested", flush=True)
                    self.wait_for_me_supplier.clear()
                #was_locked = not self.wait_for_me_supplier.is_set()
                #if was_locked:
                #    print(f"{self.name}({self.uid}) waiting supplier...", flush=True)
                #self.wait_for_me_supplier.wait()
                _poll_event(self.wait_for_me_supplier)
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
                #wait_for_me.wait()
                _poll_event(wait_for_me)
                #if was_locked:
                #    print(f"{name}({self.uid}) waiting consumer done", flush=True)
                if block:
                    self.wait_queue(False)
            if len(self.queue) > 0:
                future, i = self.queue[0]
                #print(f"{name}({self.uid}) is processing task: {i} (queue: {len(self.queue)})", flush=True)
                try:
                    ex = future.exception(timeout=self.step_duration * 1.75)
                except TimeoutError as toex:
                    #print(f"{name}({self.uid}) timeout error consumer")
                    ex = toex
                if ex is None:
                    tensors, dt = future.result()
                    if self.use_shm or self.use_shared_array:
                        tensors = from_shm(*tensors)
                    if self.record_step_duration:
                        self._step_durations.append(dt)
                else:
                    #print(f"Exception raised while getting future result from task: {i}. Task will be re-computed.", flush=True)
                    #traceback.print_exception(ex)
                    # Clean up shared memory that may have been allocated by the worker
                    if self.use_shm or self.use_shared_array:
                        self._cleanup_future_shm(future)
                    try:
                        tensors, dt = get_item(self.uid, i)
                        if self.record_step_duration:
                            self._step_durations.append(dt)
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
                yield tensors
            elif not block and not self.supplying_signal.is_set() and _SHARED_ITERATOR.get(self.uid) is not None:
                #print(f"{name}({self.uid}) yield item 0 to avoid blocking")
                tensors, _ = get_item(self.uid, 0)
                yield tensors
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
        # Do not join the run thread from within itself: stop() can be reached via
        # __del__ when the enqueuer is finalized on its own run thread (CPython drops
        # the bound-method reference inside Thread.run after _run returns), and
        # joining the current thread raises RuntimeError("cannot join current thread").
        if threading.current_thread() is not self.run_thread:
            self.run_thread.join(timeout)
            if self.run_thread.is_alive():
                # Supplier thread did not exit in time (e.g. still draining a slow
                # pool). Leave shared state intact and let it wind down on its own:
                # nulling self.semaphore / self.queue here is exactly what crashed it
                # with "'NoneType' object has no attribute 'acquire'". State is
                # released when the enqueuer object itself is dropped.
                print(f"{self.name}({self.uid}) stop: supplier thread still running after {timeout}s; "
                      f"deferring teardown (it will exit on its own)", flush=True)
                return
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
    start = time.time()
    tensors = _SHARED_ITERATOR[uid][i]
    tensors = to_shm(tensors, use_shared_array=False)
    end = time.time()
    return tensors, end - start


def get_item_shared_array(uid, i):
    start = time.time()
    tensors = _SHARED_ITERATOR[uid][i]
    tensors = to_shm(tensors, use_shared_array=True)
    end = time.time()
    return tensors, end - start


def get_item(uid, i):
    start = time.time()
    tensors = _SHARED_ITERATOR[uid][i]
    end = time.time()
    return tensors, end - start

def close_iterator(uid):  # method intended to be called by each process to free memory related to iterator
    if _SHARED_ITERATOR[uid] is not None:
        _SHARED_ITERATOR[uid].close()
        _SHARED_ITERATOR[uid] = None
        time.sleep(0.5)


def init_pool_generator(uid, seq, unpickle):
    global _SHARED_ITERATOR
    _SHARED_ITERATOR = {uid:dill.loads(seq) if unpickle else seq}


def shutdown_executor(executor, timeout=30):
    processes = list(executor._processes.keys()) if executor._processes is not None else None
    # Run shutdown in a thread with a timeout to avoid hanging indefinitely
    shutdown_thread = threading.Thread(target=lambda: executor.shutdown(wait=True, cancel_futures=True))
    shutdown_thread.start()
    shutdown_thread.join(timeout=timeout)
    #if shutdown_thread.is_alive():
    #    print(f"Warning: executor.shutdown() did not complete within {timeout}s, force-killing workers", flush=True)
    del executor
    if processes is not None:
        kill_processes(processes, timeout=timeout, verbose=True)
    time.sleep(0.1)


class _ProcessReaper:
    """Single background daemon that hunts down worker processes left over from a
    bounded executor shutdown (workers that survived the grace period + SIGKILL,
    e.g. briefly stuck in uninterruptible IO). It keeps killing/reaping them off
    the training critical path so stragglers can't accumulate across epochs (the
    memory growth we are trying to avoid) without ever stalling the supplier loop.

    Shared module-wide because worker PIDs are globally unique across the several
    enqueuers (main / validation / hard-sample-mining).

    Each batch carries an attempt counter; a batch that is still alive after
    `max_attempts` sweeps is logged and dropped (truly unkillable, e.g. a defunct
    process the OS will reap, or one wedged in D-state) so the daemon never spins
    on it forever."""
    def __init__(self, poll_interval:float=2., max_attempts:int=30):
        self._pending = deque()  # (list[int] pids, attempts)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._poll_interval = poll_interval
        self._max_attempts = max_attempts
        self._thread = None
        self._total_reaped = 0

    def pending_count(self):
        with self._lock:
            return sum(len(pids) for pids, _ in self._pending)

    def total_reaped(self):
        with self._lock:
            return self._total_reaped

    def enqueue(self, pids):
        if not pids:
            return
        with self._lock:
            self._pending.append((list(pids), 0))
        self._ensure_running()
        self._wake.set()

    def _ensure_running(self):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="enqueuer-reaper", daemon=True)
                self._thread.start()

    def _run(self):
        while True:
            self._wake.wait(timeout=self._poll_interval)
            self._wake.clear()
            with self._lock:
                batches = list(self._pending)
                self._pending.clear()
            requeue = []
            for pids, attempts in batches:
                alive = kill_processes(pids, timeout=1, verbose=False)  # SIGKILL + brief reap
                reaped = len(pids) - len(alive)
                if reaped:
                    with self._lock:
                        self._total_reaped += reaped
                    print(f"[reaper] reaped {reaped} straggler worker process(es) "
                          f"(total {self._total_reaped})", flush=True)
                if alive:
                    attempts += 1
                    if attempts >= self._max_attempts:
                        print(f"[reaper] giving up on {len(alive)} unkillable worker process(es) after "
                              f"{attempts} attempts (defunct / uninterruptible): {alive}", flush=True)
                    else:
                        requeue.append((alive, attempts))
            if requeue:  # stubborn PIDs: try again next sweep
                with self._lock:
                    self._pending.extend(requeue)
                time.sleep(self._poll_interval)  # don't busy-spin on uninterruptible zombies


_REAPER = _ProcessReaper()