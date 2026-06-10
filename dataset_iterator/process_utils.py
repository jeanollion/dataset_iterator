import os, time, subprocess, ctypes, threading
from psutil import NoSuchProcess, AccessDenied, wait_procs, Process
import concurrent.futures.process

# monkey patch -> executor shutdown easily hangs at join
try:
    def join_executor_internals(self):
        self.shutdown_workers()
        # Release the queue's resources as soon as possible.
        self.call_queue.close()
        self.call_queue.join_thread()
        with self.shutdown_lock:
            self.thread_wakeup.close()
        for p in self.processes.values():
            p.join(0.5)  # set a timeout to avoid hanging
            try:
                p.close()
            except:
                p.terminate()
    concurrent.futures.process._ExecutorManagerThread.join_executor_internals = join_executor_internals
except:
    pass

def kill_processes(pids, timeout=3, verbose=False):
    procs = get_procs(pids)
    gone, alive = wait_procs(procs, timeout=timeout)
    for p in alive:
        p.kill()
    time.sleep(0.1)
    procs = get_procs(pids)
    gone, alive = wait_procs(procs, timeout=timeout)
    if verbose and len(alive)>0:
        mem_leak = sum([p.memory_info().rss / float(2 ** 30) for p in alive])
        print(f"memory leak: {mem_leak:.2f}Gb among {len(alive)} processes", flush=True)
    return [p.pid for p in alive]


def get_procs(pids):
    procs = []
    curpid = os.getpid()
    for pid in pids:
        try:
            p = Process(pid)
            if p.ppid() == curpid:  # make sure pid was not reused by os
                procs.append(p)
        except (NoSuchProcess, AccessDenied):
            pass
    return procs


def log_used_mem():
    result = subprocess.check_output(['bash', '-c', 'free -m'])
    result = result.splitlines()
    free_memory = int(result[1].split()[2])/1000
    print(f"used memory: {free_memory:.1f}Gb", flush=True)

def get_num_fds():
    proc = Process(os.getpid())
    return proc.num_fds()


_libc = None
def malloc_trim():
    """Return free heap retained by glibc back to the OS (no-op off glibc).

    Python may free objects while glibc keeps the freed arenas, so RSS stays
    high and the per-epoch memory *floor* creeps up even when nothing leaks in
    the Python sense. Calling malloc_trim(0) after the per-epoch gc.collect()
    forces glibc to release that memory, keeping the floor flat."""
    global _libc
    try:
        if _libc is None:
            _libc = ctypes.CDLL("libc.so.6")
        _libc.malloc_trim(0)
        return True
    except Exception:
        return False


def count_threads(name_substr=None):
    """Number of live Python threads, optionally only those whose name contains
    `name_substr` (e.g. '-shutdown' to spot executor-shutdown threads that did
    not terminate)."""
    threads = threading.enumerate()
    if name_substr is None:
        return len(threads)
    return sum(1 for t in threads if name_substr in t.name)


def log_resources(tag=""):
    """One-line snapshot of the current process' resource footprint. Returns the
    values as a dict so callers can also act on them."""
    p = Process(os.getpid())
    rss = p.memory_info().rss / float(2 ** 30)
    os_threads = p.num_threads()
    py_threads = threading.active_count()
    shutdown_threads = count_threads("-shutdown")  # executor-shutdown threads that have not joined
    try:
        children = len(p.children())
    except (NoSuchProcess, AccessDenied):
        children = -1
    try:
        fds = p.num_fds()
    except (NoSuchProcess, AccessDenied):
        fds = -1
    print(f"[resources] {tag} rss={rss:.2f}Gb os_threads={os_threads} py_threads={py_threads} "
          f"unjoined_shutdown_threads={shutdown_threads} children={children} fds={fds}", flush=True)
    return dict(rss_gb=rss, os_threads=os_threads, py_threads=py_threads,
                shutdown_threads=shutdown_threads, children=children, fds=fds)