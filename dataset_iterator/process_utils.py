import os, time, subprocess, ctypes, threading, gc
from psutil import NoSuchProcess, AccessDenied, wait_procs, Process
import concurrent.futures.process

# Opt-in (DATASET_ITERATOR_HEAP_PROBE=1): fork-safe Python-heap probe in log_resources,
# to attribute a sudden RSS step to Python objects (esp. numpy) vs native memory
# (TF/CUDA/driver/glibc). tracemalloc is deliberately NOT used: its per-allocation
# lock deadlocks forked ProcessPoolExecutor workers in this many-threaded process.
_HEAP_PROBE = os.environ.get("DATASET_ITERATOR_HEAP_PROBE", "0") == "1"

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


def _read_cgroup_mem():
    """(current, peak) container memory in bytes from the cgroup (v2 then v1), or
    (None, None). This is the OOM-relevant figure: it counts shared (copy-on-write)
    pages once across the parent + all forked workers, unlike summed per-process RSS."""
    for cur_f, peak_f in (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.peak"),
                          ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
                           "/sys/fs/cgroup/memory/memory.max_usage_in_bytes")):
        try:
            with open(cur_f) as f:
                cur = int(f.read())
        except OSError:
            continue
        peak = None
        try:
            with open(peak_f) as f:
                peak = int(f.read())
        except OSError:
            pass
        return cur, peak
    return None, None


def _gc_object_count():
    """Fork-safe count of gc-tracked Python objects. Read-only (no allocator hooks),
    so unlike tracemalloc it cannot deadlock forked workers. A sudden RSS step that
    coincides with a jump here is a Python-object pile-up (a growing list/dict/etc.);
    a step with this count flat is numpy-data or native memory (TF/CUDA/driver/glibc),
    since numeric numpy arrays and native buffers are not gc-tracked."""
    return len(gc.get_objects())


def log_resources(tag=""):
    """One-line snapshot of the current process' resource footprint. Returns the
    values as a dict so callers can also act on them."""
    p = Process(os.getpid())
    rss = p.memory_info().rss / float(2 ** 30)
    heap = ""
    heap_obj = None
    if _HEAP_PROBE:
        heap_obj = _gc_object_count()
        heap = f"gc_objs={heap_obj} "
    os_threads = p.num_threads()
    py_threads = threading.active_count()
    sd = [t for t in threading.enumerate() if "-shutdown" in t.name]  # unjoined executor-shutdown threads
    cg_cur, cg_peak = _read_cgroup_mem()
    cg = ""
    if cg_cur is not None:
        cg = f"cgroup_cur={cg_cur / 2 ** 30:.2f}Gb"
        if cg_peak is not None:
            cg += f" cgroup_peak={cg_peak / 2 ** 30:.2f}Gb"  # high-water of the whole container (incl. workers)
    child_info = []  # (pid, age_s, rss_gb) -> identifies persistent children and whether they hold memory
    try:
        for c in p.children():
            try:
                with c.oneshot():
                    child_info.append((c.pid, int(time.time() - c.create_time()), c.memory_info().rss / 2 ** 30))
            except (NoSuchProcess, AccessDenied):
                pass
    except (NoSuchProcess, AccessDenied):
        pass
    try:
        fds = p.num_fds()
    except (NoSuchProcess, AccessDenied):
        fds = -1
    children_str = ",".join(f"{pid}:{age}s:{r:.1f}Gb" for pid, age, r in child_info) or "none"
    sd_str = f"={[t.ident for t in sd]}" if sd else ""  # same ident across epochs => real leak; rotating => transient
    print(f"[resources] {tag} rss={rss:.2f}Gb {cg} {heap}os_threads={os_threads} py_threads={py_threads} "
          f"unjoined_shutdown_threads={len(sd)}{sd_str} children={len(child_info)}[{children_str}] fds={fds}",
          flush=True)
    return dict(rss_gb=rss, cgroup_cur=cg_cur, cgroup_peak=cg_peak, gc_objs=heap_obj,
                os_threads=os_threads, py_threads=py_threads,
                shutdown_threads=len(sd), children=len(child_info), fds=fds)