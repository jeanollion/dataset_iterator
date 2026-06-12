import os, time, subprocess, ctypes, threading, tracemalloc
from psutil import NoSuchProcess, AccessDenied, wait_procs, Process
import concurrent.futures.process

# Opt-in (DATASET_ITERATOR_TRACEMALLOC=1): track Python allocations so log_resources
# can attribute a sudden RSS step to Python (and name the line) vs native memory
# (TF/CUDA/glibc, which tracemalloc cannot see). Has allocation overhead -> diagnostic only.
_TM_ON = os.environ.get("DATASET_ITERATOR_TRACEMALLOC", "0") == "1"
if _TM_ON and not tracemalloc.is_tracing():
    tracemalloc.start(15)  # keep 15 frames so the traceback is informative
_tm_last_snapshot = None
_tm_last_rss = None

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


def _tracemalloc_report(rss_gb, jump_threshold_gb=1.5):
    """When tracemalloc is enabled and RSS jumped by >= jump_threshold_gb since the
    last call, print the top Python allocation growths and the Python-tracked total
    vs RSS. If tracked << RSS, the step is native (TF/CUDA/glibc), not Python."""
    global _tm_last_snapshot, _tm_last_rss
    if not _TM_ON:
        return
    snap = tracemalloc.take_snapshot()
    if _tm_last_snapshot is not None and _tm_last_rss is not None and (rss_gb - _tm_last_rss) >= jump_threshold_gb:
        traced_cur, traced_peak = tracemalloc.get_traced_memory()
        verdict = "mostly NATIVE (TF/CUDA/glibc) -- not a Python allocation" if traced_cur / 2 ** 30 < rss_gb * 0.5 \
            else "Python-visible -- see growths below"
        print(f"[tracemalloc] RSS step {_tm_last_rss:.2f}->{rss_gb:.2f}Gb | python-tracked={traced_cur / 2 ** 30:.2f}Gb "
              f"(peak {traced_peak / 2 ** 30:.2f}Gb) => {verdict}", flush=True)
        for stat in snap.compare_to(_tm_last_snapshot, 'traceback')[:10]:
            frame = stat.traceback.format()[-1].strip() if stat.traceback else "?"
            print(f"[tracemalloc]   {stat.size_diff / 2 ** 20:+.1f}MiB ({stat.count_diff:+d} blocks) {frame}", flush=True)
    _tm_last_snapshot = snap
    _tm_last_rss = rss_gb


def log_resources(tag=""):
    """One-line snapshot of the current process' resource footprint. Returns the
    values as a dict so callers can also act on them."""
    p = Process(os.getpid())
    rss = p.memory_info().rss / float(2 ** 30)
    _tracemalloc_report(rss)
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
    print(f"[resources] {tag} rss={rss:.2f}Gb {cg} os_threads={os_threads} py_threads={py_threads} "
          f"unjoined_shutdown_threads={len(sd)}{sd_str} children={len(child_info)}[{children_str}] fds={fds}",
          flush=True)
    return dict(rss_gb=rss, cgroup_cur=cg_cur, cgroup_peak=cg_peak, os_threads=os_threads, py_threads=py_threads,
                shutdown_threads=len(sd), children=len(child_info), fds=fds)