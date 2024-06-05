import os, time, subprocess
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
