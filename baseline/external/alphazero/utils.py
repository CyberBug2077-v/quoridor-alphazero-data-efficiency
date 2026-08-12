import os
import psutil
import torch


class dotdict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _read_cgroup_mem_limit_bytes():
    """Return cgroup memory limit in bytes."""
    paths = [
        "/sys/fs/cgroup/memory.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    ]
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if raw in ("", "max"):
                continue
            limit = int(raw)
            # Some systems report a very large number when unlimited.
            if limit > 1 << 60:
                return None
            return limit
        except (OSError, ValueError):
            continue
    return None


def log_ram_usage(label=None, include_peak=False):
    """Log compact CPU/RAM/GPU usage for this process."""
    sys_mem = psutil.virtual_memory()
    sys_used_gb = sys_mem.used / (1024 * 1024 * 1024)
    sys_total_gb = sys_mem.total / (1024 * 1024 * 1024)
    proc = psutil.Process(os.getpid())
    proc_rss_gb = proc.memory_info().rss / (1024 * 1024 * 1024)
    proc_cpu_pct = proc.cpu_percent(interval=0.1)
    mem_limit_bytes = _read_cgroup_mem_limit_bytes()
    mem_limit_gb = mem_limit_bytes / (1024 * 1024 * 1024) if mem_limit_bytes else None

    label_str = f" {label}" if label else ""
    if mem_limit_gb:
        ram_proc = f"{proc_rss_gb:.1f}G/{mem_limit_gb:.1f}G"
    else:
        ram_proc = f"{proc_rss_gb:.1f}G"
    msg = (f"[Resources]{label_str} CPU(proc): {proc_cpu_pct:.0f}% | "
           f"RAM(proc): {ram_proc} | RAM(node): {sys_used_gb:.1f}G/{sys_total_gb:.1f}G")

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        alloc_gb = torch.cuda.memory_allocated(device) / (1024 * 1024 * 1024)
        reserved_gb = torch.cuda.memory_reserved(device) / (1024 * 1024 * 1024)
        msg += f" | GPU alloc/res: {alloc_gb:.1f}G/{reserved_gb:.1f}G"
        if include_peak:
            peak_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024 * 1024 * 1024)
            msg += f" | GPU peak alloc: {peak_alloc_gb:.1f}G"
    else:
        msg += " | GPU: N/A"

    print(msg)


def format_duration(seconds):
    """Format seconds as H:MM:SS."""
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"
