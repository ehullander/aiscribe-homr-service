"""Opt-in numeric CPU diagnostics; no detector changes or logs. AGPL-3.0-or-later."""
from functools import lru_cache
import os
from pathlib import Path
from time import perf_counter, process_time


def read(path):
    try:
        return path.read_text().strip()
    except (OSError, UnicodeError):
        return None


def numbers(path):
    text = read(path)
    if text is None:
        return {}
    result = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                result[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return result


@lru_cache(maxsize=1)
def cpu_model():
    text = read(Path('/proc/cpuinfo'))
    for line in (text or '').splitlines():
        if line.lower().startswith('model name'):
            return line.partition(':')[2].strip()[:160]
    return None


class CPUProbe:
    def __init__(self, root=Path('/sys/fs/cgroup'), *, wall=perf_counter, cpu=process_time):
        self.root = Path(root)
        self.wall, self.cpu = wall, cpu
        self.v2 = (self.root/'cpu.stat').exists()
        self.folder = self.root
        if not self.v2:
            for name in ('cpu,cpuacct', 'cpu'):
                if (self.root/name/'cpu.stat').exists():
                    self.folder = self.root/name
                    break

    def stats(self):
        values = numbers(self.folder/'cpu.stat')
        local = numbers(self.folder/'cpu.stat.local')
        usage = values.get('usage_usec')
        if usage is None:
            raw = read(self.folder/'cpuacct.usage')
            try:
                usage = int(raw)/1000 if raw is not None else None
            except ValueError:
                usage = None
        throttled = values.get('throttled_usec')
        if throttled is None and 'throttled_time' in values:
            throttled = values['throttled_time']/1000
        return dict(usage_usec=usage, throttled_usec=throttled,
                    local_throttled_usec=local.get('throttled_usec'),
                    nr_periods=values.get('nr_periods'), nr_throttled=values.get('nr_throttled'))

    def quota(self):
        raw = read(self.root/'cpu.max')
        try:
            if raw is not None:
                quota, period = raw.split()
                return None if quota == 'max' else int(quota)/int(period)
            quota = read(self.folder/'cpu.cfs_quota_us')
            period = read(self.folder/'cpu.cfs_period_us')
            return int(quota)/int(period) if quota is not None and int(quota) > 0 else None
        except (ValueError, TypeError, ZeroDivisionError):
            return None

    def start(self):
        self.before = self.stats()
        self.before_quota = self.quota()
        self.started_wall, self.started_cpu = self.wall(), self.cpu()
        return self

    def finish(self):
        ended_cpu, ended_wall = self.cpu(), self.wall()
        after = self.stats()
        delta = {key:(after[key]-value if value is not None and after[key] is not None
                      and after[key] >= value else None) for key,value in self.before.items()}
        wall = max(0, ended_wall-self.started_wall)
        cpu = max(0, ended_cpu-self.started_cpu)
        try:
            affinity = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            affinity = None
        try:
            import cv2
            cv_threads = cv2.getNumThreads()
        except (ImportError, AttributeError):
            cv_threads = None
        def seconds(key):
            return delta[key]/1e6 if delta[key] is not None else None
        quota = self.quota()
        raw_quota = read(self.root/'cpu.max') or read(self.folder/'cpu.cfs_quota_us') or ''
        mode = 'limited' if quota is not None else ('unlimited' if raw_quota.split()[:1] in (['max'], ['-1']) else 'unknown')
        return dict(wall_seconds=wall, process_cpu_seconds=cpu,
                    process_cpu_to_wall=cpu/wall if wall else None,
                    cgroup_usage_seconds=seconds('usage_usec'),
                    cgroup_throttled_seconds=seconds('throttled_usec'),
                    cgroup_local_throttled_seconds=seconds('local_throttled_usec'),
                    cgroup_periods=delta['nr_periods'], cgroup_throttled_periods=delta['nr_throttled'],
                    cpu_quota_cores=quota, cpu_quota_cores_before=self.before_quota,
                    cpu_quota_mode=mode, os_cpu_count=os.cpu_count(),
                    affinity_cpu_count=affinity, opencv_threads=cv_threads, cpu_model=cpu_model(),
                    cgroup_stats_available=any(value is not None for value in self.before.values()),
                    cgroup_scope='all processes in current cgroup; ancestor limits may be invisible')
