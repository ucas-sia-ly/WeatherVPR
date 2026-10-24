#!/usr/bin/env python3
"""Run one experiment in a memory-limited systemd user scope (Linux/cgroup v2).

Example: python scripts/run_bounded.py --log runs/job.memory.jsonl -- python scripts/train.py ...
Only this script's workload is terminated when available system RAM is low.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

GIB = 1024 ** 3
ROOT = Path(__file__).resolve().parents[1]


def available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Linux MemAvailable is required for the RAM guard')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--max-gib', type=float, default=8.)
    parser.add_argument('--high-gib', type=float, default=6.)
    parser.add_argument('--reserve-gib', type=float, default=4.)
    parser.add_argument('--inside-scope', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 0 < args.high_gib <= args.max_gib <= 8 or not 0 < args.reserve_gib:
        parser.error('Provide a command, 0 < high-gib <= max-gib <= 8, and positive reserve-gib')
    max_bytes, high_bytes = int(args.max_gib * GIB), int(args.high_gib * GIB)
    reserve = int(args.reserve_gib * GIB)
    if not args.inside_scope:
        if available_bytes() < max_bytes + reserve:
            raise SystemExit('Insufficient available RAM to start safely; need max-gib + reserve-gib')
        unit = 'weathervpr-' + uuid.uuid4().hex[:10]
        wrapped = ['systemd-run', '--user', '--scope', '--unit=' + unit,
                   '-p', f'MemoryHigh={high_bytes}', '-p', f'MemoryMax={max_bytes}',
                   '-p', 'MemorySwapMax=0',
                   sys.executable, str(Path(__file__).resolve()), '--inside-scope',
                   '--log', str(args.log.resolve()), '--max-gib', str(args.max_gib),
                   '--high-gib', str(args.high_gib), '--reserve-gib', str(args.reserve_gib),
                   '--', *command]
        return subprocess.call(wrapped)

    # Hold the lock inside the scope: it survives a disconnected outer tool or
    # shell while the actual workload is still running.
    lock_path = ROOT / 'runs/.experiment_memory.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another bounded experiment is running; run workloads sequentially')

    membership = Path('/proc/self/cgroup').read_text().strip().splitlines()
    relative = next((line[3:] for line in membership if line.startswith('0::')), None)
    if relative is None:
        raise RuntimeError('A cgroup v2 memory-limited scope is required')
    cgroup = Path('/sys/fs/cgroup') / relative.lstrip('/')
    limit = (cgroup / 'memory.max').read_text().strip()
    if limit == 'max' or int(limit) > max_bytes:
        raise RuntimeError('MemoryMax is not active; refusing to launch workload')
    if (cgroup / 'memory.swap.max').read_text().strip() != '0':
        raise RuntimeError('MemorySwapMax=0 is required')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    peak = 0
    minimum_available = available_bytes()
    started = time.monotonic()
    # The workload retains the same lock if this monitor is interrupted or
    # selected as a cgroup OOM victim. A second job must not slip in meanwhile.
    child = subprocess.Popen(command, start_new_session=True, pass_fds=(lock.fileno(),))

    def stop_child(signum=signal.SIGTERM):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    pending_signal = None

    def interrupted(signum, frame):
        nonlocal pending_signal
        pending_signal = signum
        stop_child(signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    reason = None
    try:
        with args.log.open('w', buffering=1) as log:
            log.write(json.dumps({'event': 'start', 'cgroup': str(cgroup), 'max_bytes': max_bytes,
                                  'high_bytes': high_bytes, 'reserve_bytes': reserve,
                                  'command': command}) + '\n')
            while True:
                current = int((cgroup / 'memory.current').read_text())
                available = available_bytes()
                peak = max(peak, current)
                minimum_available = min(minimum_available, available)
                log.write(json.dumps({'event': 'sample', 'elapsed_seconds': time.monotonic() - started,
                                      'cgroup_bytes': current, 'system_available_bytes': available}) + '\n')
                if pending_signal is not None:
                    reason = f'signal_{pending_signal}'
                    break
                if child.poll() is not None:
                    break
                if available < reserve:
                    reason = 'system_available_below_reserve'
                    break
                time.sleep(.5)
            if reason:
                stop_child()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    stop_child(signal.SIGKILL)
            code = child.wait()
            events = dict(line.split() for line in (cgroup / 'memory.events').read_text().splitlines())
            summary = {'event': 'finish', 'returncode': code, 'termination_reason': reason,
                       'peak_cgroup_bytes_sampled': peak,
                       'peak_cgroup_bytes_kernel': int((cgroup / 'memory.peak').read_text()),
                       'minimum_system_available_bytes': minimum_available, 'memory_events': events,
                       'elapsed_seconds': time.monotonic() - started}
            log.write(json.dumps(summary) + '\n')
            print('Memory guard:', json.dumps(summary), flush=True)
    finally:
        # Also clear descendants if the immediate command has already exited.
        stop_child()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            stop_child(signal.SIGKILL)
            child.wait()
        stop_child(signal.SIGKILL)
    return (128 + pending_signal if pending_signal is not None else
            75 if reason else code if code >= 0 else 128 - code)


if __name__ == '__main__':
    sys.exit(main())
