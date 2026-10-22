"""SSH discovery, read-only GPU collection, and independent polling."""
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import glob
import io
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import threading
import time
from typing import List, Optional
from processes import Process, REMOTE_PROCESS_COMMAND, attach_processes

BEGIN = '__GPU_MONITOR_BEGIN__'
END = '__GPU_MONITOR_END__'
REMOTE_COMMAND = (
    'LC_ALL=C; export LC_ALL; '
    'if ! command -v nvidia-smi >/dev/null 2>&1; then '
    "printf '__GPU_MONITOR_MISSING__\\n' >&2; exit 127; fi; "
    "printf '" + BEGIN + "\\n'; "
    'nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,'
    'memory.total,memory.free,temperature.gpu --format=csv,noheader,nounits '
    "&& printf '" + END + "\\n' || exit $?; " + REMOTE_PROCESS_COMMAND
)


@dataclass
class Host:
    alias: str
    hostname: str
    port: str
    user: str
    route: str = ''
    error: str = ''


@dataclass
class GPU:
    index: str
    uuid: str
    name: str
    util: Optional[float]
    used: Optional[float]
    total: Optional[float]
    free: Optional[float]
    temperature: Optional[float]
    processes: List[Process] = field(default_factory=list)
    process_error: str = 'Process data not collected.'


@dataclass
class Snapshot:
    host: Host
    state: str = 'WAIT'
    gpus: List[GPU] = field(default_factory=list)
    sampled_at: float = 0
    error: str = ''
    duration: float = 0


def alias_key(alias):
    match = re.fullmatch(r'ncrc_(\d+)(_codex)?', alias)
    return (0, int(match[1]), bool(match[2])) if match else (1, alias, False)


def literal_aliases(config, stack=None):
    """Inventory literal Host tokens; OpenSSH remains the authority on options."""
    config = Path(config).expanduser().resolve()
    stack = set() if stack is None else stack
    if config in stack:
        return []
    stack = stack | {config}
    found = []
    for line in config.read_text().splitlines():
        match = re.match(r'^\s*([A-Za-z]+)\s*(?:=\s*)?(.*)$', line)
        if not match or match[1].lower() not in ('host', 'include'):
            continue
        tokens = shlex.split(match[2], comments=True)
        if match[1].lower() == 'host':
            found.extend(t for t in tokens if re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', t))
        else:
            for token in tokens:
                token = os.path.expanduser(token.replace('%d', str(Path.home())))
                if '%' in token:
                    continue
                if not os.path.isabs(token):
                    token = str(Path.home() / '.ssh' / token)
                for path in sorted(glob.glob(token)):
                    found.extend(literal_aliases(path, stack))
    return list(dict.fromkeys(found))


def default_aliases(config):
    return sorted((a for a in literal_aliases(config)
                   if re.fullmatch(r'ncrc_\d+(_codex)?|slai|slai_lf', a)), key=alias_key)


def ssh_base(config, timeout):
    cmd = ['ssh', '-F', str(Path(config).expanduser()), '-n', '-T']
    for option in [
        'BatchMode=yes', 'ClearAllForwardings=yes', 'RemoteCommand=none',
        'RequestTTY=no', 'ControlMaster=no', 'ControlPath=none', 'ControlPersist=no',
        'ForwardAgent=no', 'ForwardX11=no', 'PermitLocalCommand=no',
        'StrictHostKeyChecking=yes', 'ConnectionAttempts=1',
        'ConnectTimeout=' + str(max(1, math.ceil(timeout))),
        'ServerAliveInterval=5', 'ServerAliveCountMax=1', 'LogLevel=ERROR',
    ]:
        cmd += ['-o', option]
    return cmd


def resolve_host(alias, config, timeout=5):
    if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', alias):
        raise ValueError('Invalid SSH alias: ' + alias)
    runner = CommandRunner()
    try:
        code, out, err = runner.run(ssh_base(config, timeout) + ['-G', alias], timeout)
        if code:
            raise ValueError(err.strip() or 'ssh -G failed')
        values = dict(line.split(' ', 1) for line in out.splitlines() if ' ' in line)
        return Host(alias, values['hostname'], values['port'], values['user'],
                    values.get('proxyjump', '') + '|' + values.get('proxycommand', ''))
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        return Host(alias, '?', '?', '?', error=str(exc))
    finally:
        runner.close()


def deduplicate(hosts):
    result, seen = [], set()
    for host in sorted(hosts, key=lambda h: (h.alias.endswith('_codex'), alias_key(h.alias))):
        key = (host.hostname, host.port, host.user, host.route) if not host.error else (host.alias,)
        if key not in seen:
            result.append(host)
            seen.add(key)
    return sorted(result, key=lambda h: alias_key(h.alias))


def discover(config, aliases=None):
    aliases = aliases or default_aliases(config)
    return deduplicate([resolve_host(alias, config) for alias in aliases])


def number(value, maximum=None):
    value = value.strip()
    if value.upper() in ('N/A', '[N/A]', '[NOT SUPPORTED]', 'NOT SUPPORTED', '-'):
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0 or (maximum is not None and result > maximum):
        raise ValueError('Invalid GPU measurement: ' + value)
    return result


def parse_gpus(text):
    gpus, indices = [], set()
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row or not any(x.strip() for x in row):
            continue
        if len(row) != 8:
            raise ValueError('Expected 8 GPU fields; received ' + str(len(row)))
        index, uuid, name = (x.strip() for x in row[:3])
        if not index.isdigit() or index in indices:
            raise ValueError('Invalid or duplicate GPU index')
        indices.add(index)
        gpu = GPU(index, uuid, name, number(row[3], 100), *(number(v) for v in row[4:]))
        if gpu.total is not None:
            if any(v is not None and v > gpu.total + 1 for v in [gpu.used, gpu.free]):
                raise ValueError('GPU memory exceeds total memory')
        gpus.append(gpu)
    return gpus


def candidate(gpu, min_free=8, max_util=10):
    return (gpu.free is not None and gpu.util is not None
            and gpu.free >= min_free * 1024 and gpu.util <= max_util)


def sample_state(sample, now, stale_after):
    if sample.state == 'OK' and now - sample.sampled_at > stale_after:
        return 'STALE'
    return sample.state


class Stopped(Exception):
    pass


class CommandRunner:
    """Own subprocesses so a timeout or UI exit also terminates proxy children."""
    def __init__(self):
        self.active = set()
        self.lock = threading.Lock()
        self.closed = False

    @staticmethod
    def kill(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def run(self, command, timeout):
        with self.lock:
            if self.closed:
                raise Stopped()
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       stdin=subprocess.DEVNULL, text=True, errors='replace',
                                       start_new_session=True)
            self.active.add(process)
        try:
            try:
                out, err = process.communicate(timeout=timeout)
            except BaseException:
                self.kill(process)
                process.communicate()
                raise
            return process.returncode, out, err
        finally:
            with self.lock:
                self.active.discard(process)

    def close(self):
        with self.lock:
            self.closed = True
            for process in list(self.active):
                self.kill(process)


def collect(host, config, timeout, runner):
    start = time.monotonic()
    sampled_at = time.time()
    sample = Snapshot(host)
    try:
        if host.error:
            sample.state, sample.error = 'CONFIG', host.error
            return sample
        code, out, err = runner.run(ssh_base(config, timeout) + [host.alias, REMOTE_COMMAND], timeout)
        message = (err.strip() or out.strip())[-2000:]
        if code:
            if '__GPU_MONITOR_MISSING__' in message:
                sample.state, sample.error = 'NO TOOL', 'nvidia-smi is not available in the remote PATH.'
            elif 'Host key verification failed' in message or 'REMOTE HOST IDENTIFICATION HAS CHANGED' in message:
                sample.state, sample.error = 'HOST KEY', message
            elif re.search(r'connect to host .*: (Operation not permitted|Permission denied)', message):
                sample.state, sample.error = 'BLOCKED', message
            elif 'Permission denied' in message or 'sign_and_send_pubkey' in message:
                sample.state, sample.error = 'AUTH', message
            elif code == 255:
                sample.state, sample.error = 'OFFLINE', message
            elif 'No devices were found' in message:
                sample.state, sample.error = 'NO GPU', message
            else:
                sample.state, sample.error = 'GPU ERROR', message
        else:
            lines = out.splitlines()
            try:
                begin = lines.index(BEGIN)
                end = lines.index(END, begin + 1)
            except ValueError:
                raise ValueError('Incomplete GPU response (missing sample markers)')
            body = '\n'.join(lines[begin + 1:end])
            sample.gpus = [] if body.strip() == 'No devices were found' else parse_gpus(body)
            attach_processes(sample.gpus, out)
            sample.state = 'OK' if sample.gpus else 'NO GPU'
    except subprocess.TimeoutExpired as exc:
        sample.state, sample.error = 'TIMEOUT', 'SSH / GPU query exceeded {:.0f}s.'.format(timeout)
        # The GPU section precedes optional process queries. Keep a complete GPU
        # sample if only those later queries timed out, with unknown process data.
        partial = exc.output or ''
        if isinstance(partial, bytes):
            partial = partial.decode('utf-8', errors='replace')
        try:
            lines = partial.splitlines()
            begin = lines.index(BEGIN)
            end = lines.index(END, begin + 1)
            body = '\n'.join(lines[begin + 1:end])
            sample.gpus = [] if body.strip() == 'No devices were found' else parse_gpus(body)
            attach_processes(sample.gpus, partial)
            for gpu in sample.gpus:
                if gpu.process_error:
                    gpu.process_error = 'Process query timed out; GPU metrics are available.'
            sample.state, sample.error = ('OK' if sample.gpus else 'NO GPU'), ''
        except ValueError:
            pass
    except ValueError as exc:
        sample.state, sample.error = 'BAD DATA', str(exc)
    except (OSError, Stopped) as exc:
        sample.state, sample.error = 'OFFLINE', str(exc) or 'Collector stopped.'
    finally:
        sample.sampled_at = sampled_at
        sample.duration = time.monotonic() - start
    return sample


class Poller:
    def __init__(self, hosts, config, interval, timeout):
        self.hosts, self.config, self.interval, self.timeout = hosts, config, interval, timeout
        self.runner = CommandRunner()
        self.pool = ThreadPoolExecutor(max_workers=min(16, max(1, len(hosts))))
        self.snapshots = [Snapshot(h) for h in hosts]
        self.futures = {}
        self.due = [0.0] * len(hosts)
        self.stale_after = max(15, interval * 2 + timeout)

    def tick(self):
        now = time.monotonic()
        for index, future in list(self.futures.items()):
            if future.done():
                self.snapshots[index] = future.result()
                del self.futures[index]
        for index, host in enumerate(self.hosts):
            if index not in self.futures and now >= self.due[index]:
                self.futures[index] = self.pool.submit(collect, host, self.config, self.timeout, self.runner)
                self.due[index] = now + self.interval

    def refresh(self):
        self.due = [0.0] * len(self.hosts)

    def close(self):
        self.runner.close()
        self.pool.shutdown(wait=True, cancel_futures=True)


def demo_samples():
    samples = []
    for i, alias in enumerate(['ncrc_60', 'ncrc_64', 'ncrc_66', 'ncrc_88', 'ncrc_111', 'ncrc_167', 'ncrc_169', 'slai', 'slai_lf']):
        host = Host(alias, 'demo.invalid', '22', 'demo')
        gpus = [GPU(str(j), 'DEMO-GPU-{}-{}'.format(i, j), 'NVIDIA A100 80GB PCIe',
                    0 if (i+j) % 3 == 0 else 76, 1024 if (i+j) % 3 == 0 else 66560,
                    81920, 80896 if (i+j) % 3 == 0 else 15360, 34 if (i+j) % 3 == 0 else 68)
                for j in range(8 if i == 0 else 4)]
        for j, gpu in enumerate(gpus):
            gpu.process_error = ''
            gpu.processes = [Process(1600, 'gdm', 'Graphic', 4, 0, 0.0, 130, '/usr/lib/Xorg :0')]
            if gpu.util:
                gpu.processes.insert(0, Process(48000 + i*10 + j, 'labuser', 'Compute', 66556,
                                               76, 98.5, 6528, '/opt/conda/bin/python -u train.py --epochs 400 --batch_size 32'))
        samples.append(Snapshot(host, 'OK', gpus, time.time()))
    samples[3] = Snapshot(samples[3].host, 'TIMEOUT', sampled_at=time.time(), error='Demo: connection timed out.')
    samples[6].sampled_at -= 60
    samples[8] = Snapshot(samples[8].host, 'AUTH', sampled_at=time.time(), error='Demo: Permission denied (publickey).')
    return samples
