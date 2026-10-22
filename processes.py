"""Read-only NVIDIA process snapshots and optional Linux process metadata."""
from dataclasses import dataclass
import math
import re
from typing import Optional
import xml.etree.ElementTree as ET


REMOTE_PROCESS_COMMAND = r'''
if gm_xml=$(nvidia-smi -q -x 2>/dev/null); then
    printf '__GPU_MONITOR_PROCESSES_BEGIN__\n%s\n__GPU_MONITOR_PROCESSES_END__\n' "$gm_xml"
    gm_pids=$(printf '%s\n' "$gm_xml" | sed -n 's/.*<pid>\([0-9][0-9]*\)<\/pid>.*/\1/p' | sort -u | paste -sd, -)
    if [ -n "$gm_pids" ]; then
        printf '__GPU_MONITOR_PS_BEGIN__\n'
        ps -ww -p "$gm_pids" -o pid=,user:32=,pcpu=,rss=,args= 2>/dev/null || :
        printf '__GPU_MONITOR_PS_END__\n'
        printf '__GPU_MONITOR_PMON_BEGIN__\n'
        nvidia-smi pmon -c 1 -s u 2>/dev/null || :
        printf '__GPU_MONITOR_PMON_END__\n'
    fi
fi
:
'''


@dataclass
class Process:
    pid: int
    user: str = '--'
    kind: str = 'Unknown'
    gpu_memory: Optional[float] = None
    gpu_util: Optional[float] = None
    cpu_percent: Optional[float] = None
    host_memory: Optional[float] = None
    command: str = '--'


def section(output, name):
    lines = output.splitlines()
    try:
        start = lines.index('__GPU_MONITOR_' + name + '_BEGIN__')
        end = lines.index('__GPU_MONITOR_' + name + '_END__', start + 1)
        return '\n'.join(lines[start + 1:end])
    except ValueError:
        return None


def numeric(value, limit=None):
    try:
        result = float(value)
        if math.isfinite(result) and result >= 0 and (limit is None or result <= limit):
            return result
    except (ValueError, TypeError):
        pass
    return None


def memory(value):
    match = re.fullmatch(r'\s*([0-9]+(?:\.[0-9]+)?)\s*(MiB|MB|GiB|GB|KiB|KB)?\s*', value or '')
    if not match:
        return None
    factor = {'GiB': 1024, 'GB': 1024, 'KiB': 1/1024, 'KB': 1/1024}.get(match[2], 1)
    return float(match[1]) * factor


def parse_ps(text):
    rows = {}
    for line in (text or '').splitlines():
        fields = line.strip().split(None, 4)
        if len(fields) == 5 and fields[0].isdigit():
            pid, user, cpu, rss, command = fields
            rss = numeric(rss)
            rows[int(pid)] = (user, numeric(cpu), None if rss is None else rss/1024, command)
    return rows


def parse_pmon(text):
    """Use header names: driver versions add columns such as jpg and ofa."""
    header, rows = None, {}
    for line in (text or '').splitlines():
        fields = line.split()
        if line.lstrip().startswith('#'):
            names = line.lstrip()[1:].split()
            if all(key in names for key in ('gpu', 'pid', 'sm')):
                header = names
            continue
        if header is None or len(fields) < len(header):
            continue
        row = dict(zip(header, fields))
        if row['gpu'].isdigit() and row['pid'].isdigit():
            rows[(row['gpu'], int(row['pid']))] = numeric(row['sm'], 100)
    return rows


def attach_processes(gpus, output):
    """XML is authoritative for membership; ps/pmon only enrich matching rows."""
    for gpu in gpus:
        gpu.processes = []
        gpu.process_error = 'Process snapshot unavailable.'
    xml = section(output, 'PROCESSES')
    if xml is None:
        return
    try:
        root = ET.fromstring(xml)
        if root.tag != 'nvidia_smi_log':
            raise ValueError('Not an NVIDIA snapshot')
    except (ET.ParseError, ValueError):
        for gpu in gpus:
            gpu.process_error = 'Invalid process snapshot.'
        return
    metadata = parse_ps(section(output, 'PS'))
    utilization = parse_pmon(section(output, 'PMON'))
    nodes = {node.findtext('uuid'): node for node in root.findall('gpu')}
    kinds = {'C': 'Compute', 'G': 'Graphic', 'C+G': 'C+G', 'M': 'MPS', 'M+C': 'M+C', 'O': 'Other'}
    for gpu in gpus:
        node = nodes.get(gpu.uuid)
        process_node = node.find('processes') if node is not None else None
        if process_node is None or (process_node.text or '').strip() not in ('', 'None'):
            continue
        result = {}
        try:
            for row in process_node.findall('process_info'):
                pid_text = (row.findtext('pid') or '').strip()
                if not pid_text.isdigit() or int(pid_text) <= 0:
                    raise ValueError('Invalid PID')
                process = Process(int(pid_text), kind=kinds.get((row.findtext('type') or '').strip(), 'Unknown'),
                                  gpu_memory=memory(row.findtext('used_memory')),
                                  command=row.findtext('process_name') or '--')
                if process.pid in metadata:
                    process.user, process.cpu_percent, process.host_memory, process.command = metadata[process.pid]
                process.gpu_util = utilization.get((gpu.index, process.pid))
                # Some driver versions report separate compute/graphics contexts.
                if process.pid in result:
                    previous = result[process.pid]
                    if previous.kind != process.kind:
                        process.kind = 'C+G' if {previous.kind, process.kind} == {'Compute', 'Graphic'} else 'Mixed'
                    values = [v for v in (previous.gpu_memory, process.gpu_memory) if v is not None]
                    process.gpu_memory = max(values) if values else None
                result[process.pid] = process
            gpu.processes = sorted(result.values(), key=lambda p: (-(p.gpu_memory or 0), p.pid))
            gpu.process_error = ''
        except ValueError:
            gpu.process_error = 'Invalid PID in process snapshot.'
