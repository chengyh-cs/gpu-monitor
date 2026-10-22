"""GPU process snapshots, partial failures, and bounded process tables."""
import dataclasses
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core
import ui

XML = '''<?xml version="1.0"?>
<nvidia_smi_log>
 <gpu><uuid>GPU-a</uuid><processes>
  <process_info><pid>123</pid><type>C</type><process_name>python</process_name><used_memory>4096 MiB</used_memory></process_info>
  <process_info><pid>456</pid><type>G</type><process_name>/usr/lib/Xorg</process_name><used_memory>4 MiB</used_memory></process_info>
 </processes></gpu>
 <gpu><uuid>GPU-b</uuid><processes>
  <process_info><pid>123</pid><type>C</type><process_name>python</process_name><used_memory>N/A</used_memory></process_info>
 </processes></gpu>
 <gpu><uuid>GPU-empty</uuid><processes/></gpu>
 <gpu><uuid>GPU-na</uuid><processes>N/A</processes></gpu>
</nvidia_smi_log>'''
PS = '123 alice 175.5 1048576 /env/bin/python -u train.py --name "a b"\n456 gdm 0.0 1024 /usr/lib/Xorg :0\n'
PMON = '# gpu pid type sm mem enc dec command\n# Idx # C/G % % % % name\n0 123 C 85 21 - - python\n0 456 G - - - - Xorg\n1 123 C 10 5 - - python\n'


def section(name, body):
    return '__GPU_MONITOR_' + name + '_BEGIN__\n' + body + '\n__GPU_MONITOR_' + name + '_END__\n'


def fixture():
    csv = '0, GPU-a, A100, 85, 4100, 81920, 77820, 60\n1, GPU-b, A100, 10, 4096, 81920, 77824, 40\n2, GPU-empty, A100, 0, 0, 81920, 81920, 30\n3, GPU-na, A100, N/A, 0, 81920, 81920, 30\n'
    return core.BEGIN + '\n' + csv + core.END + '\n' + section('PROCESSES', XML) + section('PS', PS) + section('PMON', PMON)


class ProcessTests(unittest.TestCase):
    def sample(self, output=None):
        runner = mock.Mock()
        runner.run.return_value = (0, fixture() if output is None else output, '')
        return core.collect(core.Host('ncrc_64', 'example', '22', 'test'), Path('/tmp/config'), 5, runner)

    def test_processes_match_gpu_uuid_and_same_pid_on_multiple_gpus(self):
        sample = self.sample()
        self.assertEqual(sample.state, 'OK')
        self.assertTrue(hasattr(sample.gpus[0], 'processes'), 'Per-GPU processes must be collected')
        a, b = sample.gpus[:2]
        self.assertEqual([p.pid for p in a.processes], [123, 456])
        process = a.processes[0]
        self.assertEqual(process.user, 'alice')
        self.assertEqual(process.kind, 'Compute')
        self.assertEqual(process.gpu_memory, 4096)
        self.assertEqual(process.gpu_util, 85)
        self.assertEqual(process.cpu_percent, 175.5)
        self.assertEqual(process.host_memory, 1024)
        self.assertIn('--name "a b"', process.command)
        self.assertEqual(a.processes[1].kind, 'Graphic')
        self.assertEqual(b.processes[0].gpu_util, 10)
        self.assertIsNone(b.processes[0].gpu_memory)

    def test_empty_and_unavailable_process_lists_are_distinct(self):
        sample = self.sample()
        self.assertEqual(sample.gpus[2].processes, [])
        self.assertEqual(sample.gpus[2].process_error, '')
        self.assertTrue(sample.gpus[3].process_error)

    def test_process_failure_does_not_discard_gpu_measurements(self):
        sample = self.sample(fixture().split('__GPU_MONITOR_PROCESSES_BEGIN__')[0])
        self.assertEqual(sample.state, 'OK')
        self.assertEqual(len(sample.gpus), 4)
        self.assertTrue(sample.gpus[0].process_error)

    def test_inaccessible_ps_and_unsupported_pmon_keep_unknown_fields(self):
        sample = self.sample(fixture().split('__GPU_MONITOR_PS_BEGIN__')[0])
        process = sample.gpus[0].processes[0]
        self.assertEqual(process.user, '--')
        self.assertIsNone(process.gpu_util)
        self.assertIsNone(process.cpu_percent)
        self.assertEqual(process.command, 'python')

    def test_optional_process_timeout_preserves_completed_gpu_sample(self):
        runner = mock.Mock()
        partial = fixture().split('__GPU_MONITOR_PROCESSES_BEGIN__')[0]
        runner.run.side_effect = subprocess.TimeoutExpired('ssh', 5, output=partial.encode())
        sample = core.collect(core.Host('ncrc_64', 'example', '22', 'test'), Path('/tmp/config'), 5, runner)
        self.assertEqual(sample.state, 'OK')
        self.assertEqual(len(sample.gpus), 4)
        self.assertIn('timed out', sample.gpus[0].process_error.lower())

    def test_real_remote_shell_collects_only_gpu_pids_and_preserves_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            xmlfile = base / 'snapshot.xml'
            xmlfile.write_text(XML)
            nvidia = base / 'nvidia-smi'
            nvidia.write_text('#!/bin/sh\ncase "$1" in\n-q) cat "' + str(xmlfile) + '";;\npmon) printf "# gpu pid type sm mem enc dec command\\n0 123 C 85 21 - - python\\n";;\n*) printf "0, GPU-a, A100, 85, 4100, 81920, 77820, 60\\n";;\nesac\n')
            ps = base / 'ps'
            ps.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "' + str(base / 'ps.args') + '"\nprintf "123 alice 175.5 1048576 /env/bin/python -u train.py --epochs 400\\n"\n')
            nvidia.chmod(0o700)
            ps.chmod(0o700)
            result = subprocess.run(['/bin/sh', '-c', core.REMOTE_COMMAND], env=dict(os.environ, PATH=tmp + ':/usr/bin:/bin'), capture_output=True, text=True, check=True)
            sample = self.sample(result.stdout)
            process = sample.gpus[0].processes[0]
            self.assertEqual(process.user, 'alice')
            self.assertEqual(process.gpu_util, 85)
            self.assertIn('--epochs 400', process.command)
            self.assertIn('123,456', (base / 'ps.args').read_text())

    def test_malformed_process_xml_never_looks_like_an_empty_gpu(self):
        result = self.sample(fixture().replace('<nvidia_smi_log>', '<broken>'))
        self.assertEqual(result.state, 'OK')
        self.assertTrue(result.gpus[0].process_error)

    def test_pmon_extended_header_does_not_shift_sm_utilization(self):
        result = self.sample(fixture().replace(PMON, '# gpu pid type sm mem enc dec jpg ofa command\n0 123 C 37 20 - - - - python\n'))
        self.assertEqual(result.gpus[0].processes[0].gpu_util, 37)

    def test_unicode_decoration_does_not_rewrite_process_commands(self):
        line = ' | 123 alice Compute 10% 4096M sh -c "echo [###...] 50% | cat" |'
        visual = ui.display_line(line, previous=' | PID USER TYPE GPU% VRAM COMMAND |')
        self.assertIn('echo [###...] 50% | cat', visual)

    def test_details_show_processes_below_gpu_and_paginate_without_overflow(self):
        sample = self.sample()
        for width, height in [(72, 20), (80, 24), (120, 40)]:
            text = '\n'.join(ui.detail_lines(sample, width, height, 0))
            self.assertIn('PID', text)
            self.assertIn('alice', text)
            self.assertLess(text.index('VRAM'), text.index('PID'))
        process = sample.gpus[0].processes[0]
        sample.gpus[0].processes = [dataclasses.replace(process, pid=1000+i) for i in range(20)]
        first = '\n'.join(ui.detail_lines(sample, 80, 24, 0, process_page=0))
        second = '\n'.join(ui.detail_lines(sample, 80, 24, 0, process_page=1))
        self.assertIn('1000', first)
        self.assertNotIn('1000', second)
        self.assertNotEqual(first, second)
        self.assertIn('quit', second.splitlines()[-1])
        self.assertLessEqual(len(second.splitlines()), 24)
        self.assertTrue(all(len(line)<80 for line in second.splitlines()))


if __name__ == '__main__':
    unittest.main()
