import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
READY = importlib.util.find_spec('core') is not None
if READY:
    import core
    import ui


class ImplementationTests(unittest.TestCase):
    def test_monitor_modules_exist(self):
        self.assertTrue(READY, 'GPU monitor implementation is not present yet')


@unittest.skipUnless(READY, 'Implementation not present yet')
class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.host = core.Host('ncrc_60', '192.0.2.10', '10010', 'example_user')
        self.csv = '0, GPU-abc, NVIDIA A100, 0, 1024, 40960, 39936, 35\n'

    def test_inventory_reads_includes_and_ignores_wildcards(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'extra').write_text('Host ncrc_64 ncrc_64_codex\nHost * !bad\n')
            config = base / 'config'
            config.write_text('Host ncrc_60_codex ncrc_60\nInclude "' + str(base / 'extra') + '"\nHost github.com orb slai slai_lf\n')
            self.assertEqual(core.default_aliases(config), ['ncrc_60', 'ncrc_60_codex', 'ncrc_64', 'ncrc_64_codex', 'slai', 'slai_lf'])

    def test_endpoint_dedup_prefers_standard_and_preserves_ports(self):
        hosts = [core.Host('ncrc_167_codex', 'same', '22167', 'u'), core.Host('ncrc_169', 'same', '22169', 'u'), core.Host('ncrc_167', 'same', '22167', 'u')]
        self.assertEqual([h.alias for h in core.deduplicate(hosts)], ['ncrc_167', 'ncrc_169'])

    def test_discovery_timeout_terminates_match_exec_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / 'helper.pid'
            config = Path(tmp) / 'config'
            config.write_text('Match exec "echo $$ > ' + str(pidfile) + '; exec sleep 30"\n  User test\n')
            try:
                host = core.resolve_host('example', config, timeout=.2)
                self.assertTrue(host.error)
                self.assertTrue(pidfile.exists())
                pid = int(pidfile.read_text())
                alive = True
                deadline = time.monotonic() + .5
                while alive and time.monotonic() < deadline:
                    try:
                        os.kill(pid, 0)
                        time.sleep(.01)
                    except ProcessLookupError:
                        alive = False
                self.assertFalse(alive, 'Match exec helper still alive after discovery timeout')
            finally:
                if pidfile.exists():
                    try:
                        os.kill(int(pidfile.read_text()), 9)
                    except ProcessLookupError:
                        pass

    def test_gpu_csv_unknown_is_not_zero_or_available(self):
        gpu = core.parse_gpus(self.csv)[0]
        self.assertTrue(core.candidate(gpu, 8, 10))
        unknown = core.parse_gpus('0, GPU-x, A100, [N/A], 0, 40960, 40960, N/A\n')[0]
        self.assertIsNone(unknown.util)
        self.assertFalse(core.candidate(unknown, 8, 10))
        self.assertFalse(core.candidate(gpu, 40, 10))

    def test_malformed_gpu_data_is_rejected(self):
        for value in ['garbage', '0, id, A100, 0, -1, 40960, 40960, 35', '0, id, A100, nan, 0, 40960, 40960, 35']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                core.parse_gpus(value)

    def test_ssh_overrides_are_effective_in_real_openssh(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config'
            config.write_text('Host example\n HostName 127.0.0.1\n User test\n RequestTTY force\n RemoteCommand exec bash -l\n RemoteForward 34567 127.0.0.1:7890\n ControlMaster auto\n ControlPath ~/.ssh/test-control\n')
            cmd = core.ssh_base(config, 5)
            r = subprocess.run(cmd + ['-G', 'example'], capture_output=True, text=True, check=True)
            values = dict(l.split(' ', 1) for l in r.stdout.splitlines() if ' ' in l)
            self.assertEqual(values['requesttty'], 'false')
            self.assertEqual(values['clearallforwardings'], 'yes')
            self.assertEqual(values['batchmode'], 'yes')
            self.assertNotIn('remoteforward', values)
            self.assertNotIn('remotecommand', values)
            self.assertNotIn('controlpath', values)

    def test_failed_and_stale_samples_have_no_candidates(self):
        gpu = core.parse_gpus(self.csv)[0]
        sample = core.Snapshot(self.host, 'OK', [gpu], sampled_at=100)
        self.assertEqual(core.sample_state(sample, 101, 15), 'OK')
        self.assertEqual(core.sample_state(sample, 120, 15), 'STALE')
        view = ui.overview_lines([sample], 80, 24, 0, now=120, stale_after=15)
        self.assertIn('STALE', '\n'.join(view))
        self.assertNotIn('39.0G', '\n'.join(view))

    def test_collect_success_and_missing_tool(self):
        runner = mock.Mock()
        runner.run.return_value = (0, core.BEGIN + '\n' + self.csv + core.END + '\n', '')
        sample = core.collect(self.host, Path('/tmp/config'), 5, runner)
        self.assertEqual(sample.state, 'OK')
        self.assertEqual(len(sample.gpus), 1)
        runner.run.return_value = (127, '', '__GPU_MONITOR_MISSING__\n')
        sample = core.collect(self.host, Path('/tmp/config'), 5, runner)
        self.assertEqual(sample.state, 'NO TOOL')
        self.assertEqual(sample.gpus, [])

    def test_auth_and_unknown_host_are_distinct(self):
        runner = mock.Mock()
        for message, state in [('Permission denied (publickey).', 'AUTH'), ('Host key verification failed.', 'HOST KEY'), ('Connection refused', 'OFFLINE')]:
            runner.run.return_value = (255, '', message)
            self.assertEqual(core.collect(self.host, Path('/tmp/config'), 5, runner).state, state)

    def test_local_network_denial_is_not_reported_as_server_offline(self):
        runner = mock.Mock()
        runner.run.return_value = (255, '', 'ssh: connect to host 192.0.2.10 port 10010: Operation not permitted')
        self.assertEqual(core.collect(self.host, Path('/tmp/config'), 5, runner).state, 'BLOCKED')

    def test_long_error_hint_remains_readable(self):
        sample = core.Snapshot(self.host, 'HOST KEY', sampled_at=time.time())
        rendered = ' '.join(ui.detail_lines(sample, 72, 20, 0))
        self.assertIn('trusting its key.', rendered)

    def test_remote_shell_protocol_with_fake_nvidia_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'nvidia-smi'
            binary.write_text('#!/bin/sh\nprintf "0, GPU-test, NVIDIA A100, 0, 1024, 40960, 39936, 35\\n"\n')
            binary.chmod(0o700)
            env = dict(os.environ, PATH=tmp)
            result = subprocess.run(['/bin/sh', '-c', core.REMOTE_COMMAND], env=env, capture_output=True, text=True, check=True)
            runner = mock.Mock()
            runner.run.return_value = (result.returncode, result.stdout, result.stderr)
            sample = core.collect(self.host, Path('/tmp/config'), 5, runner)
            self.assertEqual(sample.state, 'OK')
            self.assertEqual(sample.gpus[0].uuid, 'GPU-test')
            binary.unlink()
            missing = subprocess.run(['/bin/sh', '-c', core.REMOTE_COMMAND], env=env, capture_output=True, text=True)
            self.assertEqual(missing.returncode, 127)
            self.assertIn('__GPU_MONITOR_MISSING__', missing.stderr)

    def test_close_interrupts_an_active_request(self):
        runner = core.CommandRunner()
        worker = threading.Thread(target=lambda: runner.run([sys.executable, '-c', 'import time; time.sleep(30)'], 30))
        worker.start()
        deadline = time.monotonic() + 2
        while not runner.active and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(runner.active)
        runner.close()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertFalse(runner.active)

    def test_runner_timeout_and_close_kill_children(self):
        runner = core.CommandRunner()
        start = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired):
            runner.run([sys.executable, '-c', 'import time; time.sleep(30)'], .15)
        self.assertLess(time.monotonic() - start, 2)
        self.assertFalse(runner.active)
        runner.close()
        with self.assertRaises(core.Stopped):
            runner.run([sys.executable, '-c', 'pass'], 1)

    def test_polling_fast_host_not_blocked_by_slow_host(self):
        slow = core.Host('ncrc_64', 'other', '22', 'u')
        def fake(host, *args):
            if host.alias == 'ncrc_64':
                time.sleep(.3)
            return core.Snapshot(host, 'OK', [], sampled_at=time.time())
        with mock.patch('core.collect', side_effect=fake):
            poller = core.Poller([self.host, slow], Path('/tmp/config'), 5, 1)
            try:
                poller.tick()
                deadline = time.monotonic() + .2
                while poller.snapshots[0].state == 'WAIT' and time.monotonic() < deadline:
                    time.sleep(.01)
                    poller.tick()
                self.assertEqual(poller.snapshots[0].state, 'OK')
                self.assertEqual(poller.snapshots[1].state, 'WAIT')
            finally:
                poller.close()

    def test_overview_and_details_are_bounded_and_paginated(self):
        samples = core.demo_samples()
        for width, height in [(72, 20), (80, 24), (120, 40), (50, 12)]:
            for lines in [ui.overview_lines(samples, width, height, 0), ui.detail_lines(samples[0], width, height, 0)]:
                self.assertLessEqual(len(lines), height)
                self.assertTrue(all(len(line) < width for line in lines))
                self.assertIn('quit', lines[-1])
        sample = samples[0]
        sample.gpus = [dataclasses.replace(sample.gpus[0], index=str(n)) for n in range(8)]
        first = '\n'.join(ui.detail_lines(sample, 80, 24, 0))
        second = '\n'.join(ui.detail_lines(sample, 80, 24, 1))
        self.assertIn('GPU 0', first)
        self.assertNotIn('GPU 7', first)
        self.assertNotEqual(first, second)

    def test_cli_demo_json_and_bad_arguments(self):
        r = subprocess.run([sys.executable, str(ROOT / 'monitor.py'), '--demo', '--json'], capture_output=True, text=True, timeout=5)
        self.assertEqual(r.returncode, 0, r.stderr)
        result = json.loads(r.stdout)
        self.assertTrue(result['demo'])
        self.assertEqual(len(result['servers']), 9)
        bad = subprocess.run([sys.executable, str(ROOT / 'monitor.py'), '--interval', '0'], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)


if __name__ == '__main__':
    unittest.main()
