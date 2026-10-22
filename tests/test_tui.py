"""Exercise navigation frames and real curses startup on a small PTY."""
import fcntl
import dataclasses
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import termios
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import core
import monitor
import ui


class Screen:
    def __init__(self, keys):
        self.keys = iter(keys)
        self.frames = []
        self.lines = {}
        self.styles = {}
        self.style_frames = []

    def getmaxyx(self):
        return 24, 80

    def keypad(self, value):
        pass

    def timeout(self, value):
        pass

    def erase(self):
        self.lines = {}
        self.styles = {}

    def addnstr(self, y, x, line, width, style):
        for column in range(x, x + len(line[:width])):
            self.styles[y, column] = style
        previous = self.lines.get(y, '')
        self.lines[y] = previous[:x].ljust(x) + line[:width] + previous[x + len(line[:width]):]

    def refresh(self):
        self.frames.append('\n'.join(self.lines.values()))
        self.style_frames.append(dict(self.styles))

    def getch(self):
        return next(self.keys)


class TUITests(unittest.TestCase):
    def test_sgr_wheel_and_trackpad_scroll_in_both_directions(self):
        args = monitor.parser().parse_args(['--demo', '--no-color'])
        wheel_down = list(b'\x1b[<65;20;10M')
        wheel_up = list(b'\x1b[<64;20;10M')
        screen = Screen([10] + wheel_down * 30 + wheel_up * 30 + [ord('q')])
        with mock.patch('ui.curses.curs_set'):
            ui.run_tui(screen, None, core.demo_samples(), args)
        self.assertTrue(any('76/76' in frame for frame in screen.frames))
        self.assertIn('Lines 1-15/76', screen.frames[-1])
        self.assertNotIn('OVERVIEW', screen.frames[-1])

    def test_scrollbar_can_be_dragged_to_bottom(self):
        args = monitor.parser().parse_args(['--demo', '--no-color'])
        drag = list(b'\x1b[<0;79;7M\x1b[<32;79;21M\x1b[<0;79;21m')
        screen = Screen([10] + drag + [ord('q')])
        with mock.patch('ui.curses.curs_set'):
            ui.run_tui(screen, None, core.demo_samples(), args)
        self.assertIn('76/76', screen.frames[-1])
        self.assertIn('GPU 7 ', screen.frames[-1])

    def test_continuous_detail_scroll_reaches_every_gpu_and_clamps(self):
        sample = core.demo_samples()[0]
        sample.gpus = [dataclasses.replace(sample.gpus[0], index=i) for i in range(8)]
        for width, height in [(72, 20), (80, 24), (120, 40), (180, 60)]:
            top = ui.detail_frame(sample, width, height, 0)
            bottom = ui.detail_frame(sample, width, height, 100000)
            self.assertGreater(bottom.offset, 0)
            self.assertEqual(top.thumb_start, 0)
            self.assertEqual(bottom.thumb_start + bottom.thumb_size, bottom.track_size)
            self.assertIn('GPU 7 ', '\n'.join(bottom.lines))
            seen = set()
            for offset in range(bottom.offset + 1):
                frame = ui.detail_frame(sample, width, height, offset)
                self.assertEqual(frame.lines[:6], top.lines[:6])
                self.assertIn('q: quit', frame.lines[height - 2])
                self.assertLessEqual(max(map(len, frame.lines)), width - 1)
                for i in range(8):
                    if 'GPU {} '.format(i) in '\n'.join(frame.lines):
                        seen.add(i)
            self.assertEqual(seen, set(range(8)))
        sample.gpus = sample.gpus[:1]
        self.assertEqual(ui.detail_frame(sample, 120, 40, 100000).offset, 0)

    def test_all_frame_edges_use_same_white_attribute(self):
        args = monitor.parser().parse_args(['--demo'])
        screen = Screen([10, ui.curses.KEY_DOWN, ui.curses.KEY_END, ord('q')])
        with mock.patch('ui.curses.curs_set'), mock.patch('ui.curses.has_colors', return_value=True), \
                mock.patch('ui.curses.start_color'), mock.patch('ui.curses.use_default_colors'), \
                mock.patch('ui.curses.init_pair'), mock.patch('ui.curses.color_pair', side_effect=lambda n: n << 8):
            ui.run_tui(screen, None, core.demo_samples(), args)
        for frame, styles in zip(screen.frames, screen.style_frames):
            edges = [(y, x) for y, line in enumerate(frame.splitlines())
                     for x, char in enumerate(line) if char in '╭╮╰╯│─']
            self.assertTrue(edges)
            self.assertEqual({styles[position] for position in edges}, {7 << 8 | ui.curses.A_BOLD})
        self.assertIn('█', screen.frames[-1])

    def test_process_reflow_and_resize_keep_current_gpu_anchor(self):
        sample = core.demo_samples()[0]
        example = sample.gpus[0]
        sample.gpus = [dataclasses.replace(example, index=i,
            processes=[dataclasses.replace(example.processes[0], pid=1000+j) for j in range(8)])
            for i in range(8)]
        initial = ui.detail_frame(sample, 80, 24, 60)
        self.assertIn('GPU 4 ', initial.lines[6])
        changed = ui.detail_frame(sample, 80, 24, initial.offset, process_page=1, anchor=initial.anchor)
        self.assertIn('GPU 4 ', changed.lines[6])
        tiny = ui.detail_frame(sample, 50, 12, changed.offset, anchor=changed.anchor)
        restored = ui.detail_frame(sample, 80, 24, tiny.offset, process_page=1, anchor=tiny.anchor)
        self.assertEqual(restored.offset, changed.offset)
        for gpu in sample.gpus:
            gpu.processes = gpu.processes[:1]
        refreshed = ui.detail_frame(sample, 80, 24, initial.offset, anchor=initial.anchor)
        self.assertIn('GPU 4 ', refreshed.lines[6])

    def test_resource_meters_keep_unknown_distinct_from_idle(self):
        self.assertEqual(ui.meter(0, 10), '[..........]   0%')
        self.assertEqual(ui.meter(50, 10), '[#####.....]  50%')
        self.assertEqual(ui.meter(100, 10), '[##########] 100%')
        self.assertEqual(ui.meter(None, 10), '[??????????]  --%')

    def test_server_memory_meter_requires_complete_data(self):
        sample = core.demo_samples()[0]
        self.assertAlmostEqual(ui.memory_percent(sample.gpus),
                               sum(g.used for g in sample.gpus) / sum(g.total for g in sample.gpus) * 100)
        sample.gpus[0] = dataclasses.replace(sample.gpus[0], used=None)
        self.assertIsNone(ui.memory_percent(sample.gpus))

    def test_visual_meters_have_semantic_color_and_no_stale_values(self):
        line = 'VRAM [#########.]  90%  GPU [..........]   0%'
        spans = ui.color_spans(line)
        self.assertIn(('#########', 'critical'), [(line[a:b], role) for a,b,role in spans])
        self.assertIn(('0%', 'positive'), [(line[a:b], role) for a,b,role in spans])
        sample = core.demo_samples()[6]
        row = next(l for l in ui.overview_lines([sample], 80, 24, 0) if sample.host.alias in l)
        self.assertIn('--', row)
        self.assertNotIn('#', row)

    def test_overview_columns_align_with_summary_at_all_widths(self):
        samples = core.demo_samples()
        samples[0].gpus = [dataclasses.replace(samples[0].gpus[0], used=81920)]
        for width in range(72, 145):
            with self.subTest(width=width):
                lines = ui.overview_lines(samples, width, 24, 0)
                self.assertEqual(len(lines[2]), len(lines[8]))
                self.assertEqual(len(lines[7]), len(lines[8]))
                for row in lines[9:18]:
                    self.assertEqual(len(row), len(lines[8]))
                self.assertIn('100%', lines[9])
                self.assertTrue(lines[9].rstrip().endswith('0s'))
                self.assertTrue(all(len(line) < width for line in lines))

    def test_overview_models_preserve_variants_and_mark_mixed_hardware(self):
        sample = core.demo_samples()[0]
        sample.gpus = [dataclasses.replace(sample.gpus[0], name='NVIDIA GeForce RTX 4090', total=24576)]
        for width in (72, 80, 100, 130):
            lines = ui.overview_lines([sample], width, 24, 0)
            self.assertIn('GPU MODEL', lines[7])
            self.assertIn('RTX 4090 (24 GiB)', lines[9])
        sample.gpus.append(dataclasses.replace(sample.gpus[0], name='NVIDIA GeForce RTX 3090'))
        self.assertIn('+1', ui.overview_lines([sample], 72, 24, 0)[9])
        wide = ui.overview_lines([sample], 130, 24, 0)[9]
        self.assertIn('RTX 4090 (24 GiB) +1 type', wide)
        self.assertIn('RTX 3090 (24 GiB)', ui.gpu_model(sample.gpus, 60))
        sample.gpus = [dataclasses.replace(sample.gpus[0], name='NVIDIA A100-SXM4-80GB')]
        self.assertIn('A100-SXM4-80GB', ui.overview_lines([sample], 130, 24, 0)[9])
        sample.state = 'OFFLINE'
        self.assertNotIn('A100', ui.overview_lines([sample], 130, 24, 0)[9])

    def test_model_capacity_distinguishes_variants_and_missing_values(self):
        gpu = dataclasses.replace(core.demo_samples()[0].gpus[0], name='NVIDIA A100', total=40960)
        other = dataclasses.replace(gpu, total=81920)
        self.assertEqual(ui.gpu_model([gpu, other], 60), 'A100 (40 GiB) / A100 (80 GiB)')
        self.assertEqual(ui.gpu_model([gpu, gpu], 60), 'A100 (40 GiB)')
        self.assertEqual(ui.gpu_model([dataclasses.replace(gpu, total=None)], 60), 'A100 (-- GiB)')

    def test_navigation_overview_details_scroll_help_and_back(self):
        args = monitor.parser().parse_args(['--demo', '--no-color'])
        screen = Screen([10, ui.curses.KEY_NPAGE, ui.curses.KEY_RIGHT,
                         ord('b'), ord('?'), ord('?'), ord('q')])
        with mock.patch('ui.curses.curs_set'):
            ui.run_tui(screen, None, core.demo_samples(), args)
        self.assertIn('OVERVIEW', screen.frames[0])
        self.assertIn('GPU 0', screen.frames[1])
        self.assertIn('GPU 2', screen.frames[2])
        self.assertIn('Lines 15-', screen.frames[2])
        self.assertNotIn('GPU 0 ', screen.frames[2])
        self.assertIn('GPU MONITOR / SERVER', screen.frames[3])
        self.assertIn('ncrc_64', screen.frames[3].splitlines()[1])
        self.assertIn('OVERVIEW', screen.frames[4])
        self.assertIn('HELP', screen.frames[5])
        self.assertIn('OVERVIEW', screen.frames[6])

    def test_home_end_and_line_scroll_preserve_header_and_border_context(self):
        args = monitor.parser().parse_args(['--demo', '--no-color'])
        samples = core.demo_samples()
        screen = Screen([10, ui.curses.KEY_END, ui.curses.KEY_HOME,
                         ui.curses.KEY_DOWN, ui.curses.KEY_UP, ord('q')])
        with mock.patch('ui.curses.curs_set'):
            ui.run_tui(screen, None, samples, args)
        self.assertIn('GPU 7 ', screen.frames[2])
        self.assertEqual(screen.frames[1], screen.frames[3])
        self.assertEqual(screen.frames[1], screen.frames[5])
        self.assertIn('Lines 2-', screen.frames[4])
        sample = samples[0]
        whole = ui.detail_frame(sample, 80, 200, 0)
        bottom_row = next(i for i, line in enumerate(whole.lines[6:]) if line.startswith(' +---'))
        scrolled = ui.detail_frame(sample, 80, 24, bottom_row)
        self.assertTrue(ui.display_line(scrolled.lines[6], previous=scrolled.previous[6]).startswith(' ╰'))

    def test_process_pagination_and_command_scroll_keep_same_gpu(self):
        args = monitor.parser().parse_args(['--demo', '--no-color'])
        samples = core.demo_samples()
        example = samples[0].gpus[0].processes[0]
        samples[0].gpus[0].processes = [dataclasses.replace(example, pid=1000+i,
            command='/very/long/path/to/python train.py --epochs 400 --batch-size 32') for i in range(20)]
        screen = Screen([10, ord(']'), ord('.'), ord('q')])
        with mock.patch('ui.curses.curs_set'):
            ui.run_tui(screen, None, samples, args)
        self.assertIn('1000', screen.frames[1])
        self.assertNotIn('1000', screen.frames[2])
        self.assertIn('GPU 0', screen.frames[2])
        self.assertNotEqual(screen.frames[2], screen.frames[3])

    def test_demo_refreshes_healthy_samples(self):
        args = monitor.parser().parse_args(['--demo', '--no-color', '--interval', '5'])
        screen = Screen([-1, ord('q')])
        with mock.patch('ui.curses.curs_set'), mock.patch('ui.time.monotonic', side_effect=[0, 30, 30, 60, 60]), mock.patch('ui.demo_samples', wraps=core.demo_samples) as generate:
            ui.run_tui(screen, None, core.demo_samples(), args)
        self.assertEqual(generate.call_count, 2)
        self.assertIn('6 online', screen.frames[-1])

    def test_real_color_curses_starts_and_exits_at_four_sizes(self):
        for rows, cols in [(24, 80), (40, 120), (20, 72), (12, 50)]:
            with self.subTest(rows=rows, cols=cols):
                master, slave = os.openpty()
                process = None
                try:
                    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', rows, cols, 0, 0))
                    process = subprocess.Popen([sys.executable, str(ROOT / 'monitor.py'), '--demo'],
                                               stdin=slave, stdout=slave, stderr=slave,
                                               env=dict(os.environ, TERM='xterm-256color'))
                    os.close(slave)
                    slave = None
                    output = b''
                    expected = b'OVERVIEW' if cols >= 72 else b'Please resize'
                    deadline = time.monotonic() + 5
                    while expected not in output and time.monotonic() < deadline:
                        if select.select([master], [], [], .1)[0]:
                            output += os.read(master, 65536)
                    self.assertIn(expected, output)
                    if cols >= 72:
                        # Curses can update only the digit in a GPU heading.
                        for keys, expected in [
                            (b'\r', b'GPU 0'),
                            (b'\x1b[<65;20;10M', '4-{}'.format(rows - 6).encode()),
                            (b'\x1b[<64;20;10M', '1-{}'.format(rows - 9).encode()),
                            (b'\x1bOF', b'76/76'),
                        ]:
                            os.write(master, keys)
                            detail_output = b''
                            deadline = time.monotonic() + 3
                            while expected not in detail_output and time.monotonic() < deadline:
                                if select.select([master], [], [], .1)[0]:
                                    detail_output += os.read(master, 65536)
                            self.assertIn(expected, detail_output)
                            output += detail_output
                    os.write(master, b'q')
                    # Keep draining: macOS PTYs can block curses on final output
                    # if the test waits for exit without reading the master.
                    deadline = time.monotonic() + 3
                    while process.poll() is None and time.monotonic() < deadline:
                        if select.select([master], [], [], .1)[0]:
                            try:
                                output += os.read(master, 65536)
                            except OSError:
                                break
                    self.assertEqual(process.wait(timeout=3), 0)
                    self.assertIn(b'\x1b[?1006h', output)
                    self.assertIn(b'\x1b[?1006l', output)
                    if cols >= 72:
                        self.assertIn('╭'.encode('utf-8'), output)
                        self.assertIn('╰'.encode('utf-8'), output)
                        self.assertIn('█'.encode('utf-8'), output)
                        self.assertIn(b'\x1b[32m', output)
                        self.assertIn(b'\x1b[33m', output)
                        if rows >= 24:  # AUTH server is on page two at 20 rows.
                            self.assertIn(b'\x1b[31m', output)
                finally:
                    if process and process.poll() is None:
                        process.kill()
                        process.wait()
                    os.close(master)
                    if slave is not None:
                        os.close(slave)


if __name__ == '__main__':
    unittest.main()
