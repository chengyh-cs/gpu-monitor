import curses
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terminal_input import MouseEvent, TerminalInput


class InputTests(unittest.TestCase):
    def reader(self, keys):
        screen = mock.Mock()
        screen.getch.side_effect = keys
        return TerminalInput(screen)

    def test_sgr_modifiers_fragmented_report_and_horizontal_wheel(self):
        reader = self.reader(list(b'\x1b[<69;300;10') + [-1] + list(b'M') + list(b'\x1b[<66;20;10M'))
        self.assertEqual(reader.read(), MouseEvent('down', 299, 9))
        self.assertEqual(reader.read(), -1)

    def test_plain_escape_does_not_swallow_next_key(self):
        reader = self.reader([27, ord('q')])
        self.assertEqual(reader.read(), 27)
        self.assertEqual(reader.read(), ord('q'))
        self.assertEqual(self.reader([27, -1]).read(), 27)

    def test_escape_delay_is_reduced_and_restored_even_on_interrupt(self):
        with mock.patch('terminal_input.curses.get_escdelay', return_value=1000), \
                mock.patch('terminal_input.curses.set_escdelay') as delay, \
                mock.patch('terminal_input.curses.mousemask', return_value=(1, 0)), \
                mock.patch('terminal_input.sys.stdout.isatty', return_value=False):
            with self.assertRaises(KeyboardInterrupt):
                with self.reader([]):
                    delay.assert_called_once_with(25)
                    raise KeyboardInterrupt
            self.assertEqual(delay.call_args_list, [mock.call(25), mock.call(1000)])

    def test_modern_curses_mouse_and_unavailable_event(self):
        reader = self.reader([curses.KEY_MOUSE] * 3)
        with mock.patch('terminal_input.curses.BUTTON5_PRESSED', 1 << 25, create=True), \
                mock.patch('terminal_input.curses.getmouse', side_effect=[
                    (0, 3, 9, 0, 1 << 25), (0, 3, 9, 0, curses.BUTTON4_PRESSED), curses.error()]):
            self.assertEqual(reader.read(), MouseEvent('down', 3, 9))
            self.assertEqual(reader.read(), MouseEvent('up', 3, 9))
            self.assertEqual(reader.read(), -1)

    def test_capture_is_disabled_after_interrupt(self):
        stream = io.StringIO()
        with mock.patch('terminal_input.sys.stdout', stream), \
                mock.patch.object(stream, 'isatty', return_value=True), \
                mock.patch('terminal_input.curses.mousemask', return_value=(1, 0)) as mask:
            with self.assertRaises(KeyboardInterrupt):
                with self.reader([]):
                    raise KeyboardInterrupt
        self.assertIn('\x1b[?1006h', stream.getvalue())
        self.assertTrue(stream.getvalue().endswith('\x1b[?1006l\x1b[?1002l\x1b[?1000l'))
        self.assertEqual(mask.call_args, mock.call(0))
