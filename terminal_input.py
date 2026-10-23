"""Curses mouse events with SGR fallback for macOS's older mouse ABI.

Protocol: https://invisible-island.net/xterm/ctlseqs/ctlseqs.html
"""
from collections import deque
import curses
from dataclasses import dataclass
import re
import sys
import time


@dataclass
class MouseEvent:
    kind: str
    x: int
    y: int


class TerminalInput:
    def __init__(self, screen):
        self.screen = screen
        self.pending = deque()
        self.old_mask = None
        self.old_escdelay = None
        self.enabled = False

    def __enter__(self):
        # Curses normally waits 1000 ms to distinguish Esc from a key sequence.
        # This local TUI needs only a short window; preserve faster user settings.
        try:
            self.old_escdelay = curses.get_escdelay()
            curses.set_escdelay(min(self.old_escdelay, 25))
        except (AttributeError, curses.error):
            self.old_escdelay = None
        try:
            _, self.old_mask = curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)
        except curses.error:
            pass
        # Explicit SGR avoids the missing BUTTON5 definition on older macOS
        # curses. Modern curses decodes it; older builds use read()'s fallback.
        if sys.stdout.isatty():
            self.enabled = True
            sys.stdout.write('\x1b[?1000h\x1b[?1002h\x1b[?1006h')
            sys.stdout.flush()
        return self

    def __exit__(self, *exc):
        try:
            if self.enabled:
                sys.stdout.write('\x1b[?1006l\x1b[?1002l\x1b[?1000l')
                sys.stdout.flush()
        finally:
            if self.old_escdelay is not None:
                curses.set_escdelay(self.old_escdelay)
            if self.old_mask is not None:
                try:
                    curses.mousemask(self.old_mask)
                except curses.error:
                    pass

    def read(self):
        key = self.pending.popleft() if self.pending else self.screen.getch()
        if key == curses.KEY_MOUSE:
            try:
                _, x, y, _, state = curses.getmouse()
            except curses.error:
                return -1
            for name, kind in [('BUTTON4_PRESSED', 'up'), ('BUTTON5_PRESSED', 'down'),
                               ('BUTTON1_RELEASED', 'release'), ('BUTTON1_CLICKED', 'click'),
                               ('BUTTON1_PRESSED', 'press')]:
                if state & getattr(curses, name, 0):
                    return MouseEvent(kind, x, y)
            if state & curses.REPORT_MOUSE_POSITION:
                return MouseEvent('drag', x, y)
            return -1
        if key != 27:
            return key
        # Consume a whole report, so its ESC never becomes "back" and its
        # digits/brackets never activate unrelated keyboard shortcuts.
        prefix = []
        for expected in (ord('['), ord('<')):
            if not prefix:
                # getch already waited to resolve Esc. Check queued bytes
                # without adding the normal 100 ms polling timeout again.
                self.screen.timeout(0)
                try:
                    char = self.screen.getch()
                finally:
                    self.screen.timeout(100)
            else:
                char = self.screen.getch()
            if char != -1:
                prefix.append(char)
            if char != expected:
                self.pending.extend(prefix)
                return 27
        report = ''
        deadline = time.monotonic() + .25
        while len(report) < 48 and time.monotonic() < deadline:
            char = self.screen.getch()
            if char == -1:
                continue
            if not 0 <= char <= 127:
                self.pending.append(char)
                return -1
            report += chr(char)
            if chr(char) in 'Mm':
                match = re.fullmatch(r'(\d+);(\d+);(\d+)([Mm])', report)
                if not match:
                    return -1
                button, x, y = map(int, match.group(1, 2, 3))
                button &= ~28  # Ignore Shift/Alt/Ctrl modifiers.
                kind = ('release' if match[4] == 'm' else
                        {64: 'up', 65: 'down', 0: 'press', 32: 'drag'}.get(button))
                return MouseEvent(kind, x - 1, y - 1) if kind else -1
            if chr(char) not in '0123456789;':
                return -1
        return -1
