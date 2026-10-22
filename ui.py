"""Readable overview and per-host pages; render functions are terminal-independent."""
import curses
from dataclasses import dataclass
import math
import re
import sys
import textwrap
import time

from core import candidate, demo_samples, sample_state
from terminal_input import MouseEvent, TerminalInput


def clean(value):
    return ''.join(c if 32 <= ord(c) <= 126 else ' ' for c in str(value))


def fit(lines, width, height):
    return [clean(line)[:max(0, width - 1)] for line in lines[:height]]


def small(width, height):
    if width < 72 or height < 20:
        return fit(['GPU MONITOR', '', 'Please resize the terminal to at least 72 x 20.',
                    'Current: {} x {}'.format(width, height), '', 'q: quit'], width, height)
    return None


def gib(value):
    return '--' if value is None else '{:.1f}G'.format(value / 1024)


def meter(percent, width=10):
    """Fixed-width ASCII bars also work without color or Unicode support."""
    if percent is None:
        return '[' + '?' * width + ']  --%'
    percent = max(0, min(100, percent))
    filled = round(percent * width / 100)
    return '[{}{}] {:>3.0f}%'.format('#' * filled, '.' * (width - filled), percent)


def memory_percent(gpus):
    if not gpus or any(g.used is None or g.total is None or g.total <= 0 for g in gpus):
        return None
    return sum(g.used for g in gpus) / sum(g.total for g in gpus) * 100


def gpu_model(gpus, width):
    variants = list(dict.fromkeys((re.sub(r'^(?:NVIDIA\s+)?(?:GeForce\s+)?', '', clean(g.name)).strip()
                                  or 'Unknown', g.total) for g in gpus))
    if not variants:
        return '--'

    def capacity(total):
        value = '--' if total is None else '{:.1f}'.format(total / 1024).removesuffix('.0')
        return ' ({} GiB)'.format(value)

    joined = ' / '.join(name + capacity(total) for name, total in variants)
    if len(joined) <= width:
        return joined
    name, total = variants[0]
    memory = capacity(total)
    suffix = ' +{} type{}'.format(len(variants) - 1, 's' if len(variants) > 2 else '') if len(variants) > 1 else ''
    if len(memory) + len(suffix) + 8 > width and suffix:
        suffix = ' +{}'.format(len(variants) - 1)
    room = width - len(memory) - len(suffix)
    first = name if len(name) <= room else name[:max(0, room - 3)] + '...'
    return first + memory + suffix


def load_role(percent):
    return 'critical' if percent >= 85 else 'warning' if percent >= 60 else 'positive'


def color_spans(line):
    """Semantic token ranges; terminal-specific styling stays in the renderer."""
    spans = []
    roles = {
        'OK': 'positive', 'CANDIDATE': 'positive', 'BUSY': 'warning',
        'STALE': 'warning', 'UNKNOWN': 'warning', 'CHECKING': 'accent',
        'TIMEOUT': 'warning', 'OFFLINE': 'warning', 'BLOCKED': 'warning',
        'AUTH': 'critical', 'HOST KEY': 'critical', 'GPU ERROR': 'critical',
        'BAD DATA': 'critical', 'CONFIG': 'critical', 'NO TOOL': 'warning',
        'NO GPU': 'muted', 'DEMO - SIMULATED': 'special',
        'Compute': 'special', 'Graphic': 'warning', 'MPS': 'special',
    }
    for match in re.finditer(r'\b(?:' + '|'.join(map(re.escape, roles)) + r')\b', line):
        spans.append((match.start(), match.end(), roles[match[0]]))
    for match in re.finditer(r'\[([#. ?]+)\]\s+(\d+|--)%', line):
        spans.append((match.start(), match.end(), 'muted'))
        if match[2] != '--':
            role = load_role(float(match[2]))
            filled = match[1].count('#')
            if filled:
                spans.append((match.start(1), match.start(1) + filled, role))
            spans.append((match.start(2), match.end(), role))
    for match in re.finditer(r'\b\d+ (?:online|candidate GPUs)\b', line):
        spans.append((match.start(), match.end(), 'positive'))
    for match in re.finditer(r'\bTemp: (?:\d+(?:\.\d+)? C|--)\b', line):
        spans.append((match.start(), match.end(), 'special'))
    for match in re.finditer(r'(?<!\w)(?:Enter|Esc|PgUp/PgDn|Up/Down|Left/Right|r|b|q|\?):(?= )', line):
        spans.append((match.start(), match.end() - 1, 'accent'))
    return spans


def age(sample, now):
    return '--' if not sample.sampled_at else '{}s'.format(max(0, int(now - sample.sampled_at)))


def overview_capacity(height):
    return max(1, height - 14)


def process_capacity(height):
    return max(3, min(10, height - 17))


def summary_cards(values, width):
    available = min(width - 3, 110)
    cell = (available - 4) // 3
    widths = [cell, cell, available - 4 - 2 * cell]
    border = '  '.join('+' + '-' * (w - 2) + '+' for w in widths)
    middle = '  '.join('|' + value.center(w - 2) + '|' for value, w in zip(values, widths))
    return [' ' + border, ' ' + middle, ' ' + border]


def card(title, badge, contents, width):
    size = min(width - 3, 160)
    suffix = ' [' + badge + '] '
    room = size - len(suffix) - 6
    title = clean(title)[:max(0, room)]
    top = '+- ' + title + ' '
    top += '-' * max(0, size - len(top) - len(suffix) - 1) + suffix + '+'
    body = [' | ' + clean(line)[:size - 4].ljust(size - 4) + ' |' for line in contents]
    return [' ' + top] + body + [' +' + '-' * (size - 2) + '+']


def process_lines(gpu, width, rows, page=0, command_offset=0):
    size = min(width - 3, 160) - 4
    if gpu.process_error:
        return ['Processes: unavailable', clean(gpu.process_error)[:size]]
    if not gpu.processes:
        return ['Processes: 0', 'No visible GPU processes.']
    pages = max(1, math.ceil(len(gpu.processes) / rows))
    page = min(max(0, page), pages - 1)
    wide = size >= 100
    fmt = '{:>7} {:<10.10} {:<7.7} {:>4} {:>7} '
    header = fmt.format('PID', 'USER', 'TYPE', 'GPU%', 'VRAM')
    if wide:
        header += '{:>6} {:>7} '.format('CPU%', 'RAM')
    lines = ['Processes: {}  Page {}/{}  [/]: page  ,/.: cmd +{}'.format(len(gpu.processes), page + 1, pages, command_offset),
             header + 'COMMAND']
    for process in gpu.processes[page*rows:(page+1)*rows]:
        util = '--' if process.gpu_util is None else '{:.0f}%'.format(process.gpu_util)
        mem = '--' if process.gpu_memory is None else '{:.0f}M'.format(process.gpu_memory)
        prefix = fmt.format(process.pid, clean(process.user), process.kind, util, mem)
        if wide:
            cpu = '--' if process.cpu_percent is None else '{:.1f}'.format(process.cpu_percent)
            prefix += '{:>6} {:>7} '.format(cpu, gib(process.host_memory))
        room = max(1, size - len(prefix))
        command = clean(process.command)[command_offset:]
        if len(command) > room:
            command = command[:max(0, room - 3)] + '...'
        lines.append(prefix + command)
    return lines


def display_line(line, unicode=True, previous=''):
    """Only decorate generated frames/meters; never reinterpret server text."""
    if not unicode:
        return line
    chars = list(line)
    # Every replacement occupies one terminal column, keeping color spans aligned.
    if line.startswith(' +'):
        bottom = previous.startswith(' |')
        for match in re.finditer(r'(?<![\w-])-+(?![\w-])', line):
            for i in range(match.start(), match.end()):
                chars[i] = '─'
        for i, char in enumerate(line):
            if char == '+':
                left = i + 1 < len(line) and line[i + 1] == '-'
                chars[i] = ('╰' if left else '╯') if bottom else ('╭' if left else '╮')
    elif line.startswith(' |'):
        if re.fullmatch(r' (?:\+-+\+  ){2}\+-+\+', previous):
            positions = [i for i, char in enumerate(line) if char == '|']
        else:
            positions = [1, len(line) - 1]
        for i in positions:
            if chars[i] == '|':
                chars[i] = '│'
    if line.lstrip().startswith('---'):
        chars = ['─' if char == '-' else char for char in chars]
    prefix = line.partition('[')[0].split()
    if prefix and prefix[0] == '>':
        prefix = prefix[1:]
    overview_meter = (len(prefix) >= 4 and prefix[1] in ('OK', 'OK*')
                      and re.fullmatch(r'\d+/\d+', prefix[-1]))
    is_meter_row = line.startswith((' | VRAM ', ' | GPU  ')) or overview_meter
    for match in re.finditer(r'\[([#. ?]+)\]\s+(\d+|--)%', line if is_meter_row else ''):
        for i in range(match.start(1), match.end(1)):
            if chars[i] == '#':
                chars[i] = '█'
            elif chars[i] == '.':
                chars[i] = '░'
    return ''.join(chars)


def border_spans(line, previous=''):
    """Frame edges only: leave titles, badges and process command text alone."""
    if line.startswith(' +'):
        spans = [(m.start(), m.end()) for m in re.finditer(r'(?<![\w-])-+(?![\w-])', line)]
        spans += [(i, i + 1) for i, char in enumerate(line) if char == '+']
        return spans
    if line.startswith(' |'):
        positions = ([i for i, char in enumerate(line) if char == '|']
                     if re.fullmatch(r' (?:\+-+\+  ){2}\+-+\+', previous)
                     else [1, len(line) - 1])
        return [(i, i + 1) for i in positions]
    if line.lstrip().startswith('---'):
        return [(len(line) - len(line.lstrip()), len(line))]
    return []


def footer(lines, height, text):
    lines += [''] * max(0, height - 2 - len(lines))
    return lines + [text]


def overview_lines(samples, width, height, selected, now=None, stale_after=20,
                   min_free=8, max_util=10, interval=60, demo=False, pending=()):
    narrow = small(width, height)
    if narrow is not None:
        return narrow
    now = time.time() if now is None else now
    states = [sample_state(s, now, stale_after) for s in samples]
    healthy = sum(s == 'OK' for s in states)
    available = sum(candidate(g, min_free, max_util) for s, state in zip(samples, states)
                    if state == 'OK' for g in s.gpus)
    page_size = overview_capacity(height)
    page = selected // page_size
    size = min(width - 3, 110)
    compact = size < 90
    alias_width = 10 if compact else 12
    model_width = min(28, size - 51) if compact else min(32, size - 65)
    ready_width, free_width = (7, 6) if compact else (9, 8)
    memory_width = size - (2 + alias_width + 10 + model_width + ready_width + free_width + 5 + 6)
    bar_width = memory_width - 7
    row_format = '{} {:<' + str(alias_width) + '.' + str(alias_width) + '} {:<10.10} {:<' + str(model_width) + '} '
    row_format += ' '.join(['{:>' + str(ready_width) + '}', '{:<' + str(memory_width) + '}',
                            '{:>' + str(free_width) + '}', '{:>5}'])
    heading = row_format.format(' ', 'SERVER', 'STATE', 'GPU MODEL', 'RDY/GPU' if compact else 'READY/GPU',
                                'VRAM USED' if memory_width >= 13 else 'VRAM', 'FREE' if compact else 'MAX FREE', 'AGE')
    mode = '[DEMO - SIMULATED]' if demo else '[LIVE]'
    clock = time.strftime('%H:%M:%S', time.localtime(now))
    title = 'GPU MONITOR  /  OVERVIEW'
    refresh = 'Refresh: {:g}s   Independent sampling'.format(interval)
    lines = [' ' + title.ljust(size - len(mode)) + mode,
             ' ' + refresh.ljust(size - len(clock)) + clock]
    lines += summary_cards(['{} servers'.format(len(samples)), '{} online'.format(healthy),
                            '{} candidate GPUs'.format(available)], width)
    lines += [' Filter: free >= {:g} GiB AND GPU util <= {:g}%'.format(min_free, max_util), '',
              ' ' + heading, ' ' + '-' * size]
    for index in range(page * page_size, min(len(samples), (page + 1) * page_size)):
        s, state = samples[index], states[index]
        valid = state == 'OK'
        count = str(len(s.gpus)) if valid else '--'
        ready = str(sum(candidate(g, min_free, max_util) for g in s.gpus)) if valid else '--'
        free = [g.free for g in s.gpus if g.free is not None] if valid else []
        label = ('CHECKING' if state == 'WAIT' else state) + ('*' if index in pending and state != 'WAIT' else '')
        percent = memory_percent(s.gpus) if valid else None
        if percent is None:
            memory = '--  no data' if memory_width >= 11 else '--'
        else:
            memory = meter(percent, bar_width) if memory_width >= 13 else '{:.0f}%'.format(percent)
        lines.append(' ' + row_format.format(
            '>' if index == selected else ' ', s.host.alias, label,
            gpu_model(s.gpus, model_width) if valid else '--', ready + '/' + count,
            memory, gib(max(free)) if free else '--', age(s, now)))
    lines += [' ' + '-' * size,
              ' Page {}/{}   {}-{} of {} servers   * refreshing'.format(
                  page + 1, max(1, math.ceil(len(samples) / page_size)),
                  page * page_size + 1 if samples else 0,
                  min(len(samples), (page + 1) * page_size), len(samples))]
    return fit(footer(lines, height, 'Up/Down: select Enter: open PgUp/PgDn: page r: refresh ?: help q: quit'), width, height)


HINTS = {
    'AUTH': 'Authenticate once with this alias and unlock/load its SSH key in your agent.',
    'HOST KEY': 'Connect manually and verify the server fingerprint before trusting its key.',
    'OFFLINE': 'Check the laboratory network/VPN, server address and SSH port.',
    'BLOCKED': 'The local environment denied this connection. Server availability is unknown.',
    'TIMEOUT': 'Check network/VPN and server load; increase --timeout if necessary.',
    'NO TOOL': 'Check the NVIDIA driver and whether nvidia-smi is in the remote PATH.',
    'NO GPU': 'No NVIDIA GPU is visible to this login.',
    'GPU ERROR': 'Inspect nvidia-smi manually on this server for driver/device errors.',
    'BAD DATA': 'The query returned incomplete or invalid GPU data; retry or inspect nvidia-smi.',
    'CONFIG': 'Fix the SSH configuration error below, then restart the monitor.',
    'STALE': 'This sample is expired. Candidate availability is unknown until refreshed.',
    'WAIT': 'Waiting for the first SSH sample; other servers continue independently.',
}


@dataclass
class DetailFrame:
    lines: list
    previous: list
    offset: int = 0
    track_size: int = 0
    thumb_start: int = 0
    thumb_size: int = 0
    anchor: tuple = None
    max_scroll: int = 0


def detail_header(sample, width, now, state, min_free, max_util, demo):
    """Six fixed rows: identity, connection, telemetry, then the resource rule."""
    size = min(width - 3, 160)

    def paired(left, right):
        room = size - len(right) - 2
        left = clean(left)
        if len(left) > room:
            left = left[:max(0, room - 3)] + '...'
        return ' ' + left.ljust(size - len(right)) + right

    sampled = (time.strftime('%H:%M:%S', time.localtime(sample.sampled_at))
               if sample.sampled_at else '--')
    state = 'CHECKING' if state == 'WAIT' else state
    query = '{:.1f}s'.format(sample.duration) if sample.sampled_at else '--'
    return [
        paired('GPU MONITOR / SERVER', '[DEMO - SIMULATED]' if demo else '[LIVE]'),
        paired(sample.host.alias, '[{}]'.format(state)),
        ' SSH      {}@{}:{}'.format(sample.host.user, sample.host.hostname, sample.host.port),
        ' SAMPLED  {:<10}   AGE  {:<6}   QUERY  {}'.format(sampled, age(sample, now), query),
        ' FILTER   free >= {:g} GiB  |  util <= {:g}%  (not reserved)'.format(min_free, max_util),
        ' ' + '-' * size,
    ]


def header_spans(row, line):
    """Separate quiet labels from readable values without dimming entire rows."""
    if row == 0:
        start = line.rfind('[')
        return [(start, len(line), 'special' if '[DEMO' in line else 'accent')]
    labels = {2: ('SSH',), 3: ('SAMPLED', 'AGE', 'QUERY'), 4: ('FILTER',)}
    return [(match.start(), match.end(), 'muted')
            for label in labels.get(row, ())
            for match in [re.search(r'\b' + label + r'\b', line)] if match]


def detail_frame(sample, width, height, offset, now=None, stale_after=20,
                 min_free=8, max_util=10, demo=False, process_page=0, command_offset=0,
                 anchor=None, **unused):
    narrow = small(width, height)
    if narrow is not None:
        return DetailFrame(narrow, [''] + narrow[:-1], offset=max(0, offset), anchor=anchor)
    now = time.time() if now is None else now
    state = sample_state(sample, now, stale_after)
    header = detail_header(sample, width, now, state, min_free, max_util, demo)
    lines = []
    cards = []
    content_width = width - 2  # Keep a gap between card edges and the scroll rail.
    if state != 'OK':
        lines += textwrap.wrap('[{}] {}'.format(state, HINTS.get(state, 'No current GPU data.')), width - 4) + ['']
        if sample.error:
            lines.extend(textwrap.wrap(clean(sample.error), width - 4))
        if state == 'STALE':
            lines.append('Previous GPU readings are hidden to avoid showing outdated availability.')
    else:
        for gpu in sample.gpus:
            start = len(lines)
            label = 'CANDIDATE' if candidate(gpu, min_free, max_util) else (
                'UNKNOWN' if gpu.free is None or gpu.util is None else 'BUSY')
            temp = '--' if gpu.temperature is None else '{:g} C'.format(gpu.temperature)
            bar_width = 20 if width >= 100 else 12
            lines += card('GPU {}  |  {}'.format(gpu.index, gpu.name[:36]), label, [
                'VRAM {}  {} / {} used  Free: {}'.format(meter(memory_percent([gpu]), bar_width), gib(gpu.used), gib(gpu.total), gib(gpu.free)),
                'GPU  {}  Temp: {}'.format(meter(gpu.util, bar_width), temp),
                'UUID: ' + gpu.uuid] + process_lines(gpu, content_width, process_capacity(height), process_page, command_offset), content_width) + ['']
            cards.append(((gpu.uuid, gpu.index), start, len(lines) - start))
    if lines and not lines[-1]:
        lines.pop()
    capacity = height - len(header) - 3
    total = len(lines)
    maximum = max(0, total - capacity)
    if anchor and offset:
        for identity, start, length in cards:
            if identity == anchor[0]:
                offset = start + min(anchor[1], length - 1)
                break
    offset = max(0, min(offset, maximum))
    current_anchor = next(((identity, offset - start) for identity, start, length in cards
                           if start <= offset < start + length), None)
    body = lines[offset:offset + capacity]
    previous = [''] + header[:-1] + [lines[i - 1] if i else '' for i in range(offset, offset + len(body))]
    padding = capacity - len(body)
    visible = header + body + [''] * padding
    visible += [' Lines {}-{}/{}  Up/Down: scroll  Home/End: top/bottom'.format(
        min(offset + 1, total), min(offset + capacity, total), total),
        'b: back PgUp/Dn: scroll [/]: procs ,/.: cmd ?: help q: quit']
    thumb_size = max(1, round(capacity * min(capacity, total) / max(1, total)))
    thumb_start = round((capacity - thumb_size) * offset / maximum) if maximum else 0
    return DetailFrame(fit(visible, width, height), previous + [''] * (padding + 2),
                       offset, capacity, thumb_start, thumb_size, current_anchor, maximum)


def detail_lines(sample, width, height, offset, **options):
    return detail_frame(sample, width, height, offset, **options).lines


def help_lines(width, height):
    return fit(footer([
        ' GPU MONITOR  /  HELP', '',
        'Overview: one row per server. Enter opens just that server.',
        'Details: wheel / two-finger scroll; drag the scrollbar at right.',
        '[ / ]: process page    , / .: scroll full command horizontally',
        'Up / Down, j / k     Select server / scroll detail by one line',
        'Enter               Open selected server',
        'Left / Right, h / l  Switch server in detail view',
        'PgUp/PgDn: scroll by screen   Home/End: detail top/bottom',
        'b / Esc             Back to overview',
        'r                   Request a fresh sample from every server',
        '?                   Toggle this help screen',
        'q                   Quit and stop owned SSH connections', '',
        'CANDIDATE: enough free memory AND low current utilization.',
        'This does not reserve a GPU or establish that no job owns it.',
        'Bars: used %; green <60%, yellow 60-84%, red >=85%.',
        'Unknown [?] / STALE / errors never count as available GPUs.',
        'GPU%: pmon SM%; CPU%: ps lifetime avg; VRAM M=MiB, RAM G=GiB.',
    ], height, '?: close help  b/Esc: overview  q: quit'), width, height)


def run_tui(screen, poller, samples, args):
    with TerminalInput(screen) as inputs:
        return _run_tui(screen, poller, samples, args, inputs)


def _run_tui(screen, poller, samples, args, inputs):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.keypad(True)
    screen.timeout(100)
    colors = {}
    if not args.no_color and curses.has_colors():
        try:
            curses.start_color()
            curses.use_default_colors()
            palette = [('accent', curses.COLOR_CYAN), ('positive', curses.COLOR_GREEN),
                       ('warning', curses.COLOR_YELLOW), ('critical', curses.COLOR_RED),
                       ('special', curses.COLOR_MAGENTA)]
            for i, (role, color) in enumerate(palette, 1):
                curses.init_pair(i, color, -1)
                colors[role] = curses.color_pair(i) | curses.A_BOLD
            colors['muted'] = curses.A_DIM
            curses.init_pair(7, curses.COLOR_WHITE, -1)
            colors['border'] = curses.color_pair(7) | curses.A_BOLD
        except curses.error:
            colors = {}
    selected, scroll, view, show_help = 0, 0, 'overview', False
    process_page, command_offset = 0, 0
    unicode = not args.ascii
    try:
        '─█'.encode(sys.stdout.encoding or 'ascii')
    except (UnicodeError, LookupError):
        unicode = False
    last_frame = None
    last_detail = None
    dragging = None
    demo_due = time.monotonic() + args.interval
    while True:
        if poller:
            poller.tick()
            samples = poller.snapshots
        elif args.demo and time.monotonic() >= demo_due:
            samples = demo_samples()
            demo_due = time.monotonic() + args.interval
        height, width = screen.getmaxyx()
        selected = max(0, min(selected, len(samples) - 1))
        opts = dict(min_free=args.min_free_gib, max_util=args.max_util, demo=args.demo,
                    stale_after=max(15, args.interval * 2 + args.timeout))
        detail = None
        if show_help:
            lines = help_lines(width, height)
        elif view == 'overview':
            lines = overview_lines(samples, width, height, selected, interval=args.interval,
                                   pending=poller.futures if poller else (), **opts)
        else:
            visible = samples[selected].gpus
            max_process_page = max((math.ceil(len(g.processes)/process_capacity(height)) - 1 for g in visible), default=0)
            process_page = max(0, min(process_page, max_process_page))
            max_command_offset = max((len(clean(p.command)) - 1 for g in visible for p in g.processes), default=0)
            command_offset = max(0, min(command_offset, max_command_offset))
            anchor = last_detail.anchor if last_detail and scroll == last_detail.offset else None
            detail = detail_frame(samples[selected], width, height, scroll, process_page=process_page,
                                 command_offset=command_offset, anchor=anchor, **opts)
            lines, scroll = detail.lines, detail.offset
            last_detail = detail
        previous_lines = detail.previous if detail else [''] + lines[:-1]
        frame = (lines, width, height, detail)
        if frame != last_frame:
            screen.erase()
            for y, line in enumerate(lines):
                style = curses.A_NORMAL
                stripped = line.lstrip()
                is_detail_header = detail and detail.track_size and y < 6
                is_title = y == 0 and not is_detail_header
                if is_title:
                    style = colors.get('accent', curses.A_BOLD)
                elif is_detail_header and y == 0:
                    style = colors.get('accent', curses.A_BOLD)
                elif is_detail_header and y == 1:
                    style = colors.get('border', curses.A_BOLD)
                elif is_detail_header and y == 4:
                    style = colors.get('muted', curses.A_NORMAL)
                elif re.match(r'^ \+- GPU \d', line) or 'SERVER' in line and 'GPU MODEL' in line:
                    style = curses.A_BOLD
                elif 'PID' in line and 'COMMAND' in line:
                    style = curses.A_BOLD | colors.get('accent', 0)
                elif stripped.startswith(('UUID:', '| UUID:', 'Sample:', 'Refresh:', 'Filter:', 'Page ', 'GPU page ', '---', '+---')):
                    style = curses.A_DIM if colors else curses.A_NORMAL
                previous = previous_lines[y]
                visual = display_line(line, unicode, previous)
                try:
                    screen.addnstr(y, 0, visual, max(0, width - 1), style)
                    if colors and not is_title:
                        spans = color_spans(line)
                        if is_detail_header:
                            spans += header_spans(y, line)
                        for start, end, role in spans:
                            token_style = colors[role]
                            if is_detail_header and y == 1:
                                token_style |= curses.A_REVERSE
                            screen.addnstr(y, start, visual[start:end], end - start, token_style)
                    if colors and view == 'overview' and not show_help and 9 <= y < 9 + overview_capacity(height):
                        for match in re.finditer(r'(?<!\S)(\d+)%', line):
                            screen.addnstr(y, match.start(), visual[match.start():match.end()],
                                           match.end() - match.start(), colors[load_role(float(match[1]))])
                        if '--  no data' in line:
                            start = line.index('--  no data')
                            screen.addnstr(y, start, visual[start:start + 11], 11, colors['muted'])
                        for match in re.finditer(r'\bOFFLINE\*?', line):
                            screen.addnstr(y, match.start(), visual[match.start():match.end()],
                                           match.end() - match.start(), colors['muted'])
                    # A colored pointer and alias preserve focus without a solid block.
                    if stripped.startswith('>'):
                        match = re.search(r'>\s+\S+', line)
                        if match:
                            start, end = match.span()
                            screen.addnstr(y, start, visual[start:end], end - start,
                                           colors.get('accent', curses.A_BOLD))
                    if not is_title:
                        for start, end in border_spans(line, previous):
                            screen.addnstr(y, start, visual[start:end], end - start,
                                           colors.get('border', curses.A_BOLD))
                except curses.error:
                    pass
            if detail and detail.track_size:
                for row in range(detail.track_size):
                    thumb = detail.thumb_start <= row < detail.thumb_start + detail.thumb_size
                    glyph = ('█' if thumb else '│') if unicode else ('#' if thumb else '|')
                    try:
                        screen.addnstr(6 + row, width - 2, glyph, 1, colors.get('border', curses.A_BOLD))
                    except curses.error:
                        pass
            screen.refresh()
            last_frame = frame
        key = inputs.read()
        if isinstance(key, MouseEvent):
            if key.kind == 'release':
                dragging = None
            elif not show_help and 0 <= key.x < width and 0 <= key.y < height:
                if key.kind in ('up', 'down'):
                    delta = -1 if key.kind == 'up' else 1
                    if view == 'detail':
                        scroll = max(0, scroll + delta * 3)
                    else:
                        selected = max(0, min(len(samples) - 1, selected + delta))
                elif detail and detail.track_size:
                    row = key.y - 6
                    if key.kind in ('press', 'click') and key.x == width - 2 and 0 <= row < detail.track_size:
                        if dragging is None:
                            dragging = (row - detail.thumb_start if detail.thumb_start <= row < detail.thumb_start + detail.thumb_size
                                        else detail.thumb_size // 2)
                    if dragging is not None and key.kind in ('press', 'click', 'drag'):
                        travel = detail.track_size - detail.thumb_size
                        position = max(0, min(travel, row - dragging))
                        scroll = round(position * detail.max_scroll / travel) if travel else 0
                        if key.kind == 'click':
                            dragging = None
            continue
        if key != -1:
            dragging = None
        if key in (ord('q'), ord('Q')):
            break
        if key == ord('?'):
            show_help = not show_help
        elif key in (27, ord('b')):
            show_help, view, scroll = False, 'overview', 0
            last_detail = None
            process_page, command_offset = 0, 0
        elif key == ord('r') and poller:
            poller.refresh()
        elif not show_help:
            if view == 'detail' and key in (ord('['), ord(']')):
                process_page = max(0, process_page + (1 if key == ord(']') else -1))
            elif view == 'detail' and key in (ord(','), ord('.')):
                command_offset = max(0, command_offset + (20 if key == ord('.') else -20))
            elif view == 'detail' and key in (curses.KEY_HOME, curses.KEY_END):
                scroll = 0 if key == curses.KEY_HOME else sys.maxsize
            elif key in (10, 13, curses.KEY_ENTER):
                view, scroll = 'detail', 0
                last_detail = None
                process_page, command_offset = 0, 0
            elif key in (curses.KEY_RIGHT, ord('l'), curses.KEY_LEFT, ord('h')):
                delta = 1 if key in (curses.KEY_RIGHT, ord('l')) else -1
                selected, scroll = (selected + delta) % len(samples), 0
                last_detail = None
                process_page, command_offset = 0, 0
            elif key in (curses.KEY_UP, ord('k'), curses.KEY_DOWN, ord('j'), curses.KEY_NPAGE, curses.KEY_PPAGE):
                delta = -1 if key in (curses.KEY_UP, ord('k'), curses.KEY_PPAGE) else 1
                if view == 'overview':
                    step = overview_capacity(height) if key in (curses.KEY_NPAGE, curses.KEY_PPAGE) else 1
                    selected = max(0, min(len(samples) - 1, selected + delta * step))
                else:
                    step = max(1, height - 10) if key in (curses.KEY_NPAGE, curses.KEY_PPAGE) else 1
                    scroll = max(0, scroll + delta * step)
