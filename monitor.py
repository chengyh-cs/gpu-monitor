#!/usr/bin/env python3
"""Lab GPU monitor. Run --help for options; no third-party packages required."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

from core import CommandRunner, Poller, candidate, collect, demo_samples, discover, sample_state
from ui import overview_lines


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result < 1:
        raise argparse.ArgumentTypeError('must be a finite number >= 1')
    return result


def nonnegative(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError('must be finite and >= 0')
    return result


def parser():
    p = argparse.ArgumentParser(description='Read-only laboratory GPU monitor: overview + per-server pages.')
    p.add_argument('--config', type=Path, default=Path.home() / '.ssh' / 'config')
    p.add_argument('--hosts', nargs='+', help='Monitor these SSH aliases instead of the default NCRC/SLAI inventory')
    p.add_argument('--interval', type=positive, default=60, help='Refresh interval in seconds (default: 60)')
    p.add_argument('--timeout', type=positive, default=10, help='Whole SSH query timeout in seconds (default: 10)')
    p.add_argument('--min-free-gib', type=nonnegative, default=8, help='Candidate GPU minimum free GiB (default: 8)')
    p.add_argument('--max-util', type=nonnegative, default=10, help='Candidate GPU maximum utilization %% (default: 10)')
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--list', action='store_true', help='List resolved endpoints without connecting')
    modes.add_argument('--once', action='store_true', help='Print one overview and exit')
    modes.add_argument('--json', action='store_true', help='Print one machine-readable sample and exit')
    p.add_argument('--demo', action='store_true', help='Offline simulated data; never connect to servers')
    p.add_argument('--no-color', action='store_true', help='Disable terminal colors')
    p.add_argument('--ascii', action='store_true', help='Use ASCII borders and bars instead of Unicode')
    return p


def one_shot(hosts, args):
    runner = CommandRunner()
    pool = ThreadPoolExecutor(max_workers=min(16, len(hosts)))
    try:
        futures = [pool.submit(collect, h, args.config, args.timeout, runner) for h in hosts]
        return [f.result() for f in futures]
    finally:
        runner.close()
        pool.shutdown(wait=True, cancel_futures=True)


def export(samples, args):
    now = time.time()
    rows = []
    for sample in samples:
        row = asdict(sample)
        row['state'] = sample_state(sample, now, max(15, args.interval * 2 + args.timeout))
        row['age_seconds'] = round(max(0, now - sample.sampled_at), 1) if sample.sampled_at else None
        for gpu, source in zip(row['gpus'], sample.gpus):
            gpu['candidate'] = row['state'] == 'OK' and candidate(source, args.min_free_gib, args.max_util)
        rows.append(row)
    return {'demo': args.demo, 'generated_at': now, 'memory_unit': 'MiB',
            'thresholds': {'min_free_gib': args.min_free_gib, 'max_util_percent': args.max_util},
            'servers': rows}


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.max_util > 100:
        p.error('--max-util must be <= 100')
    args.config = args.config.expanduser().resolve()
    try:
        samples = demo_samples() if args.demo else None
        hosts = [s.host for s in samples] if args.demo else discover(args.config, args.hosts)
        if not hosts:
            p.error('No laboratory aliases found. Use --hosts ALIAS [ALIAS ...].')
        if args.list:
            for h in hosts:
                print('{:<16} {}@{}:{}{}'.format(h.alias, h.user, h.hostname, h.port, '  ERROR: ' + h.error if h.error else ''))
            return int(any(h.error for h in hosts))
        if args.once or args.json:
            samples = samples if args.demo else one_shot(hosts, args)
            if args.json:
                print(json.dumps(export(samples, args), indent=2))
            else:
                print('\n'.join(overview_lines(samples, 100, max(24, len(samples) + 14), 0,
                                              min_free=args.min_free_gib, max_util=args.max_util,
                                              interval=args.interval, demo=args.demo,
                                              stale_after=max(15, args.interval * 2 + args.timeout))).rstrip())
                for s in samples:
                    if s.error:
                        from ui import clean
                        print('{}: {}'.format(s.host.alias, clean(s.error)))
            return 0 if args.demo or all(s.state in ('OK', 'NO GPU') for s in samples) else 1
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            p.error('Interactive mode needs a terminal. Use --once or --json for redirected output.')
        import curses
        from ui import run_tui
        poller = None if args.demo else Poller(hosts, args.config, args.interval, args.timeout)
        try:
            curses.wrapper(run_tui, poller, samples, args)
        finally:
            if poller:
                poller.close()
        return 0
    except (OSError, ValueError) as exc:
        print('gpu-monitor: ' + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
