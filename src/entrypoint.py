"""Choose native Hydra syntax or the previous explicit --config/--set CLI."""
from __future__ import annotations
import sys


def dispatch(native_name, legacy_main, *, legacy_extra=()):
    legacy_flags = {'--config', '--set', '--checkpoint', '--resume', '--trust-checkpoint',
                    '--index', '--out', '--force', '--device', '--legacy-config',
                    '--time-max', '--num-times', '--times', '--max-examples'}
    legacy = any(arg.split('=',1)[0] in legacy_flags or arg in legacy_extra for arg in sys.argv[1:])
    if legacy:
        legacy_main()
    else:
        from src import cli
        getattr(cli, native_name)()
