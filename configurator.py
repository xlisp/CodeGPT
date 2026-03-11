"""
Poor Man's Configurator. Runs a config file then applies CLI overrides.

Usage:
    python train.py config/train_codegpt.py --batch_size=32
"""
import sys
import os


def configure(default_config=None):
    """
    Override config variables from command line and config files.
    Call this after defining all default config variables in the caller's scope.
    """
    import importlib.util

    caller_globals = sys._getframe(1).f_globals

    # first, apply config file if provided
    for arg in sys.argv[1:]:
        if arg.endswith('.py') and not arg.startswith('--'):
            config_path = arg
            if not os.path.isabs(config_path):
                config_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), config_path)
            spec = importlib.util.spec_from_file_location("config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            for key in dir(config_module):
                if not key.startswith('_'):
                    caller_globals[key] = getattr(config_module, key)

    # then apply command-line overrides
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            key, val = arg.split('=', 1) if '=' in arg else (arg, 'True')
            key = key.lstrip('-')
            if key in caller_globals:
                old_val = caller_globals[key]
                if isinstance(old_val, bool):
                    caller_globals[key] = val.lower() in ('true', '1', 'yes')
                elif isinstance(old_val, int):
                    caller_globals[key] = int(val)
                elif isinstance(old_val, float):
                    caller_globals[key] = float(val)
                elif isinstance(old_val, str):
                    caller_globals[key] = val
                elif old_val is None:
                    caller_globals[key] = val
                else:
                    raise ValueError(f"Don't know how to parse type {type(old_val)} for key {key}")
            else:
                raise ValueError(f"Unknown config key: {key}")
