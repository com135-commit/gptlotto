#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recover lotto_physics module from running process
Strategy: Import from one of the running processes and extract source
"""

import sys
import os

# Try to find and import lotto_physics from running process memory
# The running visualizer has already imported it successfully

# Method 1: Check if module is already in sys.modules (from running processes)
if 'lotto_physics' in sys.modules:
    print("Found lotto_physics in sys.modules!")
    lotto_physics = sys.modules['lotto_physics']
else:
    print("Module not in sys.modules, trying to get from damaged file location...")

    # The running processes have it in memory
    # We'll use uncompyle6 to decompile from .pyc if it exists
    import glob
    pyc_files = glob.glob('e:/gptlotto/**/*lotto_physics*.pyc', recursive=True)

    if pyc_files:
        print(f"Found .pyc files: {pyc_files}")
        # Use uncompyle6 to decompile
        try:
            import uncompyle6
            for pyc in pyc_files:
                print(f"Decompiling {pyc}...")
                with open(pyc, 'rb') as f_in:
                    with open('e:/gptlotto/lotto_physics.py', 'w', encoding='utf-8') as f_out:
                        uncompyle6.decompile_file(pyc, f_out)
                print("Success!")
                break
        except ImportError:
            print("uncompyle6 not installed")
    else:
        print("No .pyc files found")
        print("\nSearching in Python cache directories...")

        # Check AppData cache
        import pathlib
        cache_dirs = [
            pathlib.Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Python',
            pathlib.Path(os.environ.get('APPDATA', '')) / 'Python',
        ]

        for cache_dir in cache_dirs:
            if cache_dir.exists():
                pyc_files = list(cache_dir.glob('**/lotto_physics*.pyc'))
                if pyc_files:
                    print(f"Found in {cache_dir}: {len(pyc_files)} files")
                    break

        if not pyc_files:
            print("\nNo cached .pyc files found.")
            print("The running processes have the module in memory,")
            print("but we cannot extract it without external tools.")
