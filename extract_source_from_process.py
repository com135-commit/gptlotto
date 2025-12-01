#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract lotto_physics source from running process
Strategy: Use inspect module on already-loaded module
"""

import sys
import os

# Import from the Python 3.9 .pyc cache
sys.path.insert(0, 'e:/gptlotto/__pycache__')

try:
    # Try to load from .pyc directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'lotto_physics_real',
        'e:/gptlotto/__pycache__/lotto_physics.cpython-39.pyc'
    )

    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get all class and function source
        import inspect

        output = []
        output.append("#!/usr/bin/env python3")
        output.append("# -*- coding: utf-8 -*-")
        output.append("# Recovered from Python 3.9 .pyc\n")

        # Get module docstring
        if module.__doc__:
            output.append(f'"""{module.__doc__}"""')

        # Get all members
        for name, obj in inspect.getmembers(module):
            if name.startswith('_'):
                continue

            try:
                if inspect.isclass(obj):
                    source = inspect.getsource(obj)
                    output.append(f"\n# Class: {name}")
                    output.append(source)
                elif inspect.isfunction(obj):
                    source = inspect.getsource(obj)
                    output.append(f"\n# Function: {name}")
                    output.append(source)
            except (TypeError, OSError) as e:
                print(f"Cannot get source for {name}: {e}")

        # Write to file
        recovered_code = '\n'.join(output)
        with open('e:/gptlotto/lotto_physics_recovered.py', 'w', encoding='utf-8') as f:
            f.write(recovered_code)

        print(f"SUCCESS! Recovered {len(recovered_code)} characters")
        print(f"File: lotto_physics_recovered.py")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
