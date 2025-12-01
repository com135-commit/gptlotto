#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract lotto_physics source from running process memory"""

import sys
import inspect

# Import the module (from __pycache__ or memory)
sys.path.insert(0, 'e:/gptlotto')
import lotto_physics

# Get source code
try:
    source = inspect.getsource(lotto_physics)

    # Write to file
    with open('e:/gptlotto/lotto_physics_recovered.py', 'w', encoding='utf-8') as f:
        f.write(source)

    print(f'Success! Recovered {len(source)} characters')
    print(f'File saved to: lotto_physics_recovered.py')

except Exception as e:
    print(f'Failed: {e}')
    print('Trying alternative method...')

    # Alternative: get source from file before corruption
    import importlib.util
    spec = importlib.util.find_spec('lotto_physics')
    if spec and spec.origin:
        print(f'Module location: {spec.origin}')
