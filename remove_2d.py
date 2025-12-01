#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove 2D physics code from lotto_physics.py"""

# Read file
with open('e:/gptlotto/lotto_physics.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Keep header (lines 1-105) + 3D code (lines 2332-end)
header = lines[0:105]
lines_3d = lines[2332:]

# Write new file
with open('e:/gptlotto/lotto_physics.py', 'w', encoding='utf-8') as f:
    f.writelines(header)
    f.write('\n')
    f.writelines(lines_3d)

print('OK: 2D code deleted')
print(f'Deleted lines: 106-2332 ({2332-106} lines)')
print(f'Remaining: {len(header)} (header) + {len(lines_3d)} (3D code) = {len(header) + len(lines_3d)} lines')
