#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix damaged lotto_physics.py - extract only 3D classes properly"""

# Read damaged file
with open('e:/gptlotto/lotto_physics.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the start of Ball3D class
ball3d_start = None
for i, line in enumerate(lines):
    if '@dataclass' in line and i+1 < len(lines) and 'class Ball3D:' in lines[i+1]:
        ball3d_start = i
        break

if ball3d_start is None:
    # Try alternative search
    for i, line in enumerate(lines):
        if 'class Ball3D:' in line:
            ball3d_start = i-1 if i > 0 and '@dataclass' in lines[i-1] else i
            break

if ball3d_start:
    # Get header (first 105 lines) + Ball3D onwards
    header = lines[0:105]
    ball3d_code = lines[ball3d_start:]

    # Write corrected file
    with open('e:/gptlotto/lotto_physics.py', 'w', encoding='utf-8') as f:
        f.writelines(header)
        f.write('\n')
        f.writelines(ball3d_code)

    print(f'Fixed! Ball3D starts at line {ball3d_start+1}')
    print(f'New file: {len(header)} (header) + {len(ball3d_code)} (3D code) = {len(header) + len(ball3d_code)} lines')
else:
    print('ERROR: Could not find Ball3D class')
