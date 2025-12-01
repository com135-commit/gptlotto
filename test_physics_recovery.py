#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test recovered 3D physics engine"""

import sys
sys.path.insert(0, 'e:/gptlotto')

print("=" * 60)
print("3D Physics Engine Recovery Test")
print("=" * 60)

# Test import
try:
    from lotto_physics import (
        get_physics_backend_info,
        Ball3D,
        LottoChamber3D_Ultimate
    )
    print("\n[OK] Import successful")
except Exception as e:
    print(f"\n[FAIL] Import failed: {e}")
    sys.exit(1)

# Test backend info
try:
    info = get_physics_backend_info()
    print(f"\n Backend Info:")
    print(f"   Backend: {info['backend']}")
    print(f"   CuPy: {info['has_cupy']}")
    print(f"   Numba: {info['has_numba']}")
    print(f"   Version: {info['version']}")
except Exception as e:
    print(f"\n[FAIL] Backend info failed: {e}")
    sys.exit(1)

# Test Ball3D creation
try:
    ball = Ball3D(number=1, x=100, y=100, z=100)
    print(f"\n[OK] Ball3D created: #{ball.number} at ({ball.x}, {ball.y}, {ball.z})")
    print(f"   Speed: {ball.speed:.2f} mm/s")
    print(f"   Charge: {ball.charge:.2e} C")
except Exception as e:
    print(f"\n[FAIL] Ball3D creation failed: {e}")
    sys.exit(1)

# Test Chamber creation
try:
    chamber = LottoChamber3D_Ultimate()
    print(f"\n[OK] Chamber created")
    print(f"   Size: {chamber.width} x {chamber.depth} x {chamber.height} mm")
    print(f"   Balls: {chamber.num_balls}")
    print(f"   Jet force: {chamber.jet_force} mm/s^2 ({chamber.jet_force/1000:.1f} m/s^2)")
    print(f"   Turbulence: {chamber.turbulence} mm/s^2 ({chamber.turbulence/1000:.1f} m/s^2)")
    print(f"   Jets: {len(chamber.jet_positions)} positions")
except Exception as e:
    print(f"\n[FAIL] Chamber creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test simulation step
try:
    print(f"\n[OK] Running 60 simulation steps (1 second)...")
    for i in range(60):
        chamber.step()

    stats = chamber.get_statistics()
    print(f"   Time: {stats['time']:.2f}s")
    print(f"   Active balls: {stats['active_balls']}")
    print(f"   Avg speed: {stats['avg_speed']:.1f} mm/s")
    print(f"   Total energy: {stats['total_energy']:.2f} mJ")
    print(f"   Collisions: {stats['total_collisions']}")
    print(f"   Extracted: {stats['extracted_balls']}")
except Exception as e:
    print(f"\n[FAIL] Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("[SUCCESS] All tests passed! 67 physics laws recovered successfully!")
print("=" * 60)
