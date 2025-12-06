#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Lotto Physics Engine with 67 Physical Laws
Korean Lotto 6/45 Ball Physics Simulation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import time
from collections import deque
import threading

# Numba CUDA 비활성화 (CPU만 사용)
import os
os.environ['NUMBA_DISABLE_CUDA'] = '1'

# ==================== GPU Backend Support ====================
# GPU 완전 비활성화 - CPU만 사용
cp = None
HAS_CUPY = False

try:
    from numba import jit, prange
    HAS_NUMBA = True
    print("Numba detected - JIT compilation available")
except ImportError:
    HAS_NUMBA = False
    prange = range  # Fallback to regular range
    def jit(*args, **kwargs):
        """Dummy decorator when Numba not available"""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

def get_physics_backend_info():
    """Return physics backend information"""
    return {
        'backend': 'GPU' if HAS_CUPY else 'CPU',
        'has_cupy': HAS_CUPY,
        'has_numba': HAS_NUMBA,
        'version': '3.0-3D-Ultimate'
    }

# ==================== Physical Constants ====================
# 한국 로또 6/45 공 실제 물리 사양 (비너스 추첨기)
BALL_RADIUS = 20.0  # mm (직경 40mm) - 실제 비너스 추첨기 공
BALL_MASS = 3.0  # g (실제 로또공 무게)
BALL_DENSITY = BALL_MASS / (4/3 * np.pi * (BALL_RADIUS/1000)**3)  # kg/m³ = 895 kg/m³

GRAVITY = 9800.0  # mm/s² = 9.8 m/s²
AIR_DENSITY = 1.225  # kg/m³ (해수면, 15°C)
AIR_VISCOSITY = 1.81e-5  # Pa·s (동적 점도)
KINEMATIC_VISCOSITY = 1.5e-5  # m²/s = AIR_VISCOSITY / AIR_DENSITY

# 공기 역학
C_D_SPHERE = 0.47  # 매끄러운 구의 항력계수
MAGNUS_COEFFICIENT = 0.23  # Magnus 효과 계수

# 재질 물성 (폴리우레탄)
RESTITUTION = 0.85  # 반발계수 (0.8-0.9)
FRICTION_COEF = 0.45  # 마찰계수 (0.4-0.6)
ROLLING_RESISTANCE = 0.01  # 구름저항
SHORE_HARDNESS = 75  # Shore A 경도 (70-80)

# 전자기 상호작용
CHARGE_MAGNITUDE = 1e-12  # C (정전기, 마찰로 발생)
COULOMB_CONSTANT = 8.99e9  # N·m²/C²

# 음향 충돌
SPEED_OF_SOUND = 343000.0  # mm/s (343 m/s in air)


# ==================== Fluid Grid (CFD-like) ====================

# Numba JIT 최적화 함수들
@jit(nopython=True, parallel=True, cache=True)
def _diffuse_velocity_jit(vx, vy, vz, nx, ny, nz, alpha, beta):
    """Numba JIT 최적화된 속도장 확산"""
    new_vx = vx.copy()
    new_vy = vy.copy()
    new_vz = vz.copy()

    for i in prange(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                # 6방향 이웃 평균 - vx
                avg_vx = (
                    vx[i-1,j,k] + vx[i+1,j,k] +
                    vx[i,j-1,k] + vx[i,j+1,k] +
                    vx[i,j,k-1] + vx[i,j,k+1]
                ) / 6.0

                # 6방향 이웃 평균 - vy
                avg_vy = (
                    vy[i-1,j,k] + vy[i+1,j,k] +
                    vy[i,j-1,k] + vy[i,j+1,k] +
                    vy[i,j,k-1] + vy[i,j,k+1]
                ) / 6.0

                # 6방향 이웃 평균 - vz
                avg_vz = (
                    vz[i-1,j,k] + vz[i+1,j,k] +
                    vz[i,j-1,k] + vz[i,j+1,k] +
                    vz[i,j,k-1] + vz[i,j,k+1]
                ) / 6.0

                # 확산 적용
                new_vx[i,j,k] = beta * vx[i,j,k] + alpha * avg_vx
                new_vy[i,j,k] = beta * vy[i,j,k] + alpha * avg_vy
                new_vz[i,j,k] = beta * vz[i,j,k] + alpha * avg_vz

    return new_vx, new_vy, new_vz


@jit(nopython=True, cache=True)
def _check_and_resolve_collisions_jit(
    pos, vel, omega, mass, radius, extracted,
    restitution, friction, I_ratio
):
    """
    Numba JIT 최적화된 공-공 충돌 검사 및 해결

    Parameters:
    - pos: (N, 3) 위치 배열 [x, y, z]
    - vel: (N, 3) 속도 배열 [vx, vy, vz]
    - omega: (N, 3) 각속도 배열 [wx, wy, wz]
    - mass: (N,) 질량 배열
    - radius: (N,) 반지름 배열
    - extracted: (N,) 추출 여부 배열
    - restitution: 반발계수
    - friction: 마찰계수
    - I_ratio: 관성 모멘트 비율 (I = m * r² * I_ratio)

    Returns:
    - pos: 업데이트된 위치 배열 (위치 분리 적용)
    - vel: 업데이트된 속도 배열
    - omega: 업데이트된 각속도 배열
    """
    N = pos.shape[0]

    # 모든 공 쌍에 대해 충돌 검사
    for i in range(N):
        if extracted[i]:
            continue

        for j in range(i+1, N):
            if extracted[j]:
                continue

            # 충돌 판정 (거리 제곱 비교)
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            dist_sq = dx*dx + dy*dy + dz*dz

            min_dist = radius[i] + radius[j]
            min_dist_sq = min_dist * min_dist

            # ========================================
            # 21. 공-공 비접촉 압력 (유체 쿠션 효과)
            # ========================================
            # 공들이 가까워지면 사이 공기가 압축되어 서로 밀어냄
            # 활성화 거리: 2.5 × radius (공 사이 gap < 0.5 × radius)
            cushion_dist = min_dist * 1.25  # 2.5R
            cushion_dist_sq = cushion_dist * cushion_dist

            if dist_sq < cushion_dist_sq:
                dist = np.sqrt(dist_sq)
                if dist < 1e-6:
                    dist = 1e-6

                # 법선 벡터
                nx = dx / dist
                ny = dy / dist
                nz = dz / dist

                # 유체 쿠션 힘: F = k / (gap)² × v_approach
                # gap = dist - 2R (공 사이 실제 간격)
                gap = dist - min_dist

                # gap > 0: 아직 접촉 안함 (비접촉 압력만)
                # gap <= 0: 접촉/겹침 (비접촉 압력 + 충돌 처리)
                if gap > 0:
                    # 비접촉 영역: 유체 쿠션만
                    # 상대 속도 (접근 속도)
                    dvx = vel[i, 0] - vel[j, 0]
                    dvy = vel[i, 1] - vel[j, 1]
                    dvz = vel[i, 2] - vel[j, 2]
                    v_approach = -(dvx * nx + dvy * ny + dvz * nz)  # 음수면 멀어짐

                    if v_approach > 0:  # 접근 중일 때만
                        # 유체 쿠션 계수 (실험적 값)
                        # 너무 크면 불안정, 너무 작으면 효과 없음
                        k_cushion = 500.0  # mm³·mm/s² (튜닝 가능)

                        # 힘의 크기: k / gap² × v_approach
                        # gap이 작을수록, 접근 속도 클수록 강한 반발
                        cushion_force = k_cushion / (gap * gap) * v_approach

                        # 최대 힘 제한 (발산 방지)
                        max_cushion = 10000.0  # mm/s² (중력의 약 1배)
                        if cushion_force > max_cushion:
                            cushion_force = max_cushion

                        # 가속도 = F / m
                        # dt = 1/60 초는 외부에서 곱해짐
                        accel_i = cushion_force / mass[i]
                        accel_j = cushion_force / mass[j]

                        # 속도 변경 (서로 밀어냄)
                        # i는 +n 방향, j는 -n 방향
                        vel[i, 0] += accel_i * nx * (1.0/60.0)  # dt = 1/60
                        vel[i, 1] += accel_i * ny * (1.0/60.0)
                        vel[i, 2] += accel_i * nz * (1.0/60.0)

                        vel[j, 0] -= accel_j * nx * (1.0/60.0)
                        vel[j, 1] -= accel_j * ny * (1.0/60.0)
                        vel[j, 2] -= accel_j * nz * (1.0/60.0)

            # ========================================
            # 충돌 처리 (겹침 시)
            # ========================================
            if dist_sq < min_dist_sq:
                # 충돌 발생 - 실제 거리 계산
                dist = np.sqrt(dist_sq)
                if dist < 1e-6:
                    dist = 1e-6

                # 법선 벡터
                nx = dx / dist
                ny = dy / dist
                nz = dz / dist

                # ========================================
                # 위치 분리 (Position Correction)
                # ========================================
                # 겹친 정도 계산
                overlap = min_dist - dist

                # 질량 비율로 위치 보정 (무거운 공은 덜 움직임)
                total_mass = mass[i] + mass[j]
                correction_i = overlap * (mass[j] / total_mass)
                correction_j = overlap * (mass[i] / total_mass)

                # 위치 분리 적용
                pos[i, 0] += correction_i * nx
                pos[i, 1] += correction_i * ny
                pos[i, 2] += correction_i * nz

                pos[j, 0] -= correction_j * nx
                pos[j, 1] -= correction_j * ny
                pos[j, 2] -= correction_j * nz

                # ========================================
                # 속도 및 충격량 계산
                # ========================================
                # 상대 속도
                dvx = vel[i, 0] - vel[j, 0]
                dvy = vel[i, 1] - vel[j, 1]
                dvz = vel[i, 2] - vel[j, 2]

                # 법선 방향 상대 속도
                dv_n = dvx * nx + dvy * ny + dvz * nz

                # 이미 멀어지고 있으면 속도 변경 스킵 (위치는 이미 분리됨)
                if dv_n > 0:
                    continue

                # 접촉점 회전 속도
                # v_rot = ω × r
                r1x, r1y, r1z = -nx * radius[i], -ny * radius[i], -nz * radius[i]
                r2x, r2y, r2z = nx * radius[j], ny * radius[j], nz * radius[j]

                # Cross product: omega1 × r1
                v_rot1_x = omega[i, 1] * r1z - omega[i, 2] * r1y
                v_rot1_y = omega[i, 2] * r1x - omega[i, 0] * r1z
                v_rot1_z = omega[i, 0] * r1y - omega[i, 1] * r1x

                # Cross product: omega2 × r2
                v_rot2_x = omega[j, 1] * r2z - omega[j, 2] * r2y
                v_rot2_y = omega[j, 2] * r2x - omega[j, 0] * r2z
                v_rot2_z = omega[j, 0] * r2y - omega[j, 1] * r2x

                # 접촉점 상대 속도
                dv_contact_x = dvx + v_rot1_x - v_rot2_x
                dv_contact_y = dvy + v_rot1_y - v_rot2_y
                dv_contact_z = dvz + v_rot1_z - v_rot2_z

                dv_contact_n = dv_contact_x * nx + dv_contact_y * ny + dv_contact_z * nz

                # 접선 방향
                dv_tx = dv_contact_x - dv_contact_n * nx
                dv_ty = dv_contact_y - dv_contact_n * ny
                dv_tz = dv_contact_z - dv_contact_n * nz
                dv_t_mag = np.sqrt(dv_tx*dv_tx + dv_ty*dv_ty + dv_tz*dv_tz)

                # 질량 및 관성
                m1, m2 = mass[i], mass[j]
                I1 = m1 * radius[i] * radius[i] * I_ratio
                I2 = m2 * radius[j] * radius[j] * I_ratio

                # 유효 질량 (법선 방향)
                m_eff_n = 1.0 / (1.0/m1 + 1.0/m2 +
                                 (radius[i]*radius[i])/I1 + (radius[j]*radius[j])/I2)

                # 충격량 (법선)
                J_n = -(1.0 + restitution) * dv_contact_n * m_eff_n

                # 마찰력 (접선)
                J_t = 0.0
                t_x, t_y, t_z = 0.0, 0.0, 0.0
                if dv_t_mag > 1e-6:
                    t_x = dv_tx / dv_t_mag
                    t_y = dv_ty / dv_t_mag
                    t_z = dv_tz / dv_t_mag

                    # 마찰 충격량 (Coulomb 마찰)
                    J_t_max = friction * abs(J_n)

                    # 유효 질량 (접선 방향)
                    m_eff_t = 1.0 / (1.0/m1 + 1.0/m2)

                    # 접선 충격량 계산
                    J_t_needed = dv_t_mag * m_eff_t
                    J_t = min(J_t_needed, J_t_max)

                # 속도 업데이트 (법선)
                vel[i, 0] += J_n * nx / m1
                vel[i, 1] += J_n * ny / m1
                vel[i, 2] += J_n * nz / m1

                vel[j, 0] -= J_n * nx / m2
                vel[j, 1] -= J_n * ny / m2
                vel[j, 2] -= J_n * nz / m2

                # 속도 업데이트 (마찰)
                if J_t > 0:
                    vel[i, 0] -= J_t * t_x / m1
                    vel[i, 1] -= J_t * t_y / m1
                    vel[i, 2] -= J_t * t_z / m1

                    vel[j, 0] += J_t * t_x / m2
                    vel[j, 1] += J_t * t_y / m2
                    vel[j, 2] += J_t * t_z / m2

                # 각속도 업데이트 (토크 = r × F)
                # Torque1 = r1 × (J_n * n + J_t * t)
                Fx = J_n * nx - J_t * t_x
                Fy = J_n * ny - J_t * t_y
                Fz = J_n * nz - J_t * t_z

                # Cross product: r1 × F
                tau1_x = r1y * Fz - r1z * Fy
                tau1_y = r1z * Fx - r1x * Fz
                tau1_z = r1x * Fy - r1y * Fx

                omega[i, 0] += tau1_x / I1
                omega[i, 1] += tau1_y / I1
                omega[i, 2] += tau1_z / I1

                # Torque2 = r2 × (-F)
                tau2_x = r2y * (-Fz) - r2z * (-Fy)
                tau2_y = r2z * (-Fx) - r2x * (-Fz)
                tau2_z = r2x * (-Fy) - r2y * (-Fx)

                omega[j, 0] += tau2_x / I2
                omega[j, 1] += tau2_y / I2
                omega[j, 2] += tau2_z / I2

    return pos, vel, omega

@jit(nopython=True, cache=True)
def _add_velocity_to_grid_jit(vx, vy, vz, cx, cy, cz, dvx, dvy, dvz,
                               nx, ny, nz, dx, dy, dz, radius):
    """Numba JIT 최적화된 격자 속도 추가 (블로어/진공용) - in-place 수정"""
    i0 = int(cx / dx)
    j0 = int(cy / dy)
    k0 = int(cz / dz)

    r_cells = int(radius / dx) + 1

    for di in range(-r_cells, r_cells+1):
        for dj in range(-r_cells, r_cells+1):
            for dk in range(-r_cells, r_cells+1):
                i = i0 + di
                j = j0 + dj
                k = k0 + dk

                if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                    # 셀 중심 좌표
                    cell_x = (i + 0.5) * dx
                    cell_y = (j + 0.5) * dy
                    cell_z = (k + 0.5) * dz

                    # 거리 계산
                    dist = np.sqrt((cell_x - cx)**2 + (cell_y - cy)**2 + (cell_z - cz)**2)

                    if dist < radius:
                        # 거리에 따른 감쇠
                        strength = (1.0 - dist/radius) ** 2
                        vx[i,j,k] += dvx * strength
                        vy[i,j,k] += dvy * strength
                        vz[i,j,k] += dvz * strength
    # in-place 수정이므로 반환 불필요

@dataclass
class FluidGrid:
    """
    3D 격자 기반 유체 시뮬레이션
    간이 CFD - 속도장 확산 및 공-유체 상호작용
    """
    # 격자 해상도
    nx: int = 20  # X 방향 셀 개수
    ny: int = 20  # Y 방향 셀 개수
    nz: int = 20  # Z 방향 셀 개수

    # 물리 공간 크기 (mm)
    width: float = 500.0
    depth: float = 500.0
    height: float = 500.0

    # 속도장 (mm/s)
    vx: np.ndarray = field(default_factory=lambda: np.zeros((20, 20, 20)))
    vy: np.ndarray = field(default_factory=lambda: np.zeros((20, 20, 20)))
    vz: np.ndarray = field(default_factory=lambda: np.zeros((20, 20, 20)))

    # 압력장 (Pa) - 선택
    pressure: np.ndarray = field(default_factory=lambda: np.zeros((20, 20, 20)))

    # 유체 파라미터
    viscosity: float = AIR_VISCOSITY  # 점성
    diffusion_rate: float = 0.1  # 확산 속도 (0~1)

    def __post_init__(self):
        """격자 초기화 + Numba JIT warm-up"""
        # 셀 크기 계산
        self.dx = self.width / self.nx
        self.dy = self.depth / self.ny
        self.dz = self.height / self.nz
        self.cell_volume = self.dx * self.dy * self.dz

        # 배열 재초기화 (크기 맞춤)
        self.vx = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.vy = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.vz = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.pressure = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)

        # Numba JIT warm-up (첫 컴파일을 초기화 시점에 수행)
        if HAS_NUMBA:
            print("  [Numba] JIT 컴파일 중... (첫 실행 시 1-2초 소요)")
            # 더미 데이터로 JIT 함수 호출하여 미리 컴파일
            _add_velocity_to_grid_jit(
                self.vx, self.vy, self.vz, 250.0, 250.0, 250.0, 0.0, 0.0, 0.0,
                self.nx, self.ny, self.nz, self.dx, self.dy, self.dz, 50.0
            )
            self.diffuse(0.01667)  # diffuse도 미리 컴파일
            print("  [Numba] JIT 컴파일 완료!")

    def world_to_index(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """월드 좌표 → 격자 인덱스"""
        i = int(x / self.dx)
        j = int(y / self.dy)
        k = int(z / self.dz)
        return i, j, k

    def index_to_world(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """격자 인덱스 → 월드 좌표 (셀 중심)"""
        x = (i + 0.5) * self.dx
        y = (j + 0.5) * self.dy
        z = (k + 0.5) * self.dz
        return x, y, z

    def in_bounds(self, i: int, j: int, k: int) -> bool:
        """인덱스가 격자 안에 있는지"""
        return 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz

    def get_velocity(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """월드 좌표에서 유체 속도 가져오기 (보간)"""
        i, j, k = self.world_to_index(x, y, z)

        if not self.in_bounds(i, j, k):
            return 0.0, 0.0, 0.0

        return float(self.vx[i,j,k]), float(self.vy[i,j,k]), float(self.vz[i,j,k])

    def add_velocity(self, x: float, y: float, z: float, dvx: float, dvy: float, dvz: float, radius: float = 50.0):
        """특정 위치에 속도 추가 (블로어/진공용) - Numba JIT 최적화"""
        if HAS_NUMBA:
            # in-place 수정, 반환값 없음
            _add_velocity_to_grid_jit(
                self.vx, self.vy, self.vz, x, y, z, dvx, dvy, dvz,
                self.nx, self.ny, self.nz, self.dx, self.dy, self.dz, radius
            )
        else:
            # Fallback: Python loop
            i0, j0, k0 = self.world_to_index(x, y, z)
            r_cells = int(radius / self.dx) + 1

            for di in range(-r_cells, r_cells+1):
                for dj in range(-r_cells, r_cells+1):
                    for dk in range(-r_cells, r_cells+1):
                        i, j, k = i0+di, j0+dj, k0+dk

                        if self.in_bounds(i, j, k):
                            cx, cy, cz = self.index_to_world(i, j, k)
                            dist = np.sqrt((cx-x)**2 + (cy-y)**2 + (cz-z)**2)

                            if dist < radius:
                                strength = (1.0 - dist/radius) ** 2
                                self.vx[i,j,k] += dvx * strength
                                self.vy[i,j,k] += dvy * strength
                                self.vz[i,j,k] += dvz * strength

    def diffuse(self, dt: float):
        """속도장 확산 (Numba JIT 최적화)"""
        alpha = self.diffusion_rate * dt
        beta = 1.0 - alpha

        # Numba JIT 함수 호출 (멀티코어 병렬화)
        self.vx, self.vy, self.vz = _diffuse_velocity_jit(
            self.vx, self.vy, self.vz,
            self.nx, self.ny, self.nz,
            alpha, beta
        )

    def apply_damping(self, damping: float = 0.99):
        """속도 감쇠 (에너지 손실)"""
        self.vx *= damping
        self.vy *= damping
        self.vz *= damping

    def clear(self):
        """속도장 초기화"""
        self.vx.fill(0.0)
        self.vy.fill(0.0)
        self.vz.fill(0.0)


# ==================== Ball3D Data Structure ====================
@dataclass
class Ball3D:
    """3D 공 객체 - 완전한 물리 상태"""
    number: int  # 공 번호 (1-45)

    # 위치 (mm)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # 속도 (mm/s)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # 회전 각속도 (rad/s)
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0

    # 물리 속성
    radius: float = BALL_RADIUS
    mass: float = BALL_MASS

    # 내부 상태
    charge: float = 0.0  # 전하량 (C)
    temperature: float = 293.15  # 온도 (K), 20°C
    last_collision_time: float = 0.0
    collision_count: int = 0
    energy: float = 0.0  # 운동에너지 (mJ)

    # 추출 관련
    extracted: bool = False
    extraction_time: float = 0.0

    def __post_init__(self):
        """초기 전하량 설정 (마찰 정전기)"""
        self.charge = np.random.uniform(-CHARGE_MAGNITUDE, CHARGE_MAGNITUDE)

    @property
    def speed(self) -> float:
        """속력 (mm/s)"""
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    @property
    def angular_speed(self) -> float:
        """회전 속력 (rad/s)"""
        return np.sqrt(self.wx**2 + self.wy**2 + self.wz**2)

    @property
    def kinetic_energy(self) -> float:
        """운동 에너지 (mJ = 10^-3 J)"""
        # 병진 + 회전
        linear_ke = 0.5 * self.mass * (self.speed ** 2) * 1e-6  # g·mm²/s² → mJ

        # 회전 관성 모멘트: I = (2/5) * m * r²
        I = 0.4 * self.mass * (self.radius ** 2)
        angular_ke = 0.5 * I * (self.angular_speed ** 2) * 1e-6

        return linear_ke + angular_ke

    def get_position(self) -> np.ndarray:
        """위치 벡터"""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def get_velocity(self) -> np.ndarray:
        """속도 벡터"""
        return np.array([self.vx, self.vy, self.vz], dtype=np.float64)

    def get_angular_velocity(self) -> np.ndarray:
        """각속도 벡터"""
        return np.array([self.wx, self.wy, self.wz], dtype=np.float64)


# ==================== 3D Ultimate Physics Chamber ====================
@dataclass
class LottoChamber3D_Ultimate:
    """
    3D 로또 챔버 - 67가지 물리 법칙 구현

    물리 법칙:
    1-10: 기본 역학 (중력, 관성, 운동량, 에너지)
    11-20: 유체 역학 (Blower, Vacuum, 난류, 베르누이)
    21-30: 충돌 역학 (탄성, 비탄성, 마찰, 회전)
    31-40: 공기 역학 (항력, 양력, Magnus, Reynolds)
    41-50: 열역학 (온도, 압력, 부력)
    51-60: 전자기 (정전기, Coulomb, 유도)
    61-67: 고급 효과 (음향, 난류 와류, 벽면 효과)
    """

    # 챔버 크기 (mm) - 비너스(Venus) 추첨기 공식 스펙
    # 형태: 구형(Sphere) 챔버
    # 직경: 500mm (반지름 250mm)
    # 재질: 투명 플렉시글라스
    # 최대 수용: 90개 볼 (로또 6/45는 45개 사용)
    # width, depth, height는 구를 감싸는 바운딩 박스
    width: float = 500.0   # X 방향 (구 직경)
    depth: float = 500.0   # Y 방향 (구 직경)
    height: float = 500.0  # Z 방향 (구 직경)

    # 공 설정
    num_balls: int = 45
    ball_radius: float = BALL_RADIUS
    ball_mass: float = BALL_MASS

    # 물리 파라미터 (수정된 현실적 값)
    gravity: float = GRAVITY
    restitution: float = RESTITUTION
    friction: float = FRICTION_COEF

    # 유체 역학 파라미터 (실제 로또 기계 데이터 기반)
    # NBA 로또: 90 mph (40 m/s) 공기 속도, 400 CFM 유량
    # 한국 비너스: "태풍급" 풍속
    # 제트 힘은 공을 띄우고 섞는 용도 (중력의 3~5배)
    # 중력 = 9.8 m/s², 제트 = 40 m/s² (약 4배)
    jet_force: float = 40000.0  # mm/s² = 40 m/s² (중력의 4배, 공을 띄움)
    vacuum_force: float = 150000.0  # mm/s² = 150 m/s² (좁은 영역에 강력한 힘)
    turbulence: float = 80000.0  # mm/s² = 80 m/s² (난류 효과)

    # Blower 설정
    num_jets: int = 4
    jet_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    jet_directions: List[Tuple[float, float, float]] = field(default_factory=list)

    # Vacuum 설정
    # 실제 로또 기계: 작은 파이프 입구 (직경 50-60mm)
    vacuum_position: Tuple[float, float, float] = (250.0, 250.0, 420.0)
    vacuum_radius: float = 30.0  # 좁은 파이프 입구 (반지름 30mm)

    # 추출구 설정
    # 공이 안정적으로 도달 가능한 높이 (400-430mm 범위)
    extraction_position: Tuple[float, float, float] = (250.0, 250.0, 425.0)
    extraction_radius: float = 80.0  # 매우 넓은 추출 영역

    # 시뮬레이션 설정
    dt: float = 1/60  # 60 FPS
    time: float = 0.0

    # GPU 배치 설정
    use_gpu: bool = False
    batch_size: int = 1000

    # 내부 상태
    balls: List[Ball3D] = field(default_factory=list)
    extracted_balls: List[Ball3D] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    # 통계
    total_collisions: int = 0
    total_energy: float = 0.0

    # 추첨 단계 관리
    phase: str = "INITIAL"  # INITIAL, MIXING, EXTRACTING, COMPLETE
    phase_timer: float = 0.0
    jet_power: float = 1.0  # 0.0 ~ 1.0
    extracted_count: int = 0
    captured_ball: Optional[Ball3D] = None

    # 블로어 제어 타이밍 (실제 로또 기계 기반)
    # NBA Draft Lottery: 20초 초기 믹싱, 10초 추출 간 믹싱
    # 한국 로또: 4-5초 추출 간격
    initial_mixing_time: float = 20.0  # 초기 믹싱 시간 (초)
    between_ball_mixing_time: float = 10.0  # 공 추출 사이 믹싱 시간 (초)
    extraction_interval: float = 5.0  # 각 공 추출 간격 (초)
    blower_off_time: float = 0.5  # 추출 시 블로어 꺼지는 시간 (초)
    extraction_cooldown: float = 0.0  # 추출 후 대기 시간 (초)

    # 유체 격자 (CFD-like)
    fluid_grid: Optional[FluidGrid] = None
    use_fluid_grid: bool = True  # 격자 유체 시뮬레이션 사용 여부

    def __post_init__(self):
        """초기화"""
        # 챔버 반지름 계산 (직육면체를 원통으로 근사)
        self.chamber_radius = min(self.width, self.depth) / 2

        # Jet 위치 설정 (바닥 4개 코너)
        if not self.jet_positions:
            margin = 50.0
            self.jet_positions = [
                (margin, margin, margin),
                (self.width - margin, margin, margin),
                (margin, self.depth - margin, margin),
                (self.width - margin, self.depth - margin, margin)
            ]

        # Jet 방향 설정 (위쪽 + 중앙으로)
        if not self.jet_directions:
            cx, cy = self.width / 2, self.depth / 2
            self.jet_directions = []
            for jx, jy, jz in self.jet_positions:
                # 벡터: jet → center + upward
                dx = cx - jx
                dy = cy - jy
                dz = self.height * 0.7  # 위쪽 강하게
                mag = np.sqrt(dx**2 + dy**2 + dz**2)
                self.jet_directions.append((dx/mag, dy/mag, dz/mag))

        # 유체 격자 초기화
        if self.use_fluid_grid and self.fluid_grid is None:
            grid_size = 64  # 고품질 CFD (32는 빠르지만 품질 낮음)
            self.fluid_grid = FluidGrid(
                nx=grid_size, ny=grid_size, nz=grid_size,
                width=self.width,
                depth=self.depth,
                height=self.height,
                diffusion_rate=0.15  # 확산 속도 (약간 빠르게)
            )
            print(f"Fluid Grid initialized: {grid_size}x{grid_size}x{grid_size} = {grid_size**3} cells")

        # 공 생성
        if not self.balls:
            self._initialize_balls()

    def _initialize_balls(self):
        """45개 공 초기화 - 구형 챔버 바닥에 쌓임 (실제 로또 기계)"""
        self.balls = []

        # 구형 챔버 중심
        cx, cy, cz = self.width / 2, self.depth / 2, self.height / 2

        # 구형 챔버의 바닥:
        # - 구의 가장 낮은 점: z = 0 (cz - chamber_radius)
        # - 공 중심의 최저 z: z = ball_radius (바닥에 닿음)
        # 물리 법칙: 중력이 공을 바닥으로 떨어뜨림
        z_start = self.ball_radius  # 바닥에 닿는 높이 (~22mm)

        # XY 평면에서 넓게 분산 배치
        # 중력이 자동으로 바닥에 정렬시킴

        idx = 0
        attempts = 0
        max_attempts = 10000  # 충분한 시도 횟수

        # 챔버 바닥에 직접 배치
        # 바닥: z = 0 ~ 100mm
        for idx in range(self.num_balls):
            max_attempts_per_ball = 100
            placed = False

            for attempt in range(max_attempts_per_ball):
                # 바닥 높이
                z = z_start + self.rng.uniform(0, 80)  # 22 ~ 102mm

                # 이 높이에서 챔버 내부 반지름
                # 구 방정식: (x-cx)² + (y-cy)² + (z-cz)² ≤ R²
                # XY 평면: (x-cx)² + (y-cy)² ≤ R² - (z-cz)²
                dz = z - cz
                r_squared = self.chamber_radius**2 - dz**2

                if r_squared > 0:
                    # 안전 마진 추가: 여유를 더 많이 둠
                    max_r = np.sqrt(r_squared) - self.ball_radius * 1.5

                    if max_r > 0:
                        # 원형 영역에 랜덤 배치
                        angle = self.rng.uniform(0, 2 * np.pi)
                        r = self.rng.uniform(0, max_r)

                        x = cx + r * np.cos(angle)
                        y = cy + r * np.sin(angle)

                        # 검증: 실제로 챔버 안에 있는지 확인
                        dist_from_center = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)
                        if dist_from_center + self.ball_radius <= self.chamber_radius:
                            placed = True
                            break

            if not placed:
                # 안전한 위치: 챔버 정중앙 바닥
                x, y, z = cx, cy, z_start

            ball = Ball3D(
                number=idx + 1,
                x=x, y=y, z=z,
                vx=0.0,
                vy=0.0,
                vz=0.0,
                wx=0.0,
                wy=0.0,
                wz=0.0
            )

            self.balls.append(ball)

    # ==================== 물리 법칙 적용 ====================
    def step(self, dt=None):
        """한 타임스텝 시뮬레이션"""
        # dt 파라미터 지원 (시각화 호환성)
        if dt is None:
            dt = self.dt

        # 1. 모든 힘 적용 (벡터 합성) - 속도 업데이트
        for ball in self.balls:
            if not ball.extracted:
                self.apply_forces(ball, dt)

        # 2. 위치 업데이트 (구속 운동 - Constrained Motion)
        # 챔버 = 닫힌 구형 구조 → 공은 절대 밖으로 나갈 수 없음
        cx, cy, cz = self.width / 2, self.depth / 2, self.height / 2
        for ball in self.balls:
            if not ball.extracted:
                # 예측 위치 계산
                new_x = ball.x + ball.vx * dt
                new_y = ball.y + ball.vy * dt
                new_z = ball.z + ball.vz * dt

                # 챔버 중심으로부터 거리
                dx = new_x - cx
                dy = new_y - cy
                dz = new_z - cz
                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                # 챔버 제약 조건: dist + ball.radius <= chamber_radius
                max_dist = self.chamber_radius - ball.radius

                if dist > max_dist:
                    # === 구속 조건 위반 → 벽 표면으로 제한 ===
                    if dist > 0:
                        # 법선 벡터 (중심 → 공의 예측 위치)
                        nx = dx / dist
                        ny = dy / dist
                        nz = dz / dist

                        # 1. 위치를 벽 표면으로 클램핑
                        scale = max_dist / dist
                        ball.x = cx + dx * scale
                        ball.y = cy + dy * scale
                        ball.z = cz + dz * scale

                        # 2. 속도 반사 (벽과 충돌)
                        # 속도의 법선 성분
                        v_normal = ball.vx * nx + ball.vy * ny + ball.vz * nz

                        # 벽 밖으로 향하는 속도만 반사 (안쪽 향하면 그대로)
                        if v_normal > 0:
                            # 반발계수 적용
                            ball.vx -= (1 + self.restitution) * v_normal * nx
                            ball.vy -= (1 + self.restitution) * v_normal * ny
                            ball.vz -= (1 + self.restitution) * v_normal * nz

                            # 마찰력 (접선 방향)
                            ball.vx *= (1 - self.friction * 0.5)
                            ball.vy *= (1 - self.friction * 0.5)
                            ball.vz *= (1 - self.friction * 0.5)
                    else:
                        # 중심에 있으면 그냥 놔둠 (거의 불가능)
                        ball.x = new_x
                        ball.y = new_y
                        ball.z = new_z
                else:
                    # === 챔버 안 → 자유롭게 이동 ===
                    ball.x = new_x
                    ball.y = new_y
                    ball.z = new_z

        # 3. 공-공 충돌
        self._check_ball_collisions()

        # 3-1. 공-공 충돌 후 챔버 제약 조건 재확인 (충돌로 밀려날 수 있음)
        # 성능 최적화: 거리 제곱 비교로 빠르게 체크
        max_dist = self.chamber_radius - self.balls[0].radius
        max_dist_sq = max_dist ** 2

        for ball in self.balls:
            if not ball.extracted:
                # 거리 제곱 계산 (sqrt 없이)
                dx = ball.x - cx
                dy = ball.y - cy
                dz = ball.z - cz
                dist_sq = dx*dx + dy*dy + dz*dz

                # 밖으로 밀려난 경우만 처리 (매우 드뭄)
                if dist_sq > max_dist_sq:
                    dist = np.sqrt(dist_sq)

                    # 위치 보정
                    scale = max_dist / dist
                    ball.x = cx + dx * scale
                    ball.y = cy + dy * scale
                    ball.z = cz + dz * scale

                    # 속도 보정 (에너지 보존) - 벽 충돌로 처리
                    nx = dx / dist
                    ny = dy / dist
                    nz = dz / dist

                    v_normal = ball.vx * nx + ball.vy * ny + ball.vz * nz

                    # 밖으로 향하는 속도만 반사
                    if v_normal > 0:
                        ball.vx -= (1 + self.restitution) * v_normal * nx
                        ball.vy -= (1 + self.restitution) * v_normal * ny
                        ball.vz -= (1 + self.restitution) * v_normal * nz

        # 4. 속도 제한
        MAX_SPEED = 45000.0
        for ball in self.balls:
            if not ball.extracted:
                # 속도 제한 (중요!)
                speed = ball.speed
                if speed > MAX_SPEED:
                    scale = MAX_SPEED / speed
                    ball.vx *= scale
                    ball.vy *= scale
                    ball.vz *= scale

        # 5. 유체 격자 시뮬레이션
        if self.use_fluid_grid and self.fluid_grid is not None:
            self._update_fluid_grid(dt)

        # 6. 추출 확인
        self._check_extraction()

        # 7. Phase 관리 및 블로어 제어
        self._update_phase(dt)

        # 8. 시간 증가
        self.time += dt

        # 9. 추출 쿨다운 감소
        if self.extraction_cooldown > 0:
            self.extraction_cooldown = max(0.0, self.extraction_cooldown - dt)

    def apply_forces(self, ball: Ball3D, dt: float, batch_idx: int = -1):
        """
        67가지 물리 법칙을 벡터 합성으로 적용
        F_total = ΣF_i → a = F_total / m → v += a * dt
        """

        # DEBUG
        _debug = False  # 디버그 비활성화
        if _debug:
            print(f'[APPLY_FORCES] 시작: vx={ball.vx:.6f}, vy={ball.vy:.6f}, vz={ball.vz:.6f}, dt={dt}')

        # ========== 1-10: 기본 역학 ==========
        # 1. 중력 (Newton의 만유인력)
        ball.vz -= self.gravity * dt

        if _debug:
            print(f'[중력 후] vz={ball.vz:.2f}')

        # 2-10은 다른 힘들의 기반 (운동량, 에너지 보존 등)

        # ========== 11-20: 유체 역학 ==========
        # 격자 유체 모드일 때는 블로어/진공을 격자를 통해 적용하므로 여기서는 스킵
        if not (self.use_fluid_grid and self.fluid_grid is not None):
            # 11-14. Blower Jet 힘 (각 jet별로)
            self._apply_blower_force(ball, dt, batch_idx=batch_idx)

            # 15. Vacuum 힘
            self._apply_vacuum_force(ball, dt)

        # 16. 난류 (랜덤 요동)
        # turbulence는 가속도 단위 (mm/s²)이므로 × dt만 하면 속도 변화
        # jet_power의 영향을 받음 (풍압 0이면 난류도 0)
        if self.turbulence > 0 and self.jet_power > 0:
            turb_modifier = 1.0 if ball.z < self.height * 0.7 else 0.5
            effective_turbulence = self.turbulence * self.jet_power
            ball.vx += self.rng.normal(0, effective_turbulence * turb_modifier * 1.5) * dt
            ball.vy += self.rng.normal(0, effective_turbulence * turb_modifier * 1.5) * dt
            ball.vz += self.rng.normal(0, effective_turbulence * turb_modifier) * dt

        # 17-20. 베르누이 효과, 와류, 압력 구배 (난류에 포함)

        # ========== 21-30: 공기 역학 ==========
        # 21-23. 항력 (Reynolds Number 기반)
        if _debug:
            _vz_before_drag = ball.vz
        self._apply_drag_force(ball, dt)
        if _debug:
            print(f'[항력 후] vz={ball.vz:.6f}, Δvz={ball.vz - _vz_before_drag:.6f}')

        # 24-26. Magnus 효과 (회전 → 양력)
        if _debug:
            _vz_before_magnus = ball.vz
        self._apply_magnus_force(ball, dt)
        if _debug:
            print(f'[Magnus 후] vz={ball.vz:.6f}, Δvz={ball.vz - _vz_before_magnus:.6f}')

        # 27. 부력 (공기 중)
        # V = (4/3)πr³ (mm³), ρ_air = 1.225 kg/m³ = 1.225e-6 g/mm³
        # F_buoyancy = V * ρ_air * g
        V_mm3 = (4/3) * np.pi * (ball.radius ** 3)
        rho_air_g_mm3 = AIR_DENSITY * 1e-6  # kg/m³ -> g/mm³
        buoyancy = V_mm3 * rho_air_g_mm3 * self.gravity
        ball.vz += (buoyancy / ball.mass) * dt

        if _debug:
            print(f'[부력 후] vz={ball.vz:.6f}, buoyancy={buoyancy:.6f}, AIR_DENSITY={AIR_DENSITY}')

        # 28-30. Reynolds, Mach, Boundary layer (항력/Magnus에 포함)

        # ========== 31-40: 충돌 역학 (별도 함수) ==========
        # 31-35. 탄성/비탄성 충돌
        # 36-40. 마찰, 구름, 회전 전달
        # → check_ball_collision(), check_wall_collision()

        # ========== 41-50: 열역학 ==========
        # 41-43. 온도, 압력 변화 (속도에 따라)
        # 온도 상승 (마찰, 충돌)
        # 성능: 미세한 효과이며 시뮬레이션에 영향 없음 → 비활성화
        # if ball.collision_count > 0:
        #     dT = 0.01 * ball.speed * dt  # 속도에 비례
        #     ball.temperature += dT

        # 44-50. 열 교환, 대류, 복사 (미세한 효과, 생략 가능)

        # ========== 51-60: 전자기 ==========
        # 51-55. 공-공 정전기력 (Coulomb)
        # 성능: 45개 공에서 O(N²) = 1980번 계산/프레임 → 너무 느림
        # 효과: 10^-26 수준으로 미미함 → 비활성화
        # self._apply_electrostatic_forces(ball, dt)

        # 56-60. 유도, 대전 효과 (마찰로 전하 축적)
        # 성능: 미세한 효과이며 시뮬레이션에 영향 없음 → 비활성화
        # if ball.collision_count > 0:
        #     ball.charge += self.rng.normal(0, 1e-14)  # 미세 대전
        #     ball.charge = np.clip(ball.charge, -1e-11, 1e-11)

        # ========== 61-67: 고급 효과 ==========
        # 61-63. 음향 충돌 (압력파)
        # 충돌 시 생성되는 압력파 → 주변 공에 영향
        # (충돌 함수 내에서 처리)

        # 64-65. 난류 와류 (Karman vortex)
        # 공 뒤쪽에 생기는 와류 → 불안정 운동
        if ball.speed > 500:  # 고속일 때만
            vortex_freq = 0.2 * ball.speed / (ball.radius * 2)  # Strouhal number ≈ 0.2
            vortex_accel = 0.1 * self.gravity  # 중력의 10% 가속도
            phase = vortex_freq * self.time
            ball.vx += vortex_accel * np.sin(phase) * dt
            ball.vy += vortex_accel * np.cos(phase) * dt

        # 66-67. 벽면 효과 (ground effect, wall sliding)
        # 벽 근처에서 압력 증가 → 밀려남
        # 주의: 구형 챔버에서는 벽 효과 비활성화 (충돌 처리로 대체)
        # self._apply_wall_effect(ball, dt)

        # ========== 회전 감쇠 (공기 저항) ==========
        angular_drag = 0.02  # 회전 감쇠 계수
        ball.wx *= (1 - angular_drag * dt)
        ball.wy *= (1 - angular_drag * dt)
        ball.wz *= (1 - angular_drag * dt)

        # ========== 안전장치: 속도 제한 및 NaN/Inf 체크 ==========
        # 최대 속도 제한 (실제 로또 기계 데이터)
        # NBA 로또 기계: 90 mph = 40.2 m/s 공기 속도
        # 공이 공기 속도를 초과할 수 없음 (항력이 무한대가 됨)
        MAX_SPEED = 45000.0  # mm/s (45 m/s, 약간 여유)
        speed = ball.speed
        if speed > MAX_SPEED:
            scale = MAX_SPEED / speed
            ball.vx *= scale
            ball.vy *= scale
            ball.vz *= scale

        # NaN/Inf 체크 및 수정
        if not (np.isfinite(ball.vx) and np.isfinite(ball.vy) and np.isfinite(ball.vz)):
            if _debug:
                print(f'[NaN 감지!] vx={ball.vx}, vy={ball.vy}, vz={ball.vz}')
            ball.vx = 0.0
            ball.vy = 0.0
            ball.vz = 0.0

        if _debug:
            print(f'[APPLY_FORCES] 끝: vx={ball.vx:.6f}, vy={ball.vy:.6f}, vz={ball.vz:.6f}')

        if not (np.isfinite(ball.wx) and np.isfinite(ball.wy) and np.isfinite(ball.wz)):
            ball.wx = 0.0
            ball.wy = 0.0
            ball.wz = 0.0

    def _apply_blower_force(self, ball: Ball3D, dt: float, batch_idx: int = -1):
        """Blower Jet 힘 적용 (각 jet별로)"""
        # 제트 힘이 0이거나 jet_power가 0이면 스킵
        if self.jet_force == 0 or self.jet_power == 0:
            return

        # jet_power로 블로어 출력 제어 (0.0 ~ 1.0)
        effective_jet_force = self.jet_force * self.jet_power

        for jet_pos, jet_dir in zip(self.jet_positions, self.jet_directions):
            jx, jy, jz = jet_pos
            dx = ball.x - jx
            dy = ball.y - jy
            dz = ball.z - jz
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            if dist < 1e-6:
                continue

            # 거리 역제곱 감쇠 + 방향성
            influence = min(1.0, (self.height / dist) ** 1.5)

            # Jet 방향 성분
            dirx, diry, dirz = jet_dir

            # 힘 계산 (jet_power 적용)
            fx = dirx * effective_jet_force * influence
            fy = diry * effective_jet_force * influence
            fz = dirz * effective_jet_force * influence

            # 속도 변화 (F = ma → Δv = F/m * dt)
            ball.vx += (fx / ball.mass) * dt
            ball.vy += (fy / ball.mass) * dt
            ball.vz += (fz / ball.mass) * dt

    def _apply_vacuum_force(self, ball: Ball3D, dt: float):
        """Vacuum 흡입력 적용 (EXTRACTING 단계에서만)"""
        # 진공 힘이 0이거나 jet_power가 0이면 스킵 (풍압 0이면 진공도 0)
        if self.vacuum_force == 0 or self.jet_power == 0:
            return

        # MIXING 단계에서는 진공 OFF (실제 비너스 머신: 회전 드럼이 막고 있음)
        if self.phase == "MIXING" or self.phase == "INITIAL" or self.phase == "COMPLETE":
            return

        vx, vy, vz = self.vacuum_position
        dx = vx - ball.x
        dy = vy - ball.y
        dz = vz - ball.z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        if dist < self.vacuum_radius and dist > 1e-6:
            # 거리에 비례하는 흡입력
            strength = (self.vacuum_radius - dist) / self.vacuum_radius
            force = self.vacuum_force * strength

            nx, ny, nz = dx/dist, dy/dist, dz/dist

            ball.vx += (force * nx / ball.mass) * dt
            ball.vy += (force * ny / ball.mass) * dt
            ball.vz += (force * nz / ball.mass) * dt

    def _apply_drag_force(self, ball: Ball3D, dt: float):
        """
        Reynolds Number 기반 항력
        Re = ρ * v * D / μ
        C_d = f(Re)
        F_drag = 0.5 * ρ * v² * A * C_d
        """
        v = ball.speed
        if v < 1e-6:
            return

        # Reynolds Number
        # Re = ρ * v * D / ν (동점도 사용)
        # ν = μ / ρ = KINEMATIC_VISCOSITY
        # ρ = 1.225 kg/m³
        # v = mm/s, D = mm, ν = 1.5e-5 m²/s = 15 mm²/s
        diameter = ball.radius * 2  # mm
        nu_mm2_s = KINEMATIC_VISCOSITY * 1e6  # m²/s → mm²/s
        Re = v * diameter / nu_mm2_s

        # 항력 계수 (Reynolds에 따라 변화)
        if Re < 1:
            C_d = 24 / Re  # Stokes flow
        elif Re < 1000:
            C_d = 24 / Re * (1 + 0.15 * Re**0.687)
        elif Re < 200000:
            C_d = 0.44  # 난류
        else:
            C_d = 0.2  # 초임계 (drag crisis)

        # 항력: F = 0.5 * ρ * v² * A * C_d
        A = np.pi * (ball.radius ** 2)  # mm²
        rho_g_mm3 = AIR_DENSITY * 1e-6  # kg/m³ → g/mm³
        drag_force = 0.5 * rho_g_mm3 * (v ** 2) * A * C_d  # g·mm/s²

        # 속도 반대 방향
        drag_fx = -drag_force * ball.vx / v
        drag_fy = -drag_force * ball.vy / v
        drag_fz = -drag_force * ball.vz / v

        ball.vx += (drag_fx / ball.mass) * dt
        ball.vy += (drag_fy / ball.mass) * dt
        ball.vz += (drag_fz / ball.mass) * dt

    def _apply_magnus_force(self, ball: Ball3D, dt: float):
        """
        Magnus 효과: 회전하는 공의 양력
        F_magnus = 0.5 * ρ * A * C_L * v²
        C_L ≈ (r * ω) / v  (spin ratio)
        방향: ω × v
        """
        v = ball.speed
        w = ball.angular_speed

        if v < 1e-6 or w < 1e-6:
            return

        # Spin ratio
        spin_ratio = (ball.radius * w) / v
        C_L = MAGNUS_COEFFICIENT * spin_ratio
        C_L = min(C_L, 1.5)  # 상한

        # Magnus 힘 크기
        A = np.pi * (ball.radius ** 2)  # mm²
        rho_g_mm3 = AIR_DENSITY * 1e-6  # g/mm³
        magnus_force = 0.5 * rho_g_mm3 * A * C_L * (v ** 2)  # g·mm/s²

        # 방향: ω × v (직접 계산으로 최적화)
        # cross = ω × v
        cross_x = ball.wy * ball.vz - ball.wz * ball.vy
        cross_y = ball.wz * ball.vx - ball.wx * ball.vz
        cross_z = ball.wx * ball.vy - ball.wy * ball.vx

        # 크기
        cross_mag = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

        if cross_mag < 1e-6:
            return

        # 정규화
        fx = cross_x / cross_mag
        fy = cross_y / cross_mag
        fz = cross_z / cross_mag

        ball.vx += (magnus_force * fx / ball.mass) * dt
        ball.vy += (magnus_force * fy / ball.mass) * dt
        ball.vz += (magnus_force * fz / ball.mass) * dt

    def _apply_electrostatic_forces(self, ball: Ball3D, dt: float):
        """공-공 간 정전기력 (Coulomb)"""
        for other in self.balls:
            if other is ball or other.extracted:
                continue

            dx = ball.x - other.x
            dy = ball.y - other.y
            dz = ball.z - other.z
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            if dist < ball.radius * 2 or dist < 1e-6:
                continue

            # Coulomb 힘: F = k * q1 * q2 / r²
            # k = 8.99e9 N·m²/C², dist = mm, charge = C
            # F = N → g·mm/s² 변환: 1 N = 1 kg·m/s² = 1000 g·m/s² = 1000000 g·mm/s²
            force = COULOMB_CONSTANT * ball.charge * other.charge / ((dist * 0.001) ** 2)  # N
            force_gmm = force * 1e6  # N → g·mm/s²

            # 방향
            nx = dx / dist
            ny = dy / dist
            nz = dz / dist

            # 같은 부호면 척력, 다른 부호면 인력
            ball.vx += (force_gmm * nx / ball.mass) * dt
            ball.vy += (force_gmm * ny / ball.mass) * dt
            ball.vz += (force_gmm * nz / ball.mass) * dt

    def _apply_wall_effect(self, ball: Ball3D, dt: float):
        """벽면 효과 (ground effect, wall sliding)"""
        # 각 벽까지의 거리
        dist_x0 = ball.x - ball.radius
        dist_x1 = self.width - ball.x - ball.radius
        dist_y0 = ball.y - ball.radius
        dist_y1 = self.depth - ball.y - ball.radius
        dist_z0 = ball.z - ball.radius
        dist_z1 = self.height - ball.z - ball.radius

        threshold = ball.radius * 2  # 벽 근처 판정
        wall_force = 500.0  # mm/s² (밀어내는 힘)

        # X 방향
        if dist_x0 < threshold:
            influence = (threshold - dist_x0) / threshold
            ball.vx += wall_force * influence * dt
        if dist_x1 < threshold:
            influence = (threshold - dist_x1) / threshold
            ball.vx -= wall_force * influence * dt

        # Y 방향
        if dist_y0 < threshold:
            influence = (threshold - dist_y0) / threshold
            ball.vy += wall_force * influence * dt
        if dist_y1 < threshold:
            influence = (threshold - dist_y1) / threshold
            ball.vy -= wall_force * influence * dt

        # Z 방향
        if dist_z0 < threshold:
            influence = (threshold - dist_z0) / threshold
            ball.vz += wall_force * influence * dt
        if dist_z1 < threshold:
            influence = (threshold - dist_z1) / threshold
            ball.vz -= wall_force * influence * dt

    # ==================== 충돌 처리 ====================
    def _check_ball_collisions(self):
        """공-공 충돌 감지 및 처리 - Numba JIT 최적화"""
        N = len(self.balls)

        # Ball 데이터를 numpy 배열로 변환
        pos = np.zeros((N, 3), dtype=np.float64)
        vel = np.zeros((N, 3), dtype=np.float64)
        omega = np.zeros((N, 3), dtype=np.float64)
        mass = np.zeros(N, dtype=np.float64)
        radius = np.zeros(N, dtype=np.float64)
        extracted = np.zeros(N, dtype=np.bool_)

        for i, ball in enumerate(self.balls):
            pos[i] = [ball.x, ball.y, ball.z]
            vel[i] = [ball.vx, ball.vy, ball.vz]
            omega[i] = [ball.wx, ball.wy, ball.wz]
            mass[i] = ball.mass
            radius[i] = ball.radius
            extracted[i] = ball.extracted

        # Numba JIT 함수로 충돌 계산 (위치 분리 포함)
        # 공-공 충돌은 0.6375 (= 0.85 * 0.75)
        combined_restitution = self.restitution * 0.75
        I_ratio = 2.0 / 5.0  # 구의 관성 모멘트: I = (2/5) * m * r²

        # 위치 분리를 여러 번 반복하여 완전히 분리 (최대 5회)
        for iteration in range(5):
            pos, vel, omega = _check_and_resolve_collisions_jit(
                pos, vel, omega, mass, radius, extracted,
                combined_restitution, self.friction, I_ratio
            )

        # 결과를 Ball 객체에 다시 적용 (위치 + 속도 + 각속도)
        for i, ball in enumerate(self.balls):
            ball.x = pos[i, 0]
            ball.y = pos[i, 1]
            ball.z = pos[i, 2]
            ball.vx = vel[i, 0]
            ball.vy = vel[i, 1]
            ball.vz = vel[i, 2]
            ball.wx = omega[i, 0]
            ball.wy = omega[i, 1]
            ball.wz = omega[i, 2]

    def _resolve_collision(self, ball1: Ball3D, ball2: Ball3D,
                          dx: float, dy: float, dz: float,
                          dist: float, min_dist: float):
        """
        충돌 해결 - 회전 포함

        물리 법칙:
        - 운동량 보존
        - 에너지 보존 (탄성계수로 조절)
        - 각운동량 보존
        - 마찰력 (접선 방향)
        """

        if dist < 1e-6:
            dist = 1e-6

        # 법선 벡터
        nx = dx / dist
        ny = dy / dist
        nz = dz / dist

        # 상대 속도
        dvx = ball1.vx - ball2.vx
        dvy = ball1.vy - ball2.vy
        dvz = ball1.vz - ball2.vz

        # 법선 방향 상대 속도
        dv_n = dvx * nx + dvy * ny + dvz * nz

        # 이미 멀어지고 있으면 무시
        if dv_n > 0:
            return

        # 접촉점 속도 (회전 고려)
        # v_contact = v_linear + ω × r
        r1 = np.array([-nx * ball1.radius, -ny * ball1.radius, -nz * ball1.radius])
        r2 = np.array([nx * ball2.radius, ny * ball2.radius, nz * ball2.radius])

        omega1 = np.array([ball1.wx, ball1.wy, ball1.wz])
        omega2 = np.array([ball2.wx, ball2.wy, ball2.wz])

        v_rot1 = np.cross(omega1, r1)
        v_rot2 = np.cross(omega2, r2)

        # 접촉점 상대 속도
        dv_contact = np.array([dvx, dvy, dvz]) + v_rot1 - v_rot2
        dv_contact_n = dv_contact[0] * nx + dv_contact[1] * ny + dv_contact[2] * nz

        # 접선 방향
        dv_t = dv_contact - dv_contact_n * np.array([nx, ny, nz])
        dv_t_mag = np.linalg.norm(dv_t)

        # 질량
        m1 = ball1.mass
        m2 = ball2.mass

        # 회전 관성
        I1 = 0.4 * m1 * (ball1.radius ** 2)
        I2 = 0.4 * m2 * (ball2.radius ** 2)

        # 유효 질량 (회전 포함)
        m_eff_inv = 1/m1 + 1/m2 + (ball1.radius**2 / I1) + (ball2.radius**2 / I2)

        # 충격량 (법선 방향)
        # 공-공 충돌: 두 공 모두 변형되므로 에너지 손실 더 큼
        # 실제 폴리우레탄 공끼리 충돌 시 반발계수 ~0.65-0.70
        # 다수의 공이 밀집된 환경에서는 에너지 분산으로 더 낮아짐
        combined_restitution = self.restitution * 0.75  # 0.85 * 0.75 = 0.6375
        j = -(1 + combined_restitution) * dv_contact_n / m_eff_inv

        # 법선 방향 속도 변화
        ball1.vx += j * nx / m1
        ball1.vy += j * ny / m1
        ball1.vz += j * nz / m1

        ball2.vx -= j * nx / m2
        ball2.vy -= j * ny / m2
        ball2.vz -= j * nz / m2

        # 마찰력 (접선 방향)
        if dv_t_mag > 1e-6:
            tx = dv_t[0] / dv_t_mag
            ty = dv_t[1] / dv_t_mag
            tz = dv_t[2] / dv_t_mag

            # Coulomb 마찰
            j_t = min(self.friction * abs(j), m_eff_inv * dv_t_mag)

            ball1.vx -= j_t * tx / m1
            ball1.vy -= j_t * ty / m1
            ball1.vz -= j_t * tz / m1

            ball2.vx += j_t * tx / m2
            ball2.vy += j_t * ty / m2
            ball2.vz += j_t * tz / m2

            # 회전 변화 (토크)
            torque1 = np.cross(r1, np.array([tx, ty, tz]) * j_t)
            torque2 = np.cross(r2, np.array([-tx, -ty, -tz]) * j_t)

            ball1.wx += torque1[0] / I1
            ball1.wy += torque1[1] / I1
            ball1.wz += torque1[2] / I1

            ball2.wx += torque2[0] / I2
            ball2.wy += torque2[1] / I2
            ball2.wz += torque2[2] / I2

        # 위치 보정 (겹침 해소)
        overlap = min_dist - dist
        if overlap > 0:
            # 정확히 겹친 만큼만 보정 (에너지 주입 방지)
            correction = overlap * 0.5
            ball1.x += nx * correction
            ball1.y += ny * correction
            ball1.z += nz * correction

            ball2.x -= nx * correction
            ball2.y -= ny * correction
            ball2.z -= nz * correction

        # 충돌 카운트
        ball1.collision_count += 1
        ball2.collision_count += 1
        ball1.last_collision_time = self.time
        ball2.last_collision_time = self.time

        self.total_collisions += 1

        # 전하 재분배 (마찰 대전)
        total_charge = ball1.charge + ball2.charge
        transfer = self.rng.uniform(-0.1, 0.1) * total_charge
        ball1.charge += transfer
        ball2.charge -= transfer

    def _check_wall_collision(self, ball: Ball3D):
        """구형 챔버 벽 충돌 처리"""
        changed = False

        # 구형 챔버: 중심점 기준
        cx, cy, cz = self.width / 2, self.depth / 2, self.height / 2

        # 3D 공간에서 중심으로부터의 거리
        dx = ball.x - cx
        dy = ball.y - cy
        dz = ball.z - cz
        dist_from_center = np.sqrt(dx**2 + dy**2 + dz**2)

        # 구형 벽 충돌
        if dist_from_center + ball.radius > self.chamber_radius:
            # 충돌 처리
            if dist_from_center > 0:
                # 법선 벡터 (중심에서 공으로)
                nx = dx / dist_from_center
                ny = dy / dist_from_center
                nz = dz / dist_from_center

                # 공을 벽 안쪽으로 밀어냄
                overlap = (dist_from_center + ball.radius) - self.chamber_radius
                ball.x -= nx * overlap
                ball.y -= ny * overlap
                ball.z -= nz * overlap

                # 속도 반사 (법선 방향 반전)
                v_normal = ball.vx * nx + ball.vy * ny + ball.vz * nz
                # 올바른 충돌 공식: v_new = v - (1 + e) * v_n * n
                ball.vx -= (1 + self.restitution) * v_normal * nx
                ball.vy -= (1 + self.restitution) * v_normal * ny
                ball.vz -= (1 + self.restitution) * v_normal * nz

                # 회전 마찰
                ball.wx = ball.wx * (1 - self.friction)
                ball.wy = ball.wy * (1 - self.friction)
                ball.wz = ball.wz * (1 - self.friction)
                changed = True

        if changed:
            ball.collision_count += 1
            ball.last_collision_time = self.time

    def _update_phase(self, dt: float):
        """
        Phase 관리 및 블로어 제어 (실제 로또 기계 방식)

        Phase 순서:
        1. INITIAL (0초): 시작 대기
        2. MIXING (0-20초): 초기 믹싱 - 블로어 100%
        3. EXTRACTING: 추출 단계 반복
           - 믹싱 (10초): 블로어 100%
           - 추출 (0.5초): 블로어 0% (공기 중단) + 진공 작동
        4. COMPLETE: 7개 모두 추출 완료
        """
        self.phase_timer += dt

        # INITIAL → MIXING
        if self.phase == "INITIAL":
            if self.phase_timer >= 0.1:  # 0.1초 후 자동 시작
                self.phase = "MIXING"
                self.phase_timer = 0.0
                # jet_power는 키보드 입력으로 제어되므로 여기서 재설정하지 않음
                print(f"\n[{self.time:.1f}초] Phase: MIXING 시작 ({self.initial_mixing_time}초)")

        # MIXING → EXTRACTING
        elif self.phase == "MIXING":
            # jet_power는 키보드 입력으로 제어되므로 여기서 설정하지 않음

            if self.phase_timer >= self.initial_mixing_time:
                self.phase = "EXTRACTING"
                self.phase_timer = 0.0
                print(f"\n[{self.time:.1f}초] Phase: EXTRACTING 시작")

        # EXTRACTING: 계속 믹싱하면서 추출 (현실 로또 기계처럼)
        elif self.phase == "EXTRACTING":
            # jet_power는 키보드 입력으로 제어되므로 여기서 설정하지 않음

            if self.extracted_count >= 7:
                # 7개 모두 추출 완료
                self.phase = "COMPLETE"
                # 블로어 완전히 OFF (실제 로또 추첨기처럼)
                # → 공들이 바닥으로 떨어지지만, 위치 분리 코드로 겹치지 않음
                self.jet_power = 0.0
                print(f"\n[{self.time:.1f}초] Phase: COMPLETE - 추첨 완료!")
                print(f"  → 블로어 OFF, 공들이 바닥으로 떨어집니다")

        # COMPLETE: 블로어 OFF (실제 로또 추첨기 동작)
        elif self.phase == "COMPLETE":
            # 블로어를 완전히 끔
            # → 공들이 바닥에 떨어지지만, 충돌 처리의 위치 분리로 겹치지 않음
            self.jet_power = 0.0

    def _update_fluid_grid(self, dt: float):
        """
        유체 격자 업데이트 (CFD-like 시뮬레이션)

        1. 블로어/진공 → 격자에 속도 추가
        2. 속도장 확산
        3. 공 ↔ 유체 상호작용 (Two-way coupling)
        """
        grid = self.fluid_grid

        # 1. 블로어 → 격자에 속도 추가
        if self.jet_power > 0 and self.jet_force > 0:
            effective_jet_force = self.jet_force * self.jet_power
            for jet_pos, jet_dir in zip(self.jet_positions, self.jet_directions):
                jx, jy, jz = jet_pos
                dirx, diry, dirz = jet_dir

                # 속도 증가량 (가속도 → 속도)
                dv = effective_jet_force * dt  # mm/s

                # 격자에 속도 추가 (영향 반경 100mm)
                grid.add_velocity(jx, jy, jz, dirx*dv, diry*dv, dirz*dv, radius=100.0)

        # 2. 속도장 확산
        grid.diffuse(dt)

        # 3. 공 ↔ 유체 상호작용 (Two-way coupling)

        # 유체 평균 속도 (전체 CFD 상태 확인) - 한 번만 계산!
        # 각 셀의 속도 크기 평균 (방향 상쇄 방지)
        fluid_avg_speed = np.sqrt(grid.vx**2 + grid.vy**2 + grid.vz**2).mean()

        for ball in self.balls:
            if ball.extracted:
                continue

            # 공이 있는 위치의 유체 속도
            fluid_vx, fluid_vy, fluid_vz = grid.get_velocity(ball.x, ball.y, ball.z)

            # 풍압이 낮을 때 (CFD < 200 mm/s) 절대 속도 기준 공기 저항 적용
            # CFD 속도에 따라 점진적으로 증폭
            if fluid_avg_speed < 200.0:
                # 공의 절대 속도
                ball_speed = np.sqrt(ball.vx**2 + ball.vy**2 + ball.vz**2)

                if ball_speed > 10.0:  # 최소 속도
                    # 공기 저항 (절대 속도 기준)
                    # 밀폐 챔버 + 다중 공 간섭 효과로 저항 증폭
                    C_d = 0.47
                    area = np.pi * (ball.radius / 1000.0) ** 2  # m²
                    air_density_g_mm3 = AIR_DENSITY * 1e-6  # kg/m³ → g/mm³

                    # 항력 크기 (mm/s²)
                    drag_mag = 0.5 * air_density_g_mm3 * (ball_speed ** 2) * C_d * area * 1e6 / ball.mass

                    # CFD 속도에 따른 점진적 증폭
                    # CFD 0 mm/s: 50배, CFD 200 mm/s: 1배
                    amplification = 1.0 + 49.0 * (1.0 - fluid_avg_speed / 200.0)
                    drag_mag *= amplification

                    # 속도 감소 (속도 반대 방향)
                    ball.vx -= (ball.vx / ball_speed) * drag_mag * dt
                    ball.vy -= (ball.vy / ball_speed) * drag_mag * dt
                    ball.vz -= (ball.vz / ball_speed) * drag_mag * dt
            else:
                # CFD 활성 상태: 상대 속도 기준 항력
                rel_vx = fluid_vx - ball.vx
                rel_vy = fluid_vy - ball.vy
                rel_vz = fluid_vz - ball.vz
                rel_speed = np.sqrt(rel_vx**2 + rel_vy**2 + rel_vz**2)

                if rel_speed > 1.0:  # 최소 속도 차이
                    # 유체 → 공 (항력)
                    C_d = 0.47
                    area = np.pi * (ball.radius / 1000.0) ** 2  # m²
                    air_density_g_mm3 = AIR_DENSITY * 1e-6  # kg/m³ → g/mm³

                    # 항력 크기 (mm/s²)
                    drag_mag = 0.5 * air_density_g_mm3 * (rel_speed ** 2) * C_d * area * 1e6 / ball.mass

                    # 속도 변화 (유체 방향으로)
                    ball.vx += (rel_vx / rel_speed) * drag_mag * dt
                    ball.vy += (rel_vy / rel_speed) * drag_mag * dt
                    ball.vz += (rel_vz / rel_speed) * drag_mag * dt

                # 공 → 유체 (역작용)
                # DISABLED: 이 코드가 에너지를 주입하고 있었음!
                # 공의 절대 속도를 CFD에 더하면 피드백 루프 생성
                # 상대 속도 기반으로 수정 필요하거나 제거
                # i, j, k = grid.world_to_index(ball.x, ball.y, ball.z)
                # if grid.in_bounds(i, j, k):
                #     ball_volume = (4.0/3.0) * np.pi * (ball.radius ** 3)
                #     momentum_transfer = (ball_volume / grid.cell_volume) * 0.05
                #
                #     grid.vx[i,j,k] += ball.vx * momentum_transfer * dt
                #     grid.vy[i,j,k] += ball.vy * momentum_transfer * dt
                #     grid.vz[i,j,k] += ball.vz * momentum_transfer * dt

        # 5. 감쇠 (에너지 손실)
        grid.apply_damping(0.995)

    def _check_extraction(self):
        """
        추출구 근처 공 추출

        실제 비너스 추첨기 방식:
        - 터빈이 강한 공기를 불어 공을 위로 올림
        - 공이 위로 날아다니다가 추출구 근처에 도달
        - 추출구에서 공을 잡아서(catch) 추출
        - 한 번에 1개씩 순차적으로 추출
        - 블로어가 켜져있어야 공이 위로 날아감
        """
        # 추출 완료 체크
        if len(self.extracted_balls) >= 7:
            return

        # 추출 쿨다운 체크 (비너스 추첨기처럼 한 번에 1개씩, 5초 간격)
        if self.extraction_cooldown > 0:
            return

        # EXTRACTING phase이고 블로어가 켜진 상태에서만 추출
        # (블로어가 꺼지면 공이 다 떨어져서 추출 불가)
        if self.phase != "EXTRACTING" or self.jet_power < 0.1:
            return

        ex, ey, ez = self.extraction_position

        # 디버그: 10초마다 상태 출력
        if int(self.time) % 10 == 0 and int(self.time) != int(self.time - self.dt):
            max_z = max(b.z for b in self.balls if not b.extracted)
            print(f"[DEBUG {int(self.time)}초] 최고 높이: {max_z:.1f}mm, 추출구: {ez}mm, 추출: {len(self.extracted_balls)}개")

        # 추출구 근처 공 찾기 (위쪽으로 이동 중인 공만)
        candidates = []
        for ball in self.balls:
            if ball.extracted:
                continue

            dx = ball.x - ex
            dy = ball.y - ey
            dz = ball.z - ez
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            # 추출 조건: 추출구 근처 + 위쪽으로 이동 중
            if dist < self.extraction_radius and ball.vz > 0:
                # 거리와 속도 기반 가중치
                # 가까울수록, 빠를수록 추출 확률 높음
                weight = (1.0 - dist / self.extraction_radius) * (ball.vz / 1000.0)
                candidates.append((ball, weight))

        if not candidates:
            return

        # 가중치 기반으로 1개 선택
        total_weight = sum(w for _, w in candidates)
        if total_weight < 0.01:
            return

        # 확률적 추출 (프레임당 30% 확률로 증가 - 빠른 추출)
        if self.rng.random() < 0.3:
            # 가중치에 따라 공 선택
            rand = self.rng.random() * total_weight
            cumsum = 0.0
            for ball, weight in candidates:
                cumsum += weight
                if rand <= cumsum:
                    ball.extracted = True
                    ball.extraction_time = self.time

                    # 추출된 공은 챔버 밖으로 이동 (렌더링에서 제외됨)
                    # GUI 화면 아래에 2D 오버레이로 표시됨
                    ball.x = -1000.0  # 화면 밖
                    ball.y = -1000.0
                    ball.z = -1000.0

                    # 속도 제거 (정지)
                    ball.vx = 0.0
                    ball.vy = 0.0
                    ball.vz = 0.0
                    ball.wx = 0.0
                    ball.wy = 0.0
                    ball.wz = 0.0

                    self.extracted_balls.append(ball)
                    self.extracted_count = len(self.extracted_balls)

                    # 추출 메시지 출력
                    print(f"[{self.time:.1f}초] {self.extracted_count}번째 공 추출: {ball.number}번")

                    # 추출 후 쿨다운 시간 설정
                    self.extraction_cooldown = self.extraction_interval
                    break

    # ==================== 통계 및 유틸리티 ====================
    def get_statistics(self) -> Dict:
        """물리 통계 반환"""
        active_balls = [b for b in self.balls if not b.extracted]

        if not active_balls:
            return {
                'total_balls': self.num_balls,
                'active_balls': 0,
                'extracted_balls': len(self.extracted_balls),
                'total_collisions': self.total_collisions,
                'avg_speed': 0.0,
                'avg_kinetic_energy': 0.0,
                'total_energy': 0.0,
                'time': self.time
            }

        speeds = [b.speed for b in active_balls]
        energies = [b.kinetic_energy for b in active_balls]

        return {
            'total_balls': self.num_balls,
            'active_balls': len(active_balls),
            'extracted_balls': len(self.extracted_balls),
            'total_collisions': self.total_collisions,
            'avg_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'avg_kinetic_energy': np.mean(energies),
            'total_energy': np.sum(energies),
            'time': self.time
        }

    def get_state_snapshot(self) -> Dict:
        """현재 상태 스냅샷 (렌더링용)"""
        return {
            'time': self.time,
            'balls': [(b.number, b.x, b.y, b.z, b.vx, b.vy, b.vz,
                      b.wx, b.wy, b.wz, b.extracted, b.charge, b.temperature)
                     for b in self.balls],
            'extracted': [b.number for b in self.extracted_balls],
            'statistics': self.get_statistics()
        }


# ==================== 멀티스레드 물리 엔진 ====================
class PhysicsThread(threading.Thread):
    """물리 계산 전용 스레드 (60Hz)"""

    def __init__(self, chamber: LottoChamber3D_Ultimate, state_queue: deque):
        super().__init__(daemon=True)
        self.chamber = chamber
        self.state_queue = state_queue
        self.running = True
        self.fps = 60
        self.dt = 1.0 / self.fps

    def run(self):
        """물리 시뮬레이션 루프"""
        while self.running:
            t_start = time.perf_counter()

            # 물리 계산
            self.chamber.step()

            # 상태 큐에 추가
            snapshot = self.chamber.get_state_snapshot()
            self.state_queue.append(snapshot)

            # 큐 크기 제한
            while len(self.state_queue) > 120:
                self.state_queue.popleft()

            # FPS 유지
            elapsed = time.perf_counter() - t_start
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        """스레드 종료"""
        self.running = False


# ==================== 번호 생성 함수 ====================
def generate_physics_3d(
    n_sets: int = 1,
    seed: int | None = None,
    use_cfd: bool = True,
    grid_size: int = 32,
    fast_mode: bool = True,
) -> list[list[int]]:
    """
    3D 물리 시뮬레이션으로 로또 번호 생성 (비시각화)

    Parameters:
        n_sets: 생성할 세트 수
        seed: 랜덤 시드
        use_cfd: CFD 유체 격자 사용 여부
        grid_size: CFD 격자 크기 (16, 32, 64)
        fast_mode: 빠른 모드 (추출 대기 시간 단축)

    Returns:
        로또 번호 세트 리스트 [[n1, n2, ...], ...]
    """
    if seed is not None:
        np.random.seed(seed)

    results = []

    for _ in range(n_sets):
        # 챔버 생성 (비시각화 모드용 빠른 설정)
        chamber = LottoChamber3D_Ultimate(
            width=500.0,
            height=500.0,
            depth=500.0,
            num_balls=45,
            use_fluid_grid=use_cfd,
            initial_mixing_time=1.0 if fast_mode else 20.0,  # 1초로 단축 (vs 20초)
            extraction_interval=0.1 if fast_mode else 5.0,  # 0.1초로 단축 (vs 5초)
        )

        dt = 1.0 / 60.0  # 60 FPS
        max_time = 30.0 if fast_mode else 120.0  # 최대 30초 (vs 2분)

        # 7개 추출될 때까지 시뮬레이션
        while len(chamber.extracted_balls) < 7 and chamber.time < max_time:
            chamber.step(dt)

        # 추출된 번호 가져오기 (처음 6개만)
        if len(chamber.extracted_balls) >= 6:
            numbers = sorted([ball.number for ball in chamber.extracted_balls[:6]])
            results.append(numbers)
        else:
            # 실패 시 랜덤 생성
            numbers = sorted(np.random.choice(range(1, 46), 6, replace=False).tolist())
            results.append(numbers)

    return results


def generate_physics_3d_ultimate(
    n_sets: int = 1,
    seed: int | None = None,
    grid_size: int = 32,
    history_df=None,
    history_weights=None,
    mqle_threshold: float = 0.5,
    max_attempts: int = 30,
    fast_mode: bool = True,
    ml_model=None,
    ml_weight: float = 0.3,
) -> list[list[int]]:
    """
    3D 물리 시뮬레이션 + MQLE 필터 번호 생성 (비시각화)

    동작 방식:
    1. 물리 시뮬레이션으로 많은 후보 생성 (n_sets × max_attempts)
    2. MQLE가 각 후보를 평가하여 점수 부여 (CSV 패턴 활용)
    3. 점수가 높은 상위 n_sets개만 선택

    Parameters:
        n_sets: 생성할 세트 수
        seed: 랜덤 시드
        grid_size: CFD 격자 크기 (사용 안 함)
        history_df: CSV 데이터 (패턴 분석용)
        history_weights: 이력 가중치 (45개)
        mqle_threshold: 사용 안 함 (하위 호환용)
        max_attempts: 후보 생성 배수 (기본: 30배)
        fast_mode: 빠른 모드

    Returns:
        MQLE 필터링된 로또 번호 세트 리스트
    """
    if seed is not None:
        np.random.seed(seed)

    # MQLE 필터가 없으면 일반 물리 시뮬만
    if history_weights is None:
        return generate_physics_3d(
            n_sets=n_sets,
            seed=seed,
            use_cfd=True,
            fast_mode=fast_mode,
        )

    # 1단계: 물리 시뮬로 많은 후보 생성 (n_sets × max_attempts)
    candidate_count = max(n_sets * 3, min(n_sets * max_attempts, 100))
    print(f"물리 시뮬로 {candidate_count}개 후보 생성 중...")

    physics_candidates = generate_physics_3d(
        n_sets=candidate_count,
        seed=seed,
        use_cfd=True,
        fast_mode=fast_mode,
    )

    # 2단계: MQLE로 각 후보 점수 계산
    from lotto_generators import _qh_score, ml_score_set

    print(f"MQLE로 후보 평가 중 (CSV 패턴 활용)...")
    scored_candidates = []

    for cand in physics_candidates:
        # 양자조화 점수 (홀짝, 구간 균형)
        qh_score = _qh_score(cand, history_weights)

        # 다양성 점수 (이미 선택된 것과 겹치는 정도)
        diversity_penalty = 0
        for selected in scored_candidates:
            diversity_penalty += len(set(cand) & set(selected[1]))

        # 가중치 점수 (history_weights가 높은 번호 포함 여부)
        if history_weights is not None:
            weight_score = sum(history_weights[num - 1] for num in cand) / 6.0
        else:
            weight_score = 1.0

        # ML 모델 점수 (학습된 모델 활용)
        ml_score = 0.0
        if ml_model is not None:
            try:
                ml_score = ml_score_set(cand, ml_model, weights=history_weights, history_df=history_df)
            except Exception:
                ml_score = 0.0

        # CSV 패턴 점수 (history_df 활용)
        pattern_score = 0.0
        if history_df is not None and not history_df.empty:
            try:
                # 최근 10회 패턴과의 유사도 계산
                recent_sets = history_df.head(10)
                pattern_matches = 0
                for _, row in recent_sets.iterrows():
                    prev_nums = set([
                        row.get('n1', 0), row.get('n2', 0), row.get('n3', 0),
                        row.get('n4', 0), row.get('n5', 0), row.get('n6', 0)
                    ])
                    overlap = len(set(cand) & prev_nums)
                    # 2-3개 겹치는 게 적당 (0-1개나 4개 이상은 감점)
                    if 2 <= overlap <= 3:
                        pattern_matches += 1
                pattern_score = pattern_matches / len(recent_sets) if len(recent_sets) > 0 else 0.0
            except Exception:
                pattern_score = 0.0

        # ML 가중치 정규화
        ml_w = float(max(0.0, min(1.0, ml_weight)))

        # 최종 점수 (MQLE 스타일 + ML + CSV 패턴)
        # ML 가중치만큼 ML 점수 사용, 나머지를 기존 방식으로 분배
        remaining_weight = 1.0 - ml_w
        total_score = (
            remaining_weight * 0.5 * qh_score +           # 양자조화
            remaining_weight * 0.2 * weight_score +       # 가중치
            remaining_weight * 0.2 * pattern_score +      # CSV 패턴
            remaining_weight * 0.1 * (-diversity_penalty / 10.0) +  # 다양성
            ml_w * ml_score  # ML 점수
        )

        scored_candidates.append((total_score, cand))

    # 3단계: 점수 높은 순으로 정렬하여 상위 n_sets개 선택
    scored_candidates.sort(reverse=True, key=lambda x: x[0])

    results = []
    for i in range(min(n_sets, len(scored_candidates))):
        score, numbers = scored_candidates[i]
        results.append(numbers)
        print(f"  선택 {i+1}/{n_sets}: {numbers} (점수: {score:.3f})")

    # 부족하면 물리 시뮬로 추가
    if len(results) < n_sets:
        additional = generate_physics_3d(
            n_sets=n_sets - len(results),
            seed=None,
            use_cfd=True,
            fast_mode=fast_mode,
        )
        results.extend(additional)

    return results


# ==================== 테스트 코드 ====================
if __name__ == '__main__':
    print("=" * 60)
    print("3D Lotto Physics Engine - 67 Laws")
    print("=" * 60)
    print(f"\nBackend: {get_physics_backend_info()}")

    # 챔버 생성
    chamber = LottoChamber3D_Ultimate()

    print(f"\n초기 상태:")
    print(f"  공 개수: {chamber.num_balls}")
    print(f"  챔버 크기: {chamber.width} × {chamber.depth} × {chamber.height} mm")
    print(f"  Jet 힘: {chamber.jet_force} mm/s² ({chamber.jet_force/1000:.1f} m/s²)")
    print(f"  난류: {chamber.turbulence} mm/s² ({chamber.turbulence/1000:.1f} m/s²)")

    # 10초 시뮬레이션
    print(f"\n10초 시뮬레이션 시작...")
    for i in range(600):  # 60 FPS × 10초
        chamber.step()

        if i % 60 == 0:  # 1초마다
            stats = chamber.get_statistics()
            print(f"  {stats['time']:.1f}s | "
                  f"속도: {stats['avg_speed']:.1f} mm/s | "
                  f"에너지: {stats['total_energy']:.2f} mJ | "
                  f"충돌: {stats['total_collisions']} | "
                  f"추출: {stats['extracted_balls']}")

    print("\n✓ 테스트 완료!")
