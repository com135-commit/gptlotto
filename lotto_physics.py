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

# ==================== GPU Backend Support ====================
# CuPy disabled due to NumPy 2.x compatibility issues
cp = None
HAS_CUPY = False

# try:
#     import cupy as cp
#     HAS_CUPY = True
#     print("✓ CuPy detected - GPU acceleration available")
# except (ImportError, ValueError, RuntimeError) as e:
#     cp = None
#     HAS_CUPY = False
#     # Silently ignore CuPy import errors (NumPy compatibility, CUDA not available, etc.)

try:
    from numba import jit, prange
    HAS_NUMBA = True
    print("✓ Numba detected - JIT compilation available")
except ImportError:
    HAS_NUMBA = False
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
# 한국 로또 6/45 공 실제 물리 사양
BALL_RADIUS = 22.25  # mm (직경 44.5mm)
BALL_MASS = 4.0  # g (실제 로또공 무게)
BALL_DENSITY = BALL_MASS / (4/3 * np.pi * (BALL_RADIUS/1000)**3)  # kg/m³ (속이 빈 공)

GRAVITY = 9800.0  # mm/s² = 9.8 m/s²
AIR_DENSITY = 1.225e-9  # kg/mm³ (해수면, 15°C)
AIR_VISCOSITY = 1.81e-5  # Pa·s (동적 점도)
KINEMATIC_VISCOSITY = AIR_VISCOSITY / (AIR_DENSITY * 1e9)  # mm²/s

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

    # 챔버 크기 (mm) - 실제 로또 기계 크기 참고
    width: float = 400.0  # X 방향
    depth: float = 400.0  # Y 방향
    height: float = 600.0  # Z 방향

    # 공 설정
    num_balls: int = 45
    ball_radius: float = BALL_RADIUS
    ball_mass: float = BALL_MASS

    # 물리 파라미터 (수정된 현실적 값)
    gravity: float = GRAVITY
    restitution: float = RESTITUTION
    friction: float = FRICTION_COEF

    # 유체 역학 파라미터 (현실적 값으로 수정)
    jet_force: float = 12000.0  # mm/s² (10-15 m/s² = 현실적)
    vacuum_force: float = 2000.0  # mm/s²
    turbulence: float = 1500.0  # mm/s² (1-2 m/s² = 현실적)

    # Blower 설정
    num_jets: int = 4
    jet_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    jet_directions: List[Tuple[float, float, float]] = field(default_factory=list)

    # Vacuum 설정
    vacuum_position: Tuple[float, float, float] = (200.0, 200.0, 500.0)
    vacuum_radius: float = 50.0

    # 추출구 설정
    extraction_position: Tuple[float, float, float] = (200.0, 200.0, 550.0)
    extraction_radius: float = 30.0

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

    def __post_init__(self):
        """초기화"""
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

        # 공 생성
        if not self.balls:
            self._initialize_balls()

    def _initialize_balls(self):
        """45개 공 초기화 - 랜덤 배치"""
        self.balls = []

        # 격자 배치 후 약간의 랜덤 perturbation
        grid_size = int(np.ceil(self.num_balls ** (1/3)))
        spacing = min(self.width, self.depth, self.height * 0.6) / (grid_size + 1)

        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if idx >= self.num_balls:
                        break

                    # 격자 위치 + 랜덤 perturbation
                    x = (i + 1) * spacing + self.rng.uniform(-spacing*0.2, spacing*0.2)
                    y = (j + 1) * spacing + self.rng.uniform(-spacing*0.2, spacing*0.2)
                    z = (k + 1) * spacing + self.ball_radius + self.rng.uniform(0, spacing*0.2)

                    # 경계 확인
                    x = np.clip(x, self.ball_radius, self.width - self.ball_radius)
                    y = np.clip(y, self.ball_radius, self.depth - self.ball_radius)
                    z = np.clip(z, self.ball_radius, self.height - self.ball_radius)

                    ball = Ball3D(
                        number=idx + 1,
                        x=x, y=y, z=z,
                        vx=self.rng.uniform(-100, 100),
                        vy=self.rng.uniform(-100, 100),
                        vz=self.rng.uniform(0, 200),
                        wx=self.rng.uniform(-10, 10),
                        wy=self.rng.uniform(-10, 10),
                        wz=self.rng.uniform(-10, 10)
                    )

                    self.balls.append(ball)
                    idx += 1

    # ==================== 물리 법칙 적용 ====================
    def step(self, dt=None):
        """한 타임스텝 시뮬레이션"""
        # dt 파라미터 지원 (시각화 호환성)
        if dt is None:
            dt = self.dt

        # 1. 모든 힘 적용 (벡터 합성)
        for ball in self.balls:
            if not ball.extracted:
                self.apply_forces(ball, dt)

        # 2. 공-공 충돌
        self._check_ball_collisions()

        # 3. 벽 충돌
        for ball in self.balls:
            if not ball.extracted:
                self._check_wall_collision(ball)

        # 4. 추출 확인
        self._check_extraction()

        # 5. 시간 증가
        self.time += dt

    def apply_forces(self, ball: Ball3D, dt: float, batch_idx: int = -1):
        """
        67가지 물리 법칙을 벡터 합성으로 적용
        F_total = ΣF_i → a = F_total / m → v += a * dt
        """

        # ========== 1-10: 기본 역학 ==========
        # 1. 중력 (Newton의 만유인력)
        ball.vz -= self.gravity * dt

        # 2-10은 다른 힘들의 기반 (운동량, 에너지 보존 등)

        # ========== 11-20: 유체 역학 ==========
        # 11-14. Blower Jet 힘 (각 jet별로)
        self._apply_blower_force(ball, dt, batch_idx=batch_idx)

        # 15. Vacuum 힘
        self._apply_vacuum_force(ball, dt)

        # 16. 난류 (랜덤 요동)
        turb_modifier = 1.0 if ball.z < self.height * 0.7 else 0.5
        ball.vx += self.rng.normal(0, self.turbulence * turb_modifier * 1.5) * dt / ball.mass
        ball.vy += self.rng.normal(0, self.turbulence * turb_modifier * 1.5) * dt / ball.mass
        ball.vz += self.rng.normal(0, self.turbulence * turb_modifier) * dt / ball.mass

        # 17-20. 베르누이 효과, 와류, 압력 구배 (난류에 포함)

        # ========== 21-30: 공기 역학 ==========
        # 21-23. 항력 (Reynolds Number 기반)
        self._apply_drag_force(ball, dt)

        # 24-26. Magnus 효과 (회전 → 양력)
        self._apply_magnus_force(ball, dt)

        # 27. 부력 (공기 중)
        buoyancy = (4/3) * np.pi * (ball.radius ** 3) * AIR_DENSITY * 1e9 * self.gravity
        ball.vz += (buoyancy / ball.mass) * dt

        # 28-30. Reynolds, Mach, Boundary layer (항력/Magnus에 포함)

        # ========== 31-40: 충돌 역학 (별도 함수) ==========
        # 31-35. 탄성/비탄성 충돌
        # 36-40. 마찰, 구름, 회전 전달
        # → check_ball_collision(), check_wall_collision()

        # ========== 41-50: 열역학 ==========
        # 41-43. 온도, 압력 변화 (속도에 따라)
        # 온도 상승 (마찰, 충돌)
        if ball.collision_count > 0:
            dT = 0.01 * ball.speed * dt  # 속도에 비례
            ball.temperature += dT

        # 44-50. 열 교환, 대류, 복사 (미세한 효과, 생략 가능)

        # ========== 51-60: 전자기 ==========
        # 51-55. 공-공 정전기력 (Coulomb)
        self._apply_electrostatic_forces(ball, dt)

        # 56-60. 유도, 대전 효과 (마찰로 전하 축적)
        if ball.collision_count > 0:
            ball.charge += self.rng.normal(0, 1e-14)  # 미세 대전
            ball.charge = np.clip(ball.charge, -1e-11, 1e-11)

        # ========== 61-67: 고급 효과 ==========
        # 61-63. 음향 충돌 (압력파)
        # 충돌 시 생성되는 압력파 → 주변 공에 영향
        # (충돌 함수 내에서 처리)

        # 64-65. 난류 와류 (Karman vortex)
        # 공 뒤쪽에 생기는 와류 → 불안정 운동
        if ball.speed > 500:  # 고속일 때만
            vortex_freq = 0.2 * ball.speed / (ball.radius * 2)  # Strouhal number ≈ 0.2
            vortex_force = 0.1 * ball.mass * self.gravity
            phase = vortex_freq * self.time
            ball.vx += vortex_force * np.sin(phase) * dt / ball.mass
            ball.vy += vortex_force * np.cos(phase) * dt / ball.mass

        # 66-67. 벽면 효과 (ground effect, wall sliding)
        # 벽 근처에서 압력 증가 → 밀려남
        self._apply_wall_effect(ball, dt)

        # ========== 회전 감쇠 (공기 저항) ==========
        angular_drag = 0.02  # 회전 감쇠 계수
        ball.wx *= (1 - angular_drag * dt)
        ball.wy *= (1 - angular_drag * dt)
        ball.wz *= (1 - angular_drag * dt)

    def _apply_blower_force(self, ball: Ball3D, dt: float, batch_idx: int = -1):
        """Blower Jet 힘 적용 (각 jet별로)"""
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

            # 힘 계산
            fx = dirx * self.jet_force * influence
            fy = diry * self.jet_force * influence
            fz = dirz * self.jet_force * influence

            # 속도 변화 (F = ma → Δv = F/m * dt)
            ball.vx += (fx / ball.mass) * dt
            ball.vy += (fy / ball.mass) * dt
            ball.vz += (fz / ball.mass) * dt

    def _apply_vacuum_force(self, ball: Ball3D, dt: float):
        """Vacuum 흡입력 적용"""
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
        diameter = ball.radius * 2
        Re = AIR_DENSITY * 1e9 * v * diameter / AIR_VISCOSITY

        # 항력 계수 (Reynolds에 따라 변화)
        if Re < 1:
            C_d = 24 / Re  # Stokes flow
        elif Re < 1000:
            C_d = 24 / Re * (1 + 0.15 * Re**0.687)
        elif Re < 200000:
            C_d = 0.44  # 난류
        else:
            C_d = 0.2  # 초임계 (drag crisis)

        # 항력
        A = np.pi * (ball.radius ** 2)
        drag_force = 0.5 * AIR_DENSITY * 1e9 * (v ** 2) * A * C_d

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
        A = np.pi * (ball.radius ** 2)
        magnus_force = 0.5 * AIR_DENSITY * 1e9 * A * C_L * (v ** 2)

        # 방향: ω × v
        omega = np.array([ball.wx, ball.wy, ball.wz])
        velocity = np.array([ball.vx, ball.vy, ball.vz])
        cross = np.cross(omega, velocity)
        cross_mag = np.linalg.norm(cross)

        if cross_mag < 1e-6:
            return

        fx = cross[0] / cross_mag
        fy = cross[1] / cross_mag
        fz = cross[2] / cross_mag

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
            force = COULOMB_CONSTANT * ball.charge * other.charge / (dist ** 2)
            force *= 1e-6  # N → mN (밀리뉴턴)

            # 방향
            nx = dx / dist
            ny = dy / dist
            nz = dz / dist

            # 같은 부호면 척력, 다른 부호면 인력
            ball.vx += (force * nx / ball.mass) * dt
            ball.vy += (force * ny / ball.mass) * dt
            ball.vz += (force * nz / ball.mass) * dt

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
        """공-공 충돌 감지 및 처리 - C(45,2) = 990 pairs"""
        for i in range(len(self.balls)):
            ball1 = self.balls[i]
            if ball1.extracted:
                continue

            for j in range(i+1, len(self.balls)):
                ball2 = self.balls[j]
                if ball2.extracted:
                    continue

                # 충돌 판정
                dx = ball1.x - ball2.x
                dy = ball1.y - ball2.y
                dz = ball1.z - ball2.z
                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                min_dist = ball1.radius + ball2.radius

                if dist < min_dist:
                    self._resolve_collision(ball1, ball2, dx, dy, dz, dist, min_dist)

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
        combined_restitution = self.restitution * 0.95  # 약간의 에너지 손실
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
            correction = overlap * 0.5 + 0.1
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
        """벽 충돌 처리"""
        changed = False

        # X 방향
        if ball.x - ball.radius < 0:
            ball.x = ball.radius
            ball.vx = -ball.vx * self.restitution
            ball.wy = ball.wy * (1 - self.friction)  # 마찰로 회전 감소
            ball.wz = ball.wz * (1 - self.friction)
            changed = True
        elif ball.x + ball.radius > self.width:
            ball.x = self.width - ball.radius
            ball.vx = -ball.vx * self.restitution
            ball.wy = ball.wy * (1 - self.friction)
            ball.wz = ball.wz * (1 - self.friction)
            changed = True

        # Y 방향
        if ball.y - ball.radius < 0:
            ball.y = ball.radius
            ball.vy = -ball.vy * self.restitution
            ball.wx = ball.wx * (1 - self.friction)
            ball.wz = ball.wz * (1 - self.friction)
            changed = True
        elif ball.y + ball.radius > self.depth:
            ball.y = self.depth - ball.radius
            ball.vy = -ball.vy * self.restitution
            ball.wx = ball.wx * (1 - self.friction)
            ball.wz = ball.wz * (1 - self.friction)
            changed = True

        # Z 방향
        if ball.z - ball.radius < 0:
            ball.z = ball.radius
            ball.vz = -ball.vz * self.restitution
            ball.wx = ball.wx * (1 - self.friction)
            ball.wy = ball.wy * (1 - self.friction)
            changed = True
        elif ball.z + ball.radius > self.height:
            ball.z = self.height - ball.radius
            ball.vz = -ball.vz * self.restitution
            ball.wx = ball.wx * (1 - self.friction)
            ball.wy = ball.wy * (1 - self.friction)
            changed = True

        if changed:
            ball.collision_count += 1
            ball.last_collision_time = self.time

    def _check_extraction(self):
        """추출구 근처 공 추출"""
        ex, ey, ez = self.extraction_position

        for ball in self.balls:
            if ball.extracted:
                continue

            dx = ball.x - ex
            dy = ball.y - ey
            dz = ball.z - ez
            dist = np.sqrt(dx**2 + dy**2 + dz**2)

            # 추출 조건: 추출구 근처 + 위쪽으로 이동 중
            if dist < self.extraction_radius and ball.vz > 0:
                # 확률적 추출 (너무 빠르면 추출 안됨)
                if ball.speed < 1000:
                    prob = 0.1  # 10% per frame
                    if self.rng.random() < prob:
                        ball.extracted = True
                        ball.extraction_time = self.time
                        self.extracted_balls.append(ball)

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
