"""
3D Physics Visualizer for Lotto Drawing Machine
실시간 3D 물리 시뮬레이션 시각화 (PyGame + OpenGL)
"""

import sys
import numpy as np
import ctypes

# pygame 체크
try:
    import pygame
    from pygame.locals import *
except ImportError:
    print("=" * 60)
    print("오류: pygame이 설치되지 않았습니다.")
    print("=" * 60)
    print("\n설치 방법:")
    print("  1. install_pygame.bat 파일을 더블클릭")
    print("  2. 또는 명령 프롬프트에서:")
    print("     C:\\ProgramData\\anaconda3\\python.exe -m pip install pygame")
    print("=" * 60)
    sys.exit(1)

# PyOpenGL 체크
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print("=" * 60)
    print("오류: PyOpenGL이 설치되지 않았습니다.")
    print("=" * 60)
    print("\n설치 방법:")
    print("  1. install_pygame.bat 파일을 더블클릭")
    print("  2. 또는 명령 프롬프트에서:")
    print("     C:\\ProgramData\\anaconda3\\python.exe -m pip install PyOpenGL")
    print("=" * 60)
    sys.exit(1)

import threading
import time
from queue import Queue
from collections import deque
import importlib

# lotto_physics 모듈 강제 리로드 (코드 변경 시 즉시 반영)
import lotto_physics
importlib.reload(lotto_physics)
from lotto_physics import LottoChamber3D_Ultimate

# 전역 시각화 인스턴스 관리
_active_visualizers = []
_visualizers_lock = threading.Lock()


class PhysicsThread(threading.Thread):
    """물리 엔진 전용 스레드 (정확히 60Hz 유지)"""

    def __init__(self, engine):
        super().__init__(daemon=True)
        self.engine = engine
        self.running = True
        self.paused = True  # 시작 시 일시정지 상태
        self.lock = threading.Lock()

        # 상태 큐 (최대 3개 버퍼 - 보간을 위해)
        self.state_queue = deque(maxlen=3)

        # 물리 시뮬레이션 설정
        self.physics_hz = 60
        self.dt = 1.0 / self.physics_hz

        # 성능 측정
        self.actual_hz = 60.0
        self.frame_times = deque(maxlen=60)

        # 시뮬레이션 시간 (자체 추적)
        self.simulation_time = 0.0

    def run(self):
        """물리 시뮬레이션 메인 루프 (60Hz 고정)"""
        last_time = time.time()

        while self.running:
            loop_start = time.time()

            # 일시정지 상태 체크
            if not self.paused:
                # 물리 계산
                self.engine.step(self.dt)

                # 시뮬레이션 시간 증가
                self.simulation_time += self.dt

            # 현재 상태를 렌더러에 전달 (일시정지 중에도 렌더링 위해)
            state = self.get_state()
            self.state_queue.append(state)

            # 프레임 시간 기록
            frame_time = time.time() - loop_start
            self.frame_times.append(frame_time)

            # 실제 Hz 계산
            if len(self.frame_times) >= 10:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                if avg_frame_time > 0:
                    self.actual_hz = 1.0 / avg_frame_time

            # 정확히 60Hz 유지
            elapsed = time.time() - loop_start
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_state(self):
        """렌더링용 상태 스냅샷 (Thread-Safe) - 최적화됨"""
        with self.lock:
            # 최적화: 리스트 컴프리헨션 + 튜플 사용 (딕셔너리보다 빠름)
            balls_data = [
                (
                    b.x, b.y, b.z,  # 위치
                    b.vx, b.vy, b.vz,  # 속도
                    b.wx, b.wy, b.wz,  # 회전
                    (b.vx**2 + b.vy**2 + b.vz**2) ** 0.5,  # speed
                    (b.wx**2 + b.wy**2 + b.wz**2) ** 0.5,  # spin_speed
                    not b.extracted,  # active
                    b.number,  # number
                    b.mass  # mass
                )
                for b in self.engine.balls
            ]

            return {
                'balls': balls_data,
                'phase': self.engine.phase,
                'simulation_time': self.simulation_time,
                'jet_power': self.engine.jet_power,
                'extracted': [b.number for b in self.engine.extracted_balls],
                'chamber_radius': self.engine.chamber_radius
            }

    def get_latest_state(self, wait=False, timeout=0.1):
        """최신 물리 상태 가져오기

        Args:
            wait: True이면 새 상태가 나올 때까지 대기
            timeout: 최대 대기 시간 (초)
        """
        if not wait:
            # 기존 동작: 큐에 있으면 반환, 없으면 None
            if self.state_queue:
                return self.state_queue[-1]
            return None

        # 대기 모드: 새 상태가 나올 때까지 대기
        import time
        start_time = time.time()
        last_state = self.state_queue[-1] if self.state_queue else None

        while time.time() - start_time < timeout:
            if self.state_queue:
                current_state = self.state_queue[-1]
                # 새 상태인지 확인 (simulation_time으로 비교)
                if last_state is None or current_state['simulation_time'] != last_state['simulation_time']:
                    return current_state
            time.sleep(0.001)  # 1ms 대기

        # 타임아웃: 마지막 상태 반환
        return self.state_queue[-1] if self.state_queue else None

    def stop(self):
        """스레드 종료"""
        self.running = False


class PhysicsVisualizer3D:
    """3D 실시간 물리 시뮬레이션 시각화"""

    def __init__(self, num_balls=45, mode="물리시뮬3D"):
        self.num_balls = num_balls
        self.mode = mode
        self.running = False
        self.paused = True  # 시작 시 일시정지 상태 (SPACE로 시작)

        # 물리 엔진 (LottoChamber3D_Ultimate 사용 - 67가지 물리 현상)
        # 매번 다른 결과 생성 (랜덤 시드는 내부에서 자동 생성)
        # ★ CFD(격자 유체) ON으로 물리 엔진 생성
        self.engine = LottoChamber3D_Ultimate(use_fluid_grid=True)
        # initialize_balls()는 LottoChamber3D_Ultimate.__init__에서 자동 호출됨

        # 추출 설정 강제 적용 (모듈 캐싱 문제 방지)
        print("=" * 70)
        print("[시각화 초기화]")
        print(f"  CFD(격자 유체): {'ON' if self.engine.use_fluid_grid else 'OFF'}")
        if self.engine.use_fluid_grid:
            grid = self.engine.fluid_grid
            print(f"  격자 해상도: {grid.nx}x{grid.ny}x{grid.nz} = {grid.nx * grid.ny * grid.nz} cells")
        print(f"  추출구 위치: {self.engine.extraction_position}")
        print(f"  추출구 반지름: {self.engine.extraction_radius} mm")
        print(f"  진공 위치: {self.engine.vacuum_position}")
        print(f"  진공 힘: {self.engine.vacuum_force} mm/s²")
        print(f"  제트 힘: {self.engine.jet_force} mm/s²")

        # 추출 함수가 올바른지 확인
        import inspect
        check_func = inspect.getsource(self.engine._check_extraction)
        if "프레임당 30% 확률" in check_func:
            print("  추출 함수: 최신 버전 (30% 확률) OK")
        elif "프레임당 5% 확률" in check_func:
            print("  추출 함수: 구버전 (5% 확률) - 업데이트 필요!")
        else:
            print("  추출 함수: 알 수 없는 버전")

        # 공 초기 위치 확인
        import numpy as np
        z_positions = [b.z for b in self.engine.balls]
        print(f"\n  초기 공 위치:")
        print(f"    최저: {min(z_positions):.1f}mm")
        print(f"    최고: {max(z_positions):.1f}mm")
        print(f"    평균: {np.mean(z_positions):.1f}mm")
        if min(z_positions) < 50:
            print(f"    [OK] 바닥부터 시작 (최저 {min(z_positions):.1f}mm)")
        else:
            print(f"    [WARN] 너무 높음 (최저 {min(z_positions):.1f}mm)")
        print("=" * 70)

        # 카메라 설정 (정면에서 약간 위에서 내려다보기)
        self.camera_distance = 1400.0  # 최적 거리: 챔버 전체가 한눈에 보임
        self.camera_rotation_x = 30.0  # 위에서 내려다보는 각도 (30도)
        self.camera_rotation_y = 45.0  # 측면에서 보는 각도 (45도)
        self.camera_target = [250, 250, 250]  # 챔버 중심 (물리 엔진 좌표)

        print(f"\n카메라 설정:")
        print(f"  거리: {self.camera_distance}mm")
        print(f"  각도: X={self.camera_rotation_x}°, Y={self.camera_rotation_y}°")
        print(f"  타겟: {self.camera_target}")
        print(f"  → 챔버 전체(z=0~500mm)가 보입니다\n")

        # 마우스 상태
        self.mouse_down = False
        self.last_mouse_pos = (0, 0)

        # 시뮬레이션 상태
        self.selected_balls = []
        self.simulation_time = 0.0
        self.fps = 60  # ★ 물리와 동일한 FPS로 변경 (끊김 방지)
        self.current_fps = 60.0
        self.draw_complete = False  # 추첨 완료 여부
        self.completion_time = None  # 추첨 완료 시간
        self.auto_pause_message_shown = False  # 자동 정지 메시지 출력 여부

        # 텍스트 렌더링 최적화
        self.last_text_update = 0.0  # 마지막 텍스트 업데이트 시간
        self.text_update_interval = 0.1  # 텍스트 업데이트 간격 (0.1초)
        self.cached_text_image = None  # 캐시된 텍스트 이미지

        # 색상 (1-45번 공)
        self.ball_colors = self._generate_ball_colors()

        # 폰트 초기화는 init_display에서

        # 공기력 저장 (추첨 완료 후 복구용)
        self.original_jet_force = self.engine.jet_force

        # 풍압 조절 설정
        self.jet_force_multiplier = 1.0  # 풍압 배율 (0.5 ~ 2.0)
        self.jet_force_step = 0.05  # 한 번에 조절되는 비율 (5%)
        self.engine.jet_power = 1.0  # 풍량도 초기화


        # 모션 블러용 트레일 (최근 3프레임 위치 저장)
        self.ball_trails = [[] for _ in range(45)]  # 각 공의 최근 위치 기록

        # 공 회전 누적 (각 공마다 독립적인 회전 상태)
        # 각 공: [angle, axis_x, axis_y, axis_z]
        self.ball_rotations = [[0.0, 1.0, 0.0, 0.0] for _ in range(45)]  # 초기: 0도 회전

        # === 멀티스레드: 물리 엔진 스레드 생성 ===
        self.physics_thread = PhysicsThread(self.engine)
        self.latest_physics_state = None  # 최신 물리 상태 (렌더링용)

        # === VBO: GPU 버퍼 객체 (init_display에서 생성) ===
        self.sphere_vbo = None
        self.sphere_vertex_count = 0
        self.vbo_initialized = False

        # === Display List: 구 geometry 캐싱 (성능 최적화) ===
        self.sphere_display_list = None  # 기본 구 (색상용)
        self.sphere_textured_display_list = None  # 텍스처 구 (숫자용)

        # === 공 번호 텍스처 캐시 (성능 최적화) ===
        self.number_textures = {}  # {번호: texture_id}
        self.number_texture_sizes = {}  # {번호: (width, height)}

        # 전역 인스턴스 리스트에 추가
        global _active_visualizers
        with _visualizers_lock:
            _active_visualizers.append(self)

    def _generate_ball_colors(self):
        """공 번호별 색상 생성 (실제 로또처럼 각 공마다 다른 색)"""
        import colorsys
        colors = []

        # 45개 공에 대해 각각 다른 색상 생성
        for i in range(45):
            # HSV 색공간에서 균등하게 분포된 색상 생성
            hue = (i * 360 / 45) / 360.0  # 0~1 범위의 색상
            saturation = 1.0  # 최대 채도 (선명한 색상)
            value = 1.0  # 최대 밝기

            # HSV를 RGB로 변환
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((r, g, b))

        return colors

    def init_display(self):
        """PyGame + OpenGL 초기화"""
        pygame.init()
        self.display_size = (1200, 900)
        pygame.display.set_mode(self.display_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption(f"로또 물리 시뮬레이션 3D - {self.mode}")

        # 폰트 초기화 (더 크고 굵게 - 가독성 향상)
        self.font = pygame.font.SysFont("malgun gothic", 20, bold=True)
        self.font_large = pygame.font.SysFont("malgun gothic", 28, bold=True)

        # V-Sync 비활성화 (부드러운 렌더링을 위해)
        # pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)

        # OpenGL 설정
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # 조명 설정 (단일 광원)
        glLightfv(GL_LIGHT0, GL_POSITION, (500, 500, 500, 1))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))

        # 투영 설정
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.display_size[0] / self.display_size[1]), 0.1, 5000.0)
        glMatrixMode(GL_MODELVIEW)

        # 배경색 설정
        glClearColor(0.1, 0.1, 0.15, 1.0)  # 어두운 파란색 배경

        # === VBO: 구체 geometry 생성 및 업로드 ===
        self._create_sphere_vbo()

        # === Display List: 구 geometry 캐싱 (성능 최적화) ===
        self._create_sphere_display_lists()

        # === 공 번호 텍스처 생성 (1~45번) ===
        self.create_number_textures()
        print("[텍스처] 공 번호 텍스처 45개 생성 완료")

    def _generate_sphere_geometry(self, radius, slices, stacks):
        """구체 geometry 생성 (vertices + normals + UV)"""
        vertices = []
        normals = []
        uvs = []  # UV 좌표 추가

        for i in range(stacks + 1):
            lat = np.pi * (-0.5 + float(i) / stacks)
            z = radius * np.sin(lat)
            zr = radius * np.cos(lat)

            # V 좌표 (위에서 아래로 0~1)
            v = float(i) / stacks

            for j in range(slices + 1):
                lng = 2 * np.pi * float(j) / slices
                x = zr * np.cos(lng)
                y = zr * np.sin(lng)

                # U 좌표 (경도 방향 0~1)
                u = float(j) / slices

                # Vertex
                vertices.append([x, y, z])

                # Normal (normalized vertex for sphere)
                nx = x / radius if radius > 0 else 0
                ny = y / radius if radius > 0 else 0
                nz = z / radius if radius > 0 else 1
                normals.append([nx, ny, nz])

                # UV 좌표
                uvs.append([u, v])

        # Generate triangle indices
        indices = []
        for i in range(stacks):
            for j in range(slices):
                first = i * (slices + 1) + j
                second = first + slices + 1

                # Triangle 1
                indices.append(first)
                indices.append(second)
                indices.append(first + 1)

                # Triangle 2
                indices.append(second)
                indices.append(second + 1)
                indices.append(first + 1)

        return (np.array(vertices, dtype=np.float32),
                np.array(normals, dtype=np.float32),
                np.array(uvs, dtype=np.float32),
                np.array(indices, dtype=np.uint32))

    def _create_sphere_vbo(self):
        """VBO 생성 (GPU 버퍼에 구체 geometry 업로드)"""
        if self.vbo_initialized:
            return

        try:
            print("[VBO] 구체 geometry 생성 중...")

            # 구체 geometry 생성
            radius = 22.25
            slices = 20
            stacks = 20

            vertices, normals, uvs, indices = self._generate_sphere_geometry(radius, slices, stacks)
            self.sphere_vertex_count = len(indices)

            # Interleave vertices, normals, UV (VVVNNNUU -> VNUVNUVNU)
            vertex_data = np.zeros((len(vertices), 8), dtype=np.float32)
            vertex_data[:, 0:3] = vertices  # position
            vertex_data[:, 3:6] = normals   # normal
            vertex_data[:, 6:8] = uvs       # UV
            vertex_data = vertex_data.flatten()

            # VBO 생성 및 데이터 업로드
            self.sphere_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.sphere_vbo)
            glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

            # Element Buffer Object (EBO) 생성
            self.sphere_ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.sphere_ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

            self.vbo_initialized = True
            print(f"[VBO] 구체 VBO 생성 완료!")
            print(f"      - Vertices: {len(vertices)}")
            print(f"      - Triangles: {len(indices) // 3}")
            print(f"      - VBO ID: {self.sphere_vbo}")
            print(f"      - EBO ID: {self.sphere_ebo}\n")

        except Exception as e:
            print(f"[VBO] VBO 생성 실패: {e}")
            print(f"[VBO] Fallback: 기본 렌더링 사용\n")
            self.vbo_initialized = False

    def _create_sphere_display_lists(self):
        """Display List로 구 geometry 캐싱 (성능 최적화)"""
        try:
            # 1. 기본 구 (색상용)
            self.sphere_display_list = glGenLists(1)
            glNewList(self.sphere_display_list, GL_COMPILE)
            quadric = gluNewQuadric()
            gluQuadricNormals(quadric, GLU_SMOOTH)
            gluSphere(quadric, 22.25, 16, 16)
            gluDeleteQuadric(quadric)
            glEndList()

            # 2. 텍스처 구 (숫자용)
            self.sphere_textured_display_list = glGenLists(1)
            glNewList(self.sphere_textured_display_list, GL_COMPILE)
            quadric2 = gluNewQuadric()
            gluQuadricNormals(quadric2, GLU_SMOOTH)
            gluQuadricTexture(quadric2, GL_TRUE)
            gluSphere(quadric2, 22.26, 16, 16)
            gluDeleteQuadric(quadric2)
            glEndList()

            print(f"[Display List] 구 geometry 캐싱 완료!")
            print(f"      - 기본 구 ID: {self.sphere_display_list}")
            print(f"      - 텍스처 구 ID: {self.sphere_textured_display_list}\n")

        except Exception as e:
            print(f"[Display List] 생성 실패: {e}")
            self.sphere_display_list = None
            self.sphere_textured_display_list = None

    def draw_sphere(self, radius, slices=20, stacks=20):
        """구체 그리기"""
        quadric = gluNewQuadric()
        gluSphere(quadric, radius, slices, stacks)
        gluDeleteQuadric(quadric)

    def create_number_textures(self):
        """1~45번 번호 텍스처 미리 생성 (큐브맵 - 6면)"""
        font_size = 80
        font_number = pygame.font.SysFont("arial", font_size, bold=True)

        face_size = 256  # 각 면의 크기
        circle_radius = 80  # 흰색 원 반지름

        for num in range(1, 46):
            text = str(num)

            # 큐브맵 텍스처 생성
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)

            # 6개 면 각각에 대해 텍스처 생성
            # GL_TEXTURE_CUBE_MAP_POSITIVE_X = 오른쪽 (+X)
            # GL_TEXTURE_CUBE_MAP_NEGATIVE_X = 왼쪽 (-X)
            # GL_TEXTURE_CUBE_MAP_POSITIVE_Y = 위 (+Y)
            # GL_TEXTURE_CUBE_MAP_NEGATIVE_Y = 아래 (-Y)
            # GL_TEXTURE_CUBE_MAP_POSITIVE_Z = 앞 (+Z)
            # GL_TEXTURE_CUBE_MAP_NEGATIVE_Z = 뒤 (-Z)

            cube_faces = [
                GL_TEXTURE_CUBE_MAP_POSITIVE_X,
                GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
                GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
                GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
                GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
                GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
            ]

            for face in cube_faces:
                # 각 면마다 개별 텍스처 생성 (투명 배경)
                surface = pygame.Surface((face_size, face_size), pygame.SRCALPHA)
                surface.fill((0, 0, 0, 0))  # 완전 투명 배경

                # 중앙에 흰색 원과 번호 배치
                cx = face_size // 2
                cy = face_size // 2

                # 반투명 흰색 원 그리기 (알파 100 = 매우 투명)
                pygame.draw.circle(surface, (255, 255, 255, 100), (cx, cy), circle_radius)

                # 번호 텍스트 렌더링 (검은색, 완전 불투명)
                text_surf = font_number.render(text, True, (0, 0, 0))
                text_rect = text_surf.get_rect(center=(cx, cy))
                surface.blit(text_surf, text_rect)

                # OpenGL 텍스처로 변환
                text_data = pygame.image.tostring(surface, "RGBA", False)

                # 큐브맵의 각 면에 텍스처 업로드
                glTexImage2D(face, 0, GL_RGBA, face_size, face_size, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

            # 큐브맵 파라미터 설정
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

            # 캐시에 저장
            self.number_textures[num] = texture_id
            self.number_texture_sizes[num] = (face_size, face_size)

    def draw_chamber(self):
        """추첨기 챔버 (투명 구체 + 상하 캡) 그리기 - 개선된 유리 재질"""
        glPushMatrix()

        # 물리 엔진 좌표계와 맞추기 위해 챔버 중심을 (250, 250, 250)으로 이동
        glTranslatef(250, 250, 250)

        # COLOR_MATERIAL 비활성화 (재질 색상이 변하지 않도록)
        glDisable(GL_COLOR_MATERIAL)

        # PMMA 유리 재질 설정
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 유리 재질 속성 (무색 투명 + 반사)
        glass_ambient = (0.3, 0.3, 0.3, 1.0)  # 무색
        glass_diffuse = (1.0, 1.0, 1.0, 0.2)  # 무색 투명 (80% 투명)
        glass_specular = (1.0, 1.0, 1.0, 1.0)  # 강한 반사
        glass_shininess = 120.0  # 유리 광택

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, glass_ambient)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, glass_diffuse)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, glass_specular)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, glass_shininess)

        # 반투명 솔리드 구체로 챔버 표현 (성능 최적화)
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)  # 부드러운 노말
        gluSphere(quadric, 250, 24, 24)  # 반지름 250mm
        gluDeleteQuadric(quadric)

        # 상단 캡 (출구) - 메탈 재질
        # EXTRACTING 단계일 때 출구를 밝게 표시 (진공 ON 표시)
        if hasattr(self, 'engine') and self.engine.phase == "EXTRACTING":
            metal_ambient = (0.5, 0.3, 0.1, 1.0)
            metal_diffuse = (1.0, 0.8, 0.3, 0.5)  # 밝은 노란색, 진공 작동 중
            metal_specular = (1.0, 1.0, 1.0, 1.0)
            metal_shininess = 128.0
        else:
            metal_ambient = (0.3, 0.2, 0.1, 1.0)
            metal_diffuse = (0.8, 0.5, 0.2, 0.3)  # 주황색 메탈, 70% 투명
            metal_specular = (1.0, 0.9, 0.7, 1.0)
            metal_shininess = 100.0

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, metal_ambient)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, metal_diffuse)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, metal_specular)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, metal_shininess)

        glPushMatrix()
        glTranslatef(0, 0, 250)  # 챔버 최상단 (구 반지름 250mm)

        # 출구 튜브 (부드럽게) - 반지름 30mm (직경 60mm)
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluCylinder(quadric, 30, 30, 80, 24, 1)  # 반지름 30mm (공 44.5mm가 통과)
        gluDeleteQuadric(quadric)

        glPopMatrix()

        # 하단 캡 (공기 유입구) - 어두운 메탈
        dark_metal_ambient = (0.1, 0.1, 0.15, 1.0)
        dark_metal_diffuse = (0.3, 0.3, 0.4, 0.7)
        dark_metal_specular = (0.5, 0.5, 0.6, 1.0)
        dark_metal_shininess = 60.0

        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, dark_metal_ambient)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, dark_metal_diffuse)
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, dark_metal_specular)
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, dark_metal_shininess)

        glPushMatrix()
        glTranslatef(0, 0, -250)  # 챔버 최하단 (구 반지름 250mm)
        glRotatef(180, 1, 0, 0)
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluDisk(quadric, 0, 60, 24, 1)  # 공기 유입구
        gluDeleteQuadric(quadric)
        glPopMatrix()


        # COLOR_MATERIAL 다시 활성화 (공 그리기를 위해)
        glEnable(GL_COLOR_MATERIAL)
        glDisable(GL_BLEND)
        glPopMatrix()

    def draw_ball(self, ball, ball_num):
        """공 그리기 (회전 + Specular 반사 + 부드러운 표면)"""
        glPushMatrix()

        # COLOR_MATERIAL 비활성화 (명시적 재질 설정을 위해)
        glDisable(GL_COLOR_MATERIAL)

        # 물리 엔진 좌표를 그대로 사용 (Z축이 위)
        glTranslatef(ball.x, ball.y, ball.z)

        # 공 회전 계산 (간소화 - 성능 최적화)
        ball_idx = ball_num - 1
        speed = ball.vx**2 + ball.vy**2 + ball.vz**2  # 제곱만 계산 (sqrt 생략)
        if speed > 100.0:  # 충분히 빠르게 움직일 때만 회전 (10 mm/s 이상)
            # 스핀을 직접 사용 (간단한 회전)
            if hasattr(ball, 'wx') and hasattr(ball, 'wy') and hasattr(ball, 'wz'):
                # 스핀 각속도를 회전각으로 변환 (라디안 → 도)
                rot_angle = np.sqrt(ball.wx**2 + ball.wy**2 + ball.wz**2) * 57.3 * 0.016  # rad/s → deg/frame
                if rot_angle > 0.1:
                    # 스핀 축
                    spin_mag = np.sqrt(ball.wx**2 + ball.wy**2 + ball.wz**2)
                    if spin_mag > 0.01:
                        glRotatef(rot_angle, ball.wx/spin_mag, ball.wy/spin_mag, ball.wz/spin_mag)

        # 공 재질 설정 (플라스틱 광택)
        color = self.ball_colors[ball_num - 1]

        # 선택된 공은 밝게 표시
        if ball_num in self.selected_balls:
            ambient = (0.8, 0.8, 0.2, 1.0)
            diffuse = (1.0, 1.0, 0.3, 1.0)
            specular = (1.0, 1.0, 1.0, 1.0)
            shininess = 100.0
        else:
            # Ambient (주변광)
            ambient = (color[0] * 0.3, color[1] * 0.3, color[2] * 0.3, 1.0)
            # Diffuse (확산광)
            diffuse = (color[0], color[1], color[2], 1.0)
            # Specular (반사광) - 플라스틱 광택
            specular = (0.9, 0.9, 0.9, 1.0)
            # Shininess (광택 강도)
            shininess = 80.0

        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
        glMaterialf(GL_FRONT, GL_SHININESS, shininess)

        # 공 그리기 (44.5mm 직경 = 22.25mm 반지름, 부드러운 표면)
        self.draw_sphere(22.25, slices=20, stacks=20)  # 32→20으로 감소 (성능 향상)

        # COLOR_MATERIAL 다시 활성화
        glEnable(GL_COLOR_MATERIAL)

        glPopMatrix()

    def draw_ball_from_data(self, ball_data):
        """공 그리기 (튜플 데이터로부터 - VBO 버전, 최적화됨)"""
        # 튜플 인덱스: 0-2=위치, 3-5=속도, 6-8=회전, 9=speed, 10=spin_speed, 11=active, 12=number, 13=mass

        # 추출된 공은 그리지 않음 (2D 오버레이에만 표시)
        if not ball_data[11]:  # active
            return

        # 챔버 밖 위치면 그리지 않음 (-1000, -1000, -1000)
        x, y, z = ball_data[0], ball_data[1], ball_data[2]  # 위치
        if x < -500 or y < -500 or z < -500:
            return

        glPushMatrix()

        # COLOR_MATERIAL 비활성화
        glDisable(GL_COLOR_MATERIAL)

        # 위치
        glTranslatef(x, y, z)

        # 회전 (물리 엔진의 실제 각속도 사용)
        ball_num = ball_data[12]  # number
        ball_idx = ball_num - 1

        # 각속도가 있으면 회전 적용
        wx, wy, wz = ball_data[6], ball_data[7], ball_data[8]  # 회전 속도
        spin_mag = ball_data[10]  # spin_speed (이미 계산됨)

        if spin_mag > 0.01:  # 회전이 있으면
            # dt = 1/60초 동안 회전한 각도 (rad/s → deg)
            dt = 1.0 / 60.0
            delta_angle = spin_mag * dt * 57.3  # rad → deg

            # 회전축 (정규화)
            axis_x = wx / spin_mag
            axis_y = wy / spin_mag
            axis_z = wz / spin_mag

            # 누적 회전 업데이트
            current_rotation = self.ball_rotations[ball_idx]
            current_rotation[0] += delta_angle  # 각도 누적

            # 회전축 업데이트 (지수 평균)
            alpha = 0.9  # 이전 축에 90% 가중치
            current_rotation[1] = alpha * current_rotation[1] + (1 - alpha) * axis_x
            current_rotation[2] = alpha * current_rotation[2] + (1 - alpha) * axis_y
            current_rotation[3] = alpha * current_rotation[3] + (1 - alpha) * axis_z

            # 회전축 재정규화
            axis_mag = np.sqrt(current_rotation[1]**2 + current_rotation[2]**2 + current_rotation[3]**2)
            if axis_mag > 0.001:
                current_rotation[1] /= axis_mag
                current_rotation[2] /= axis_mag
                current_rotation[3] /= axis_mag

            # OpenGL 회전 적용 (누적된 회전)
            glRotatef(current_rotation[0], current_rotation[1], current_rotation[2], current_rotation[3])

        # 공 재질 설정
        color = self.ball_colors[ball_num - 1]

        # === 물리 기반 시각 효과 ===
        # 1) 속도 (물리 스레드에서 미리 계산됨)
        speed = ball_data[9]  # speed (인덱스 9)

        # 2) 회전 속도 (물리 스레드에서 미리 계산됨)
        spin_speed = ball_data[10]  # spin_speed (인덱스 10, 이미 사용함)

        # 3) 속도 기반 밝기 증가 (운동 에너지 시각화)
        # 빠른 공 → 밝게 빛남
        speed_factor = min(speed / 500.0, 1.0)  # 500mm/s = 최대
        brightness_boost = 1.0 + speed_factor * 0.3  # 최대 30% 밝기 증가

        # 4) 회전 기반 Specular 증가 (회전 운동 에너지 시각화)
        # 빠르게 회전 → 표면 광택 증가
        spin_factor = min(spin_speed / 50.0, 1.0)  # 50 rad/s = 최대
        spin_specular = 0.9 + spin_factor * 0.1  # 0.9 ~ 1.0

        # 선택된 공은 밝게
        if ball_num in self.selected_balls:
            ambient = (0.8, 0.8, 0.2, 1.0)
            diffuse = (1.0, 1.0, 0.3, 1.0)
            specular = (1.0, 1.0, 1.0, 1.0)
            shininess = 100.0
        else:
            # 속도/회전에 따라 밝기 조절
            ambient = (color[0] * 0.3 * brightness_boost,
                      color[1] * 0.3 * brightness_boost,
                      color[2] * 0.3 * brightness_boost, 1.0)
            diffuse = (color[0] * brightness_boost,
                      color[1] * brightness_boost,
                      color[2] * brightness_boost, 1.0)
            specular = (spin_specular, spin_specular, spin_specular, 1.0)
            shininess = 80.0 + spin_factor * 20.0  # 80 ~ 100

        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
        glMaterialf(GL_FRONT, GL_SHININESS, shininess)

        # === 렌더링 1: 먼저 공 색상만 그리기 ===
        # Display List 사용 (90배 빠름: 45개 공 × 2번씩 → 2번만)
        if self.sphere_display_list:
            glCallList(self.sphere_display_list)
        else:
            # Fallback: Display List 실패 시 기본 방식
            quadric = gluNewQuadric()
            gluQuadricNormals(quadric, GLU_SMOOTH)
            gluSphere(quadric, 22.25, 16, 16)
            gluDeleteQuadric(quadric)

        # === 렌더링 2: 텍스처(숫자) 위에 덧그리기 ===
        if ball_num in self.number_textures:
            glEnable(GL_TEXTURE_CUBE_MAP)
            glBindTexture(GL_TEXTURE_CUBE_MAP, self.number_textures[ball_num])

            # DECAL 모드: 투명 부분은 원본 유지, 불투명 부분만 덮어씀
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

            # 블렌딩 활성화
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # 텍스처 좌표 자동 생성
            glEnable(GL_TEXTURE_GEN_S)
            glEnable(GL_TEXTURE_GEN_T)
            glEnable(GL_TEXTURE_GEN_R)
            glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP)
            glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP)
            glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP)

            # depth test를 약간 앞으로 (z-fighting 방지)
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-1.0, -1.0)

            # 텍스처와 함께 다시 그리기 - Display List 사용
            if self.sphere_textured_display_list:
                glCallList(self.sphere_textured_display_list)
            else:
                # Fallback
                quadric2 = gluNewQuadric()
                gluQuadricNormals(quadric2, GLU_SMOOTH)
                gluQuadricTexture(quadric2, GL_TRUE)
                gluSphere(quadric2, 22.26, 16, 16)
                gluDeleteQuadric(quadric2)

            glDisable(GL_POLYGON_OFFSET_FILL)

        # 텍스처 비활성화
        if ball_num in self.number_textures:
            glDisable(GL_TEXTURE_GEN_S)
            glDisable(GL_TEXTURE_GEN_T)
            glDisable(GL_TEXTURE_GEN_R)
            glDisable(GL_BLEND)
            glDisable(GL_TEXTURE_CUBE_MAP)

        glEnable(GL_COLOR_MATERIAL)
        glPopMatrix()

    def draw_floor_grid(self):
        """바닥 그리드 그리기"""
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glColor3f(0.3, 0.3, 0.4)  # 어두운 회색

        # 그리드 설정
        grid_size = 800  # 800mm x 800mm
        grid_step = 50   # 50mm 간격

        # Z = 0 (챔버 바닥, 절대 좌표)에 그리드 그리기
        # 챔버 중심 = (250, 250, 250), 반지름 = 250mm
        # 따라서 챔버 바닥 = 250 - 250 = 0
        z_floor = 0

        glBegin(GL_LINES)

        # X축 방향 선들 (챔버 중심 250을 기준으로)
        for y in range(-grid_size // 2, grid_size // 2 + grid_step, grid_step):
            glVertex3f(250 + (-grid_size // 2), 250 + y, z_floor)
            glVertex3f(250 + (grid_size // 2), 250 + y, z_floor)

        # Y축 방향 선들
        for x in range(-grid_size // 2, grid_size // 2 + grid_step, grid_step):
            glVertex3f(250 + x, 250 + (-grid_size // 2), z_floor)
            glVertex3f(250 + x, 250 + (grid_size // 2), z_floor)

        glEnd()
        glEnable(GL_LIGHTING)

    def draw_axes(self):
        """좌표축 그리기 (Z축이 위아래)"""
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glBegin(GL_LINES)

        # X축 (빨강) - 좌우
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(400, 0, 0)

        # Y축 (초록) - 앞뒤
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 400, 0)

        # Z축 (파랑) - 위아래 (중요!)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 400)

        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

    def draw_ball_trail(self, ball_idx, color):
        """공 모션 블러 트레일 그리기"""
        if len(self.ball_trails[ball_idx]) < 2:
            return

        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glLineWidth(3.0)

        # 트레일을 선으로 그리기 (점점 투명해짐)
        num_trail = len(self.ball_trails[ball_idx])
        glBegin(GL_LINE_STRIP)
        for i, pos in enumerate(self.ball_trails[ball_idx]):
            # 오래된 것일수록 투명
            alpha = (i + 1) / num_trail * 0.5  # 최대 50% 투명도
            glColor4f(color[0], color[1], color[2], alpha)
            glVertex3f(pos[0], pos[1], pos[2])
        glEnd()

        glLineWidth(1.0)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)


    def draw_ui_overlay(self):
        """UI 오버레이 (FPS, 시뮬레이션 시간, 선택된 번호, 조작법)"""
        # OpenGL 2D 모드로 전환
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, 1200, 0, 900, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        # 배경 박스 제거 (텍스트에 외곽선이 있어서 불필요)
        # glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glColor4f(0, 0, 0, 0.7)
        # glBegin(GL_QUADS)
        # glVertex2f(10, 810)
        # glVertex2f(500, 810)
        # glVertex2f(500, 890)
        # glVertex2f(10, 890)
        # glEnd()

        # if self.selected_balls:
        #     glColor4f(0, 0, 0, 0.7)
        #     glBegin(GL_QUADS)
        #     glVertex2f(350, 10)
        #     glVertex2f(850, 10)
        #     glVertex2f(850, 80)
        #     glVertex2f(350, 80)
        #     glEnd()

        # glColor4f(0, 0, 0, 0.7)
        # glBegin(GL_QUADS)
        # glVertex2f(1010, 650)
        # glVertex2f(1190, 650)
        # glVertex2f(1190, 890)
        # glVertex2f(1010, 890)
        # glEnd()

        # glDisable(GL_BLEND)

        # PyGame으로 텍스트 렌더링
        # OpenGL에서 2D 텍스트는 텍스처로 렌더링
        self._render_text_overlay()

        # 추출된 공 표시 (화면 하단)
        self._draw_extracted_balls()

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        # 원래 투영으로 복원
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _render_text_overlay(self):
        """텍스트 오버레이 렌더링 (매 프레임 그리기)"""
        # 깜빡임 방지: 캐싱 제거하고 매 프레임 텍스트만 그리기
        import time
        current_time = time.time()

        # 텍스트 업데이트는 0.1초마다만 (성능 최적화)
        if hasattr(self, '_cached_texts') and (current_time - self.last_text_update) < self.text_update_interval:
            texts = self._cached_texts
        else:
            self.last_text_update = current_time
            texts = self._generate_text_list()
            self._cached_texts = texts

        # PyGame 표면에 직접 텍스트 그리기 (glDrawPixels 사용 안함)
        self._draw_text_directly(texts)

    def _generate_text_list(self):
        """텍스트 리스트 생성"""
        # 상태 정보
        status_text = f"FPS: {self.current_fps:.0f}"
        time_text = f"시간: {self.simulation_time:.1f}초"
        mode_text = f"모드: {self.mode}"
        pause_text = "[일시정지]" if self.paused else "[실행 중]"

        # 추첨 단계 정보 (실시간 디버그)
        phase_text = f"단계: {self.engine.phase}"
        jet_power_text = f"풍량: {self.engine.jet_power*100:.0f}%"
        jet_force_text = f"풍압: {self.jet_force_multiplier*100:.0f}% ({self.engine.jet_force:.0f} mm/s²)"
        active_count = sum(1 for b in self.engine.balls if not b.extracted)
        balls_text = f"챔버 내 공: {active_count}개"

        # CFD 평균 속도 계산
        import numpy as np
        cfd_speed = np.sqrt(
            self.engine.fluid_grid.vx.mean()**2 +
            self.engine.fluid_grid.vy.mean()**2 +
            self.engine.fluid_grid.vz.mean()**2
        )
        cfd_text = f"CFD 속도: {cfd_speed:.0f} mm/s"

        texts = [
            (status_text, 20, 870, (100, 255, 100)),
            (time_text, 20, 840, (200, 200, 255)),
            (mode_text, 250, 870, (255, 255, 100)),
            (pause_text, 250, 840, (255, 100, 100) if self.paused else (100, 255, 100)),
            (phase_text, 20, 810, (255, 200, 100)),
            (jet_power_text, 20, 780, (100, 200, 255)),
            (jet_force_text, 20, 750, (255, 200, 100)),
            (cfd_text, 20, 720, (100, 255, 200)),
            (balls_text, 20, 690, (255, 255, 255))
        ]

        # 선택된 번호 (6개 + 보너스 1개)
        if self.selected_balls:
            if len(self.selected_balls) <= 6:
                selected_text = f"당첨번호: {', '.join(map(str, sorted(self.selected_balls)))}"
                texts.append((selected_text, 20, 580, (255, 255, 0)))
            else:
                # 7개 추출 완료 - 보너스 분리 표시
                main = sorted(self.selected_balls[:6])
                bonus = self.selected_balls[6]
                main_text = f"당첨번호: {', '.join(map(str, main))}"
                bonus_text = f"보너스: {bonus}"
                texts.append((main_text, 20, 580, (255, 255, 0)))
                texts.append((bonus_text, 20, 550, (255, 128, 0)))

        # 조작법
        controls = [
            ("[ 조작법 ]", 1020, 870, (255, 255, 255)),
            ("마우스: 회전", 1020, 840, (255, 255, 255)),
            ("휠: 줌", 1020, 810, (255, 255, 255)),
            ("SPACE: 시작/정지", 1020, 780, (255, 255, 100)),
            ("R: 새 추첨", 1020, 750, (255, 255, 255)),
            ("↑/↓: 풍압 조절", 1020, 720, (100, 255, 255)),
            ("ESC: 종료", 1020, 690, (255, 255, 255)),
            ("", 1020, 660, (255, 255, 255)),
            ("자동 추첨:", 1020, 630, (100, 255, 100)),
            (f"{len(self.selected_balls)}/7개", 1020, 600, (255, 255, 100)),
        ]
        texts.extend(controls)

        return texts

    def _draw_text_directly(self, texts):
        """텍스트를 OpenGL 위에 직접 그리기 (glDrawPixels 사용 안함)"""
        if not hasattr(self, 'font'):
            return

        # OpenGL에서 PyGame 화면으로 전환
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, self.display_size[0], 0, self.display_size[1])
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 각 텍스트를 개별 텍스처로 렌더링
        for text, x, y, color in texts:
            if text:  # 빈 문자열 제외
                # 그림자
                shadow_surface = self.font.render(text, True, (0, 0, 0))
                self._draw_text_surface(shadow_surface, x + 1, y - 1)

                # 본문
                text_surface = self.font.render(text, True, color)
                self._draw_text_surface(text_surface, x, y)

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        # 원래 투영으로 복원
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def _draw_text_surface(self, surface, x, y):
        """PyGame 표면을 OpenGL 텍스처로 변환하여 그리기"""
        text_data = pygame.image.tostring(surface, "RGBA", True)
        width, height = surface.get_size()

        glRasterPos2f(x, y)
        glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    def _draw_extracted_balls(self):
        """추출된 공들을 화면 하단에 원형으로 표시"""
        if not self.selected_balls:
            return

        # 2D 오버레이 모드 (이미 draw_ui_overlay에서 설정됨)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # 화면 크기
        screen_w, screen_h = self.display_size  # 1200 x 900

        # 공 크기 및 배치
        ball_radius = 35  # 픽셀
        spacing = 90  # 공 간격
        start_y = 80  # 화면 하단에서 위로 80px

        # 7개 공의 총 너비 계산
        total_width = len(self.selected_balls) * spacing
        start_x = (screen_w - total_width) / 2 + spacing / 2  # 중앙 정렬

        # 각 공 그리기
        for i, ball_number in enumerate(self.selected_balls):
            x = start_x + i * spacing
            y = start_y

            # 보너스 공은 약간 작고 다른 색
            is_bonus = (i == 6)
            radius = ball_radius * 0.9 if is_bonus else ball_radius

            # 공 색상 (번호에 따라 다른 색)
            if ball_number <= 10:
                color = (255, 200, 50)  # 노란색
            elif ball_number <= 20:
                color = (100, 150, 255)  # 파란색
            elif ball_number <= 30:
                color = (255, 100, 100)  # 빨간색
            elif ball_number <= 40:
                color = (150, 150, 150)  # 회색
            else:
                color = (100, 255, 100)  # 녹색

            # 보너스는 주황색
            if is_bonus:
                color = (255, 150, 50)

            # 공 테두리 (어두운 원)
            self._draw_circle_2d(x, y, radius + 3, (50, 50, 50))

            # 공 본체
            self._draw_circle_2d(x, y, radius, color)

            # 번호 텍스트 (흰색, 큰 폰트)
            if hasattr(self, 'font_large'):
                font = self.font_large
            else:
                font = pygame.font.Font(None, 48)
                self.font_large = font

            text_surface = font.render(str(ball_number), True, (255, 255, 255))
            text_w, text_h = text_surface.get_size()

            # 텍스트 중앙 정렬
            text_x = x - text_w / 2
            text_y = y - text_h / 2

            # OpenGL 좌표로 변환하여 텍스트 그리기
            text_data = pygame.image.tostring(text_surface, "RGBA", True)
            glRasterPos2f(text_x, text_y)
            glDrawPixels(text_w, text_h, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

            # 보너스 라벨
            if is_bonus:
                label_font = pygame.font.Font(None, 20)
                label_surface = label_font.render("BONUS", True, (255, 200, 50))
                label_w, label_h = label_surface.get_size()
                label_x = x - label_w / 2
                label_y = y + radius + 10

                label_data = pygame.image.tostring(label_surface, "RGBA", True)
                glRasterPos2f(label_x, label_y)
                glDrawPixels(label_w, label_h, GL_RGBA, GL_UNSIGNED_BYTE, label_data)

        glDisable(GL_BLEND)

    def _draw_circle_2d(self, x, y, radius, color):
        """2D 원 그리기 (filled)"""
        glColor3ub(color[0], color[1], color[2])
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(x, y)
        segments = 32
        for i in range(segments + 1):
            angle = 2.0 * np.pi * i / segments
            dx = radius * np.cos(angle)
            dy = radius * np.sin(angle)
            glVertex2f(x + dx, y + dy)
        glEnd()

    def update_camera(self):
        """카메라 업데이트 (자유 회전)"""
        glLoadIdentity()

        # 구면 좌표계 - 자유롭게 회전
        cam_x = self.camera_distance * np.cos(np.radians(self.camera_rotation_x)) * np.cos(np.radians(self.camera_rotation_y))
        cam_y = self.camera_distance * np.cos(np.radians(self.camera_rotation_x)) * np.sin(np.radians(self.camera_rotation_y))
        cam_z = self.camera_distance * np.sin(np.radians(self.camera_rotation_x))

        gluLookAt(
            cam_x, cam_y, cam_z,  # 카메라 위치
            self.camera_target[0], self.camera_target[1], self.camera_target[2],  # 타겟
            0, 0, 1  # Up 벡터 (Z축이 위)
        )

    def handle_events(self):
        """이벤트 처리"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # ★ 추첨 완료 상태에서 스페이스바 → 자동 리셋
                    if self.draw_complete:
                        # 리셋 로직 실행
                        with self.physics_thread.lock:
                            # 난수 생성기 리셋 (새로운 시드로 다른 결과)
                            import numpy as np
                            self.engine.rng = np.random.default_rng()

                            # 공 재초기화 (바닥부터 시작)
                            self.engine._initialize_balls()
                            self.engine.extracted_balls = []
                            self.engine.time = 0.0

                            # phase 초기화
                            self.engine.phase = "INITIAL"
                            self.engine.phase_timer = 0.0
                            self.engine.extracted_count = 0
                            self.engine.captured_ball = None

                            # 물리 힘 복원
                            self.engine.jet_force = self.original_jet_force * self.jet_force_multiplier
                            self.engine.jet_power = self.jet_force_multiplier  # 풍량도 같이 복원
                            self.engine.vacuum_force = 100000.0
                            self.engine.turbulence = 80000.0

                        # 렌더러 상태 리셋
                        self.selected_balls = []
                        self.simulation_time = 0.0
                        self.draw_complete = False
                        self.completion_time = None
                        self.auto_pause_message_shown = False
                        self.paused = False
                        self.latest_physics_state = None  # ★ 이전 물리 상태 클리어

                        # 물리 스레드 동기화
                        self.physics_thread.paused = False
                        self.physics_thread.simulation_time = 0.0
                        self.physics_thread.state_queue.clear()

                        print("\n[스페이스바] 리셋! 새로운 추첨을 시작합니다...\n")
                    else:
                        # 일반 일시정지 토글
                        self.paused = not self.paused
                        # 물리 스레드 동기화
                        self.physics_thread.paused = self.paused
                elif event.key == pygame.K_r:
                    # 리셋 - 엔진 초기화 (Thread-Safe)
                    # 물리 스레드의 lock을 획득하여 안전하게 리셋
                    with self.physics_thread.lock:
                        # 난수 생성기 리셋 (새로운 시드로 다른 결과)
                        import numpy as np
                        self.engine.rng = np.random.default_rng()

                        # 공 재초기화 (바닥부터 시작)
                        self.engine._initialize_balls()
                        self.engine.extracted_balls = []  # 추출된 공 리스트도 초기화
                        self.engine.time = 0.0  # 물리 엔진 시간 리셋

                        # phase 초기화
                        self.engine.phase = "INITIAL"
                        self.engine.phase_timer = 0.0
                        self.engine.extracted_count = 0
                        self.engine.captured_ball = None

                        # 물리 힘 복원 (추첨 완료 시 꺼진 것들)
                        self.engine.jet_force = self.original_jet_force * self.jet_force_multiplier
                        self.engine.jet_power = self.jet_force_multiplier  # 풍량도 같이 복원
                        self.engine.vacuum_force = 100000.0  # 원래 값
                        self.engine.turbulence = 80000.0  # 원래 값

                    # 렌더러 상태 리셋
                    self.selected_balls = []
                    self.simulation_time = 0.0
                    self.draw_complete = False
                    self.completion_time = None
                    self.auto_pause_message_shown = False  # 메시지 플래그 리셋
                    self.paused = False  # 일시정지 해제
                    self.latest_physics_state = None  # ★ 이전 물리 상태 클리어

                    # 물리 스레드 동기화
                    self.physics_thread.paused = False
                    self.physics_thread.simulation_time = 0.0  # 물리 스레드 시간도 리셋

                    # 상태 큐 클리어 (오래된 상태 제거)
                    self.physics_thread.state_queue.clear()

                    print("\n리셋! 새로운 추첨을 시작합니다...\n")

                elif event.key == pygame.K_UP:
                    # 풍압 증가 (최대 200%)
                    self.jet_force_multiplier = min(2.0, self.jet_force_multiplier + self.jet_force_step)
                    self.engine.jet_force = self.original_jet_force * self.jet_force_multiplier
                    self.engine.jet_power = self.jet_force_multiplier  # 풍량도 같이 조절
                    print(f"풍압: {self.jet_force_multiplier*100:.0f}% ({self.engine.jet_force:.1f} mm/s²)")

                elif event.key == pygame.K_DOWN:
                    # 풍압 감소 (최소 0%)
                    self.jet_force_multiplier = max(0.0, self.jet_force_multiplier - self.jet_force_step)
                    self.engine.jet_force = self.original_jet_force * self.jet_force_multiplier
                    self.engine.jet_power = self.jet_force_multiplier  # 풍량도 같이 조절

                    if self.jet_force_multiplier == 0.0:
                        print(f"풍압: 0% (송풍기 OFF)")
                    else:
                        print(f"풍압: {self.jet_force_multiplier*100:.0f}% ({self.engine.jet_force:.1f} mm/s²)")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 왼쪽 버튼
                    self.mouse_down = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:  # 휠 업
                    self.camera_distance = max(400, self.camera_distance - 50)
                elif event.button == 5:  # 휠 다운
                    self.camera_distance = min(1500, self.camera_distance + 50)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_down = False

            elif event.type == pygame.MOUSEMOTION:
                if self.mouse_down:
                    current_pos = pygame.mouse.get_pos()
                    dx = current_pos[0] - self.last_mouse_pos[0]
                    dy = current_pos[1] - self.last_mouse_pos[1]

                    self.camera_rotation_y += dx * 0.5
                    self.camera_rotation_x += dy * 0.5

                    # 제한
                    self.camera_rotation_x = max(-89, min(89, self.camera_rotation_x))

                    self.last_mouse_pos = current_pos

    def select_top_balls(self):
        """가장 위에 있는 공 6개 선택"""
        # Z좌표가 가장 높은 공 6개 (3D에서는 Z가 위)
        z_positions = [(i+1, ball.z) for i, ball in enumerate(self.engine.balls)]
        z_positions.sort(key=lambda x: x[1], reverse=True)
        self.selected_balls = [num for num, _ in z_positions[:6]]
        self.selected_balls.sort()
        print(f"선택된 번호: {self.selected_balls}")

    def draw_selected_balls_display(self):
        """선택된 공들을 챔버 아래에 줄지어 표시"""
        if not self.selected_balls:
            return

        glPushMatrix()

        # 챔버 아래쪽에 일렬로 배치 (보기 쉽게)
        start_x = -150  # 중앙 기준 왼쪽부터 시작
        start_y = -400  # 챔버 아래
        start_z = 0     # 챔버와 같은 Z 평면
        spacing = 60    # 공 간격

        for i, num in enumerate(self.selected_balls):
            glPushMatrix()
            glTranslatef(start_x + i * spacing, start_y, start_z)

            # COLOR_MATERIAL 비활성화
            glDisable(GL_COLOR_MATERIAL)

            # 공 재질 설정 (밝게 표시)
            color = self.ball_colors[num - 1]
            ambient = (color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, 1.0)
            diffuse = (color[0], color[1], color[2], 1.0)
            specular = (1.0, 1.0, 1.0, 1.0)
            shininess = 100.0

            glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
            glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
            glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
            glMaterialf(GL_FRONT, GL_SHININESS, shininess)

            # === 렌더링 1: 먼저 공 색상만 그리기 ===
            quadric = gluNewQuadric()
            gluQuadricNormals(quadric, GLU_SMOOTH)
            gluSphere(quadric, 25, 16, 16)  # 32→16 (성능 최적화)
            gluDeleteQuadric(quadric)

            # === 렌더링 2: 텍스처(숫자) 위에 덧그리기 ===
            if num in self.number_textures:
                glEnable(GL_TEXTURE_CUBE_MAP)
                glBindTexture(GL_TEXTURE_CUBE_MAP, self.number_textures[num])
                glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

                glEnable(GL_TEXTURE_GEN_S)
                glEnable(GL_TEXTURE_GEN_T)
                glEnable(GL_TEXTURE_GEN_R)
                glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP)
                glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP)
                glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_NORMAL_MAP)

                glEnable(GL_POLYGON_OFFSET_FILL)
                glPolygonOffset(-1.0, -1.0)

                quadric2 = gluNewQuadric()
                gluQuadricNormals(quadric2, GLU_SMOOTH)
                gluQuadricTexture(quadric2, GL_TRUE)
                gluSphere(quadric2, 25.02, 16, 16)  # 32→16 (성능 최적화)
                gluDeleteQuadric(quadric2)

                glDisable(GL_POLYGON_OFFSET_FILL)
                glDisable(GL_TEXTURE_GEN_S)
                glDisable(GL_TEXTURE_GEN_T)
                glDisable(GL_TEXTURE_GEN_R)
                glDisable(GL_BLEND)
                glDisable(GL_TEXTURE_CUBE_MAP)

            glEnable(GL_COLOR_MATERIAL)
            glPopMatrix()

        glPopMatrix()

    def render(self):
        """렌더링 (멀티스레드 버전 - 물리 상태 스냅샷 사용)"""
        # 최신 물리 상태 없으면 렌더링 스킵
        if not self.latest_physics_state:
            return

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 카메라 업데이트
        self.update_camera()

        # 바닥 그리드 (환경)
        self.draw_floor_grid()

        # 1단계: 불투명 객체 먼저 그리기 (활성화된 공들만)
        glDisable(GL_BLEND)
        balls_rendered = 0

        # 물리 스레드의 스냅샷 사용 (Thread-Safe)
        # 튜플 인덱스: 0-2=위치, 3-5=속도, 6-8=회전, 9=speed, 10=spin_speed, 11=active, 12=number, 13=mass
        balls_data = self.latest_physics_state['balls']
        for ball_data in balls_data:
            if ball_data[11]:  # active (인덱스 11)
                self.draw_ball_from_data(ball_data)
                balls_rendered += 1

        # 추출된 공은 3D에서 표시하지 않음 (2D 오버레이만)
        # self.draw_selected_balls_display()  # 제거됨

        # 2단계: 투명 객체 나중에 그리기 (챔버)
        glEnable(GL_BLEND)
        glDepthMask(GL_FALSE)
        self.draw_chamber()
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

        # UI 오버레이
        self.draw_ui_overlay()

        pygame.display.flip()

    def step_simulation(self):
        """물리 상태 가져오기 (멀티스레드 버전)"""
        # ★ 개선: 일시정지 중이 아닐 때만 대기 (끊김 방지 + 먹통 방지)
        should_wait = not self.paused and not self.physics_thread.paused
        new_state = self.physics_thread.get_latest_state(wait=should_wait, timeout=0.05)

        # 새 상태가 있을 때만 업데이트
        if new_state:
            self.latest_physics_state = new_state
            self.simulation_time = new_state['simulation_time']

            # 트레일 업데이트 (렌더링 스레드에서 처리)
            # 튜플 언패킹으로 최적화
            # 튜플 인덱스: 0-2=위치, 3-5=속도, 6-8=회전, 9=speed, 10=spin_speed, 11=active, 12=number, 13=mass
            for i, ball_data in enumerate(new_state['balls']):
                active = ball_data[11]  # active
                if active:
                    # 속도는 물리 스레드에서 미리 계산됨
                    speed = ball_data[9]  # speed

                    # 속도에 비례한 트레일 길이 (빠른 공 → 긴 잔상)
                    # 100mm/s 미만: 트레일 없음
                    # 100-500mm/s: 트레일 3-10개
                    if speed > 100:
                        max_trail_length = int(3 + min(speed / 500.0, 1.0) * 7)  # 3~10
                    else:
                        max_trail_length = 0  # 느린 공은 트레일 없음

                    # 현재 위치를 트레일에 추가
                    if max_trail_length > 0:
                        x, y, z = ball_data[0], ball_data[1], ball_data[2]  # 위치
                        self.ball_trails[i].append([x, y, z])
                        # 속도에 따라 다른 길이 유지
                        if len(self.ball_trails[i]) > max_trail_length:
                            self.ball_trails[i].pop(0)
                    else:
                        # 느린 공은 트레일 초기화
                        self.ball_trails[i] = []
                else:
                    # 비활성화된 공은 트레일 초기화
                    self.ball_trails[i] = []

            # 추출된 공 확인
            extracted = new_state.get('extracted', [])
            if len(extracted) > len(self.selected_balls):
                # 새로운 공이 추출됨
                new_ball = extracted[-1]
                print(f"[{self.simulation_time:.1f}초] {len(extracted)}번째 공 추출: {new_ball}번")

                self.selected_balls = extracted[:7]  # 7개 추출
                if len(self.selected_balls) >= 6 and not self.draw_complete:
                    # 6개 추출되면 완료 표시 (보너스는 계속 진행)
                    main_numbers = sorted(self.selected_balls[:6])
                    bonus = self.selected_balls[6] if len(self.selected_balls) == 7 else "추출중..."
                    print(f"\n[완료] 추첨 완료! 당첨번호: {main_numbers} + 보너스: {bonus}\n")
                    if len(self.selected_balls) == 7:
                        self.draw_complete = True
                        self.completion_time = self.simulation_time

                        # 모든 공기력 중단 (공들이 바닥으로 떨어지도록)
                        self.engine.turbulence = 0.0
                        self.engine.jet_force = 0.0  # 제트 바람 끄기
                        self.engine.vacuum_force = 0.0  # 진공도 끄기

                        print("\n" + "=" * 70)
                        print("공들이 바닥으로 떨어지는 중...")
                        print("=" * 70 + "\n")

            # ★ 추첨 완료 후: 모든 공이 바닥에 떨어지면 자동 정지
            if self.draw_complete and not self.paused:
                # 모든 공의 속도 확인 (추출된 공 제외)
                all_settled = True
                for ball_data in new_state['balls']:
                    active = ball_data[11]  # active (챔버 안에 있는 공)
                    if active:
                        speed = ball_data[9]  # 속도
                        z = ball_data[2]  # z 좌표

                        # 속도가 빠르거나 공중에 떠있으면 아직 안정되지 않음
                        if speed > 50 or z > 100:  # 50mm/s 이상 또는 바닥에서 100mm 이상
                            all_settled = False
                            break

                if all_settled:
                    # 모든 공이 바닥에 정착 → 자동 정지
                    self.paused = True
                    self.physics_thread.paused = True
                    if not self.auto_pause_message_shown:
                        print("\n" + "=" * 70)
                        print("✅ 모든 공이 바닥에 안착 - 시뮬레이션 자동 정지")
                        print("스페이스바를 눌러 새 추첨을 시작하세요.")
                        print("=" * 70 + "\n")
                        self.auto_pause_message_shown = True
        # else: 새 상태가 없어도 self.latest_physics_state는 유지됨 (마지막 상태 재사용)

    def run(self):
        """메인 루프 (멀티스레드 버전)"""
        self.init_display()
        self.running = True
        clock = pygame.time.Clock()

        # === 물리 스레드 시작 ===
        print("\n[멀티스레드] 물리 엔진 스레드 시작...")
        self.physics_thread.start()
        print("[멀티스레드] 물리 엔진이 백그라운드에서 60Hz로 실행 중\n")

        # 시작 메시지
        print("=" * 60)
        print("3D 물리 시뮬레이션 시작! (멀티스레드 모드)")
        print("=" * 60)
        print("스페이스바를 눌러 추첨을 시작하세요.")
        print("조작법:")
        print("  - 마우스 드래그: 카메라 회전")
        print("  - 마우스 휠: 줌 인/아웃")
        print("  - SPACE: 시작/일시정지")
        print("  - R: 새 추첨 시작")
        print("  - ESC: 종료")
        print("=" * 60)

        # 디버그: 초기 상태 확인
        active_balls = sum(1 for b in self.engine.balls if not b.extracted)
        print(f"[DEBUG] Initial state: {active_balls} active balls")
        print(f"[DEBUG] Camera: dist={self.camera_distance:.1f}, rot_x={self.camera_rotation_x:.1f}, rot_y={self.camera_rotation_y:.1f}")
        print("=" * 60 + "\n")

        try:
            while self.running:
                self.handle_events()
                self.step_simulation()
                self.render()

                # FPS 업데이트
                clock.tick(self.fps)
                self.current_fps = clock.get_fps()
        except KeyboardInterrupt:
            print("\n[종료] Ctrl+C 감지됨")
        finally:
            # === 물리 스레드 종료 ===
            print("\n[멀티스레드] 물리 엔진 스레드 종료 중...")
            self.physics_thread.stop()
            self.physics_thread.join(timeout=2.0)
            print("[멀티스레드] 종료 완료")

            # pygame 종료
            pygame.quit()
            print("[종료] 시각화 프로그램 종료 완료")

            # 전역 인스턴스 리스트에서 제거
            global _active_visualizers
            with _visualizers_lock:
                if self in _active_visualizers:
                    _active_visualizers.remove(self)


def launch_visualizer(num_balls=45, mode="물리시뮬3D"):
    """시각화 런처"""
    try:
        visualizer = PhysicsVisualizer3D(num_balls=num_balls, mode=mode)
        visualizer.run()
    except Exception as e:
        print(f"시각화 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 스레드 내에서 실행될 때는 sys.exit() 호출하지 않음
        # (메인 프로그램을 종료시키지 않기 위해)
        print("[종료] 시각화 스레드 정상 종료")


def cleanup_all_visualizers():
    """모든 활성 시각화 윈도우 종료"""
    global _active_visualizers
    with _visualizers_lock:
        visualizers = list(_active_visualizers)  # 복사본 생성

    if not visualizers:
        return

    print(f"\n[정리] {len(visualizers)}개의 시각화 윈도우를 종료합니다...")

    for viz in visualizers:
        try:
            # running 플래그 종료
            viz.running = False

            # 물리 스레드 종료
            if hasattr(viz, 'physics_thread') and viz.physics_thread:
                viz.physics_thread.stop()

            # pygame 이벤트로 강제 종료 (이벤트 루프가 실행 중인 경우)
            try:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            except:
                pass

        except Exception as e:
            print(f"   [WARN] 시각화 윈도우 종료 중 오류: {e}")

    print("   [OK] 모든 시각화 윈도우 종료 완료")


if __name__ == "__main__":
    launch_visualizer()
