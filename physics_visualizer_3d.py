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

from lotto_physics import LottoChamber3D_Ultimate


class PhysicsThread(threading.Thread):
    """물리 엔진 전용 스레드 (정확히 60Hz 유지)"""

    def __init__(self, engine):
        super().__init__(daemon=True)
        self.engine = engine
        self.running = True
        self.paused = True  # 시작 시 일시정지 상태
        self.lock = threading.Lock()

        # 상태 큐 (최대 2개 버퍼)
        self.state_queue = deque(maxlen=2)

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
        """렌더링용 상태 스냅샷 (Thread-Safe)"""
        with self.lock:
            return {
                'balls': [
                    {
                        'x': b.x,
                        'y': b.y,
                        'z': b.z,
                        'vx': b.vx,
                        'vy': b.vy,
                        'vz': b.vz,
                        'wx': b.wx,  # 각속도 X축
                        'wy': b.wy,  # 각속도 Y축
                        'wz': b.wz,  # 각속도 Z축
                        'active': not b.extracted,
                        'number': b.number,
                        'mass': b.mass
                    }
                    for b in self.engine.balls
                ],
                'phase': self.engine.phase,
                'simulation_time': self.simulation_time,  # 스레드에서 추적
                'jet_power': self.engine.jet_power,
                'extracted_balls': list(getattr(self.engine, 'extracted_balls', [])),
                'chamber_radius': self.engine.chamber_radius
            }

    def get_latest_state(self):
        """최신 물리 상태 가져오기"""
        if self.state_queue:
            return self.state_queue[-1]
        return None

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
        self.engine = LottoChamber3D_Ultimate()
        # initialize_balls()는 LottoChamber3D_Ultimate.__init__에서 자동 호출됨

        # 카메라 설정 (정면에서 약간 위에서 내려다보기)
        self.camera_distance = 1400.0  # 최적 거리: 챔버 전체가 한눈에 보임
        self.camera_rotation_x = 30.0  # 위에서 내려다보는 각도 (30도)
        self.camera_rotation_y = 45.0  # 측면에서 보는 각도 (45도)
        self.camera_target = [250, 250, 250]  # 챔버 중심 (물리 엔진 좌표)

        # 마우스 상태
        self.mouse_down = False
        self.last_mouse_pos = (0, 0)

        # 시뮬레이션 상태
        self.selected_balls = []
        self.simulation_time = 0.0
        self.fps = 60
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

        # === 공 번호 텍스처 캐시 (성능 최적화) ===
        self.number_textures = {}  # {번호: texture_id}
        self.number_texture_sizes = {}  # {번호: (width, height)}

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

        # === 상부 공기 배출 슬롯 (Air Vent Slots) ===
        # 중앙 튜브 바로 아래, 챔버 표면에 길다란 슬롯 형태
        glDisable(GL_LIGHTING)

        chamber_r = 251  # 챔버 표면 위에 그리기
        slot_width = 10  # 슬롯 너비 10mm
        slot_height = 30  # 슬롯 높이 30mm

        # 슬롯 위치 (꼭대기에서 45도 아래)
        # 꼭대기(Z=251) - 45도 = 251*cos(45°) ≈ 177mm
        vent_distance = 90  # 중앙에서 90mm 떨어진 위치
        z_center = 127  # 슬롯 중심 높이 (상단 142mm, 하단 112mm)
        z_half = slot_height / 2  # 15mm

        # 어두운 슬롯 색상
        glColor4f(0.05, 0.05, 0.05, 1.0)

        # 4방향 (0도, 90도, 180도, 270도)
        for direction in range(4):
            angle_rad = np.radians(direction * 90)

            # 슬롯 중심 위치
            center_x = vent_distance * np.cos(angle_rad)
            center_y = vent_distance * np.sin(angle_rad)

            # 슬롯을 따라 세그먼트 생성 (구체 표면을 따라)
            num_segments = 15
            glBegin(GL_QUAD_STRIP)

            for seg_idx in range(num_segments + 1):
                # 슬롯을 45도 기울임 (radial 방향으로)
                t = seg_idx / num_segments
                z_offset = z_half - t * slot_height  # +25 ~ -25

                # 45도 기울임: 위쪽은 안쪽(중앙), 아래쪽은 바깥쪽(적도)
                # z_offset: +15(위) ~ -15(아래)
                tilt_angle = np.radians(45)
                radial_offset = -z_offset * np.tan(tilt_angle)  # 음수: 위는 안쪽, 아래는 바깥쪽

                # 기울어진 위치
                tilted_distance = vent_distance + radial_offset
                tilted_x = tilted_distance * np.cos(angle_rad)
                tilted_y = tilted_distance * np.sin(angle_rad)
                z_pos = z_center + z_offset

                # 구체 표면으로 투영
                dist_center = np.sqrt(tilted_x**2 + tilted_y**2 + z_pos**2)
                surf_x = tilted_x / dist_center * chamber_r
                surf_y = tilted_y / dist_center * chamber_r
                surf_z = z_pos / dist_center * chamber_r

                # 슬롯 방향 벡터 (접선 방향)
                # 중심에서 수직인 방향
                tangent_x = -center_y
                tangent_y = center_x
                tangent_length = np.sqrt(tangent_x**2 + tangent_y**2)
                if tangent_length > 0.001:
                    tangent_x /= tangent_length
                    tangent_y /= tangent_length

                # 슬롯 양쪽 끝점
                half_width = slot_width / 2
                left_x = surf_x - tangent_x * half_width
                left_y = surf_y - tangent_y * half_width
                left_z = surf_z

                right_x = surf_x + tangent_x * half_width
                right_y = surf_y + tangent_y * half_width
                right_z = surf_z

                # 양쪽 끝점을 표면에 다시 투영
                left_dist = np.sqrt(left_x**2 + left_y**2 + left_z**2)
                left_x = left_x / left_dist * chamber_r
                left_y = left_y / left_dist * chamber_r
                left_z = left_z / left_dist * chamber_r

                right_dist = np.sqrt(right_x**2 + right_y**2 + right_z**2)
                right_x = right_x / right_dist * chamber_r
                right_y = right_y / right_dist * chamber_r
                right_z = right_z / right_dist * chamber_r

                # QUAD_STRIP: 좌-우-좌-우...
                glVertex3f(left_x, left_y, left_z)
                glVertex3f(right_x, right_y, right_z)

            glEnd()

        glEnable(GL_LIGHTING)

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
        """공 그리기 (딕셔너리 데이터로부터 - VBO 버전)"""
        glPushMatrix()

        # COLOR_MATERIAL 비활성화
        glDisable(GL_COLOR_MATERIAL)

        # 위치
        glTranslatef(ball_data['x'], ball_data['y'], ball_data['z'])

        # 회전 (물리 엔진의 실제 각속도 사용)
        ball_num = ball_data['number']
        ball_idx = ball_num - 1

        # 각속도가 있으면 회전 적용
        if 'wx' in ball_data and ball_data['wx'] is not None:
            wx, wy, wz = ball_data['wx'], ball_data['wy'], ball_data['wz']
            spin_mag = np.sqrt(wx**2 + wy**2 + wz**2)

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
        # 1) 속도 계산
        speed = np.sqrt(ball_data['vx']**2 + ball_data['vy']**2 + ball_data['vz']**2)

        # 2) 회전 속도 계산
        spin_speed = 0.0
        if 'wx' in ball_data and ball_data['wx'] is not None:
            spin_speed = np.sqrt(ball_data['wx']**2 + ball_data['wy']**2 + ball_data['wz']**2)

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
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluSphere(quadric, 22.25, 32, 32)
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

            # 텍스처와 함께 다시 그리기
            quadric2 = gluNewQuadric()
            gluQuadricNormals(quadric2, GLU_SMOOTH)
            gluQuadricTexture(quadric2, GL_TRUE)
            gluSphere(quadric2, 22.26, 32, 32)  # 아주 약간 크게
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

        texts = [
            (status_text, 20, 870, (100, 255, 100)),
            (time_text, 20, 840, (200, 200, 255)),
            (mode_text, 250, 870, (255, 255, 100)),
            (pause_text, 250, 840, (255, 100, 100) if self.paused else (100, 255, 100)),
            (phase_text, 20, 810, (255, 200, 100)),
            (jet_power_text, 20, 780, (100, 200, 255)),
            (jet_force_text, 20, 750, (255, 200, 100)),
            (balls_text, 20, 720, (255, 255, 255))
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
                    self.paused = not self.paused
                    # 물리 스레드 동기화
                    self.physics_thread.paused = self.paused
                elif event.key == pygame.K_r:
                    # 리셋 - 엔진 초기화
                    # 난수 생성기 리셋 (새로운 시드로 다른 결과)
                    import numpy as np
                    self.engine.rng = np.random.default_rng()

                    self.engine._initialize_balls()
                    self.engine.extracted_balls = []  # 추출된 공 리스트도 초기화
                    self.selected_balls = []
                    self.simulation_time = 0.0
                    self.draw_complete = False
                    self.completion_time = None
                    self.auto_pause_message_shown = False  # 메시지 플래그 리셋
                    self.paused = False  # 일시정지 해제
                    # 물리 스레드 동기화
                    self.physics_thread.paused = False
                    self.physics_thread.simulation_time = 0.0  # 물리 스레드 시간도 리셋
                    # 공기력 복구 (현재 배율 적용)
                    self.engine.jet_force = self.original_jet_force * self.jet_force_multiplier
                    # 난류 복구 (추첨 완료 시 0으로 설정되었을 수 있음)
                    self.engine.turbulence = 9000.0
                    # phase도 초기화
                    self.engine.phase = "INITIAL"
                    self.engine.phase_timer = 0.0
                    self.engine.extracted_count = 0
                    self.engine.captured_ball = None
                    print("\n리셋! 새로운 추첨을 시작합니다...\n")

                elif event.key == pygame.K_UP:
                    # 풍압 증가 (최대 200%)
                    self.jet_force_multiplier = min(2.0, self.jet_force_multiplier + self.jet_force_step)
                    self.engine.jet_force = self.original_jet_force * self.jet_force_multiplier
                    print(f"풍압: {self.jet_force_multiplier*100:.0f}% ({self.engine.jet_force:.1f} mm/s²)")

                elif event.key == pygame.K_DOWN:
                    # 풍압 감소 (최소 50%)
                    self.jet_force_multiplier = max(0.5, self.jet_force_multiplier - self.jet_force_step)
                    self.engine.jet_force = self.original_jet_force * self.jet_force_multiplier
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
            gluSphere(quadric, 25, 32, 32)
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
                gluSphere(quadric2, 25.02, 32, 32)
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
        balls_data = self.latest_physics_state['balls']
        for ball_data in balls_data:
            if ball_data['active']:  # 활성화된 공만 그리기
                self.draw_ball_from_data(ball_data)
                balls_rendered += 1

        # 선택된 공들 따로 표시
        self.draw_selected_balls_display()

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
        # 물리 스레드에서 최신 상태 가져오기
        new_state = self.physics_thread.get_latest_state()

        if new_state:
            self.latest_physics_state = new_state
            self.simulation_time = new_state['simulation_time']

            # 트레일 업데이트 (렌더링 스레드에서 처리)
            for i, ball_data in enumerate(new_state['balls']):
                if ball_data['active']:
                    # 속도 계산
                    speed = np.sqrt(ball_data['vx']**2 + ball_data['vy']**2 + ball_data['vz']**2)

                    # 속도에 비례한 트레일 길이 (빠른 공 → 긴 잔상)
                    # 100mm/s 미만: 트레일 없음
                    # 100-500mm/s: 트레일 3-10개
                    if speed > 100:
                        max_trail_length = int(3 + min(speed / 500.0, 1.0) * 7)  # 3~10
                    else:
                        max_trail_length = 0  # 느린 공은 트레일 없음

                    # 현재 위치를 트레일에 추가
                    if max_trail_length > 0:
                        self.ball_trails[i].append([ball_data['x'], ball_data['y'], ball_data['z']])
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
            extracted = new_state.get('extracted_balls', [])
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
                        self.engine.turbulence = 0.0

            # 추첨 완료 후 20초 뒤 자동 일시정지 (공들이 바닥에 떨어질 시간 확보)
            if self.draw_complete and self.completion_time is not None:
                if self.simulation_time - self.completion_time >= 20.0:
                    if not self.paused:  # 아직 정지되지 않았을 때만
                        self.paused = True
                        if not self.auto_pause_message_shown:  # 메시지를 아직 출력하지 않았을 때만
                            print("\n시뮬레이션이 자동으로 정지되었습니다.")
                            print("R키를 눌러 새 추첨을 시작하세요.\n")
                            self.auto_pause_message_shown = True

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
        finally:
            # === 물리 스레드 종료 ===
            print("\n[멀티스레드] 물리 엔진 스레드 종료 중...")
            self.physics_thread.stop()
            self.physics_thread.join(timeout=2.0)
            print("[멀티스레드] 종료 완료")

        pygame.quit()


def launch_visualizer(num_balls=45, mode="물리시뮬3D"):
    """시각화 런처"""
    try:
        visualizer = PhysicsVisualizer3D(num_balls=num_balls, mode=mode)
        visualizer.run()
    except Exception as e:
        print(f"시각화 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    launch_visualizer()
