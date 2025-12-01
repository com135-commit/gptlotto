#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lotto 6/45 Simulator (KR) — Genius + Quantum + HM + MQLE + AI + Rigged Sim + 3D + GPU
메인 GUI 프로그램
"""

from __future__ import annotations

# 표준 라이브러리
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os

# 서드파티 라이브러리
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation
import matplotlib

# 로또 시뮬레이터 모듈
from lotto_utils import (
    parse_sets_from_text,
    sets_to_text,
    default_sets,
    get_rng,
)
from lotto_generators import (
    generate_random_sets,
    generate_pattern_sets,
    gen_GI,
    gen_MDA,
    gen_CC,
    gen_PR,
    gen_IS,
    gen_GAPR,
    gen_QH,
    gen_HD,
    gen_QP,
    gen_QP_tunnel,
    gen_QP_entangle,
    gen_QH_QA,
    gen_QP_jump,
    gen_MQLE,
    train_ml_scorer,
    _qh_score,
)
from lotto_history import (
    load_history_csv,
    compute_weights,
    compute_realistic_popularity_weights,
)
from lotto_simulation import (
    run_simulation,
    build_synthetic_player_pool,
    estimate_expected_winners_from_pool,
    _rigged_candidate_chunk,
    _rigged_candidate_gpu,
)

# Matplotlib 한글 폰트 설정
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

_rng = get_rng()

# GPU 지원 확인
try:
    import cupy as cp
except Exception:
    cp = None

class LottoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lotto 6/45 Simulator (Genius + Quantum + HM + MQLE + AI + 3D + Rigged+GPU)")
        self.geometry("1180x820")
        self.resizable(True, True)

        self.history_df: pd.DataFrame | None = None
        self.history_path: str | None = None
        self.history_weights = None
        self.history_exclude_set: set[int] = set()
        self.ml_model: dict | None = None

        # ★ AI 세트 평점 학습 회차 수 (GUI)
        self.ai_max_rounds = tk.StringVar(value="200")

        # 가상 조작 시뮬 관련 상태
        self.rig_win = None
        self.rig_tree = None
        self.rig_status_label = None
        self.rig_target_min = tk.IntVar(value=8)
        self.rig_target_max = tk.IntVar(value=15)
        self.rig_samples = tk.IntVar(value=20000)
        # ★ GPU 사용 여부
        self.rig_use_gpu = tk.BooleanVar(value=False)
        # ★ 가상 플레이어 수 (사용자 지정, 기본 400,000명)
        self.rig_virtual_players = tk.IntVar(value=400000)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.page_sets = ttk.Frame(self.notebook)
        self.page_generate = ttk.Frame(self.notebook)
        self.page_sim = ttk.Frame(self.notebook)
        self.page_viz = ttk.Frame(self.notebook)
        self.page_help = ttk.Frame(self.notebook)

        self.notebook.add(self.page_sets, text="세트 편집")
        self.notebook.add(self.page_generate, text="번호 추출기")
        self.notebook.add(self.page_sim, text="시뮬레이션")
        self.notebook.add(self.page_viz, text="분포 분석")
        self.notebook.add(self.page_help, text="HELP")

        self._build_sets_page()
        self._build_generate_page()
        self._build_sim_page()
        self._build_viz_page()
        self._build_help_page()

        self.text_sets.insert("1.0", sets_to_text(default_sets()))

    # --- 세트 편집 페이지 ---
    def _build_sets_page(self):
        top = self.page_sets
        ttk.Label(top, text="세트 목록 (한 줄에 6개 숫자, 공백/쉼표 구분)").pack(
            anchor="w", padx=10, pady=6
        )
        self.text_sets = tk.Text(top, height=20, wrap="none")
        self.text_sets.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill=tk.X, padx=10, pady=6)
        ttk.Button(btn_frame, text="불러오기(.txt)", command=self._load_sets).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn_frame, text="저장하기(.txt)", command=self._save_sets_txt).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn_frame, text="정렬/중복제거", command=self._normalize_sets).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn_frame, text="전체 초기화", command=self._clear_all_sets).pack(
            side=tk.LEFT, padx=4
        )

    def _load_sets(self):
        path = filedialog.askopenfilename(
            filetypes=[("Text", "*.txt"), ("All", "*.*")]
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            self.text_sets.delete("1.0", tk.END)
            self.text_sets.insert("1.0", f.read())

    def _save_sets_txt(self):
        try:
            sets_ = parse_sets_from_text(self.text_sets.get("1.0", tk.END))
        except Exception as e:
            messagebox.showerror("오류", str(e))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text", "*.txt")]
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(sets_to_text(sets_))
        messagebox.showinfo("저장 완료", f"세트 {len(sets_)}개 저장")

    def _normalize_sets(self):
        try:
            sets_ = parse_sets_from_text(self.text_sets.get("1.0", tk.END))
        except Exception as e:
            messagebox.showerror("오류", str(e))
            return
        uniq = sorted({tuple(s) for s in sets_})
        self.text_sets.delete("1.0", tk.END)
        self.text_sets.insert("1.0", sets_to_text([list(s) for s in uniq]))
        messagebox.showinfo("정리 완료", f"세트 {len(uniq)}개")

    def _clear_all_sets(self):
        self.text_sets.delete("1.0", tk.END)
        messagebox.showinfo("초기화", "세트 목록이 모두 삭제되었습니다.")

    # --- 번호 추출기 페이지 ---
    def _build_generate_page(self):
        top = self.page_generate

        hist = ttk.LabelFrame(top, text="과거 당첨 데이터(옵션)")
        hist.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(hist, text="CSV 불러오기", command=self._load_history).grid(
            row=0, column=0, padx=6, pady=6, sticky="w"
        )
        self.lbl_hist = ttk.Label(hist, text="로드되지 않음")
        self.lbl_hist.grid(row=0, column=1, padx=6, sticky="w")

        self.lbl_ai = ttk.Label(hist, text="AI 세트 평점: 준비 안 됨")
        self.lbl_ai.grid(row=0, column=2, padx=6, sticky="w")

        ttk.Label(hist, text="전략").grid(row=1, column=0, sticky="e")
        self.hist_strategy = tk.StringVar(value="사용 안 함")
        ttk.Combobox(
            hist,
            textvariable=self.hist_strategy,
            state="readonly",
            values=[
                "사용 안 함",
                "Hot(고빈도)",
                "Cold(저빈도)",
                "Overdue(오래 안 나온)",
                "Balanced(중립화)",
            ],
        ).grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(hist, text="Lookback N(최근 N회만)").grid(
            row=1, column=2, sticky="e"
        )
        self.hist_lookback = tk.StringVar(value="")
        ttk.Entry(hist, textvariable=self.hist_lookback, width=10).grid(
            row=1, column=3, sticky="w", padx=6
        )

        ttk.Label(hist, text="최근 K회 제외").grid(row=1, column=4, sticky="e")
        self.hist_exclude = tk.IntVar(value=0)
        ttk.Entry(hist, textvariable=self.hist_exclude, width=8).grid(
            row=1, column=5, sticky="w", padx=6
        )

        # ★ 추가: AI 세트 평점 학습 회차 수
        ttk.Label(hist, text="AI 학습 회차 수(최근 N회, 비우면 전체)").grid(
            row=2, column=0, sticky="e", pady=(4, 2)
        )
        ttk.Entry(hist, textvariable=self.ai_max_rounds, width=10).grid(
            row=2, column=1, sticky="w", padx=6, pady=(4, 2)
        )

        frm = ttk.LabelFrame(top, text="번호 추출기")
        frm.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(frm, text="생성 개수").grid(row=0, column=0, sticky="w")
        self.gen_count = tk.IntVar(value=10)
        ttk.Entry(frm, textvariable=self.gen_count, width=8).grid(
            row=0, column=1, sticky="w", padx=6
        )

        ttk.Label(frm, text="모드").grid(row=0, column=2, sticky="e")
        self.gen_mode = tk.StringVar(value="무작위")
        ttk.Combobox(
            frm,
            textvariable=self.gen_mode,
            state="readonly",
            values=[
                "무작위",
                "패턴",
                "GI(직관)",
                "MDA(다차원)",
                "CC(창의연결)",
                "PR(패턴공진)",
                "IS(혁신시뮬)",
                "GAP-R(간격공진)",
                "QH(다속성조화)",
                "HD(초다양성)",
                "QP-Wave(파동)",
                "QP-Tunnel(터널링)",
                "QP-Entangle(얽힘)",
                "QH-QA(어닐링)",
                "QP-Jump(모드도약)",
                "MQLE(끝판왕)",
            ],
        ).grid(row=0, column=3, sticky="w", padx=6)

        ttk.Label(frm, text="짝수 개수(선택)").grid(row=1, column=0, sticky="w", pady=6)
        self.gen_even = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.gen_even, width=6).grid(
            row=1, column=1, sticky="w"
        )

        ttk.Label(frm, text="구간 분포 (저/중/고)").grid(row=1, column=2, sticky="e")
        self.gen_low = tk.IntVar(value=2)
        self.gen_mid = tk.IntVar(value=2)
        self.gen_high = tk.IntVar(value=2)
        ttk.Entry(frm, textvariable=self.gen_low, width=5).grid(
            row=1, column=3, sticky="w"
        )
        ttk.Entry(frm, textvariable=self.gen_mid, width=5).grid(
            row=1, column=4, sticky="w"
        )
        ttk.Entry(frm, textvariable=self.gen_high, width=5).grid(
            row=1, column=5, sticky="w"
        )

        ttk.Label(frm, text="배수 포함 (3의/7의 최소개수)").grid(
            row=2, column=0, sticky="w", pady=6
        )
        self.gen_m3 = tk.IntVar(value=0)
        self.gen_m7 = tk.IntVar(value=0)
        ttk.Entry(frm, textvariable=self.gen_m3, width=5).grid(
            row=2, column=1, sticky="w"
        )
        ttk.Entry(frm, textvariable=self.gen_m7, width=5).grid(
            row=2, column=2, sticky="w"
        )

        self.qc_balance = tk.IntVar(value=60)
        self.scale_qc = tk.Scale(
            frm,
            from_=0,
            to=100,
            orient="horizontal",
            label="양자 비중(%) — MQLE 전용",
            variable=self.qc_balance,
            length=360,
        )
        self.scale_qc.grid(row=3, column=0, columnspan=6, sticky="we", pady=(8, 0))

        btns = ttk.Frame(top)
        btns.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(btns, text="번호 생성", command=self._gen_dispatch).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btns, text="세트 페이지에 추가", command=self._append_to_sets).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btns, text="생성 결과 초기화", command=self._clear_generated).pack(
            side=tk.LEFT, padx=4
        )

        self.text_generate = tk.Text(top, height=18, wrap="none")
        self.text_generate.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

    def _load_history(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            df = load_history_csv(path)
        except Exception as e:
            messagebox.showerror("CSV 오류", str(e))
            return

        self.history_df = df
        self.history_path = path
        self.lbl_hist.config(
            text=f"로드됨: {os.path.basename(path)} ({len(df)}회)"
        )

        self.ml_model = None
        self.lbl_ai.config(text="AI 세트 평점: 학습 중...")

        try:
            w_bal, _ = compute_weights(
                self.history_df,
                lookback=None,
                strategy="Balanced(중립화)",
                exclude_recent=0,
            )
        except Exception:
            w_bal = None

        # ★ AI 학습 회차 수 읽기
        max_rounds_str = self.ai_max_rounds.get().strip()
        try:
            if max_rounds_str == "":
                max_rounds = None   # 전체 사용
            else:
                max_rounds = int(max_rounds_str)
        except ValueError:
            max_rounds = 200

        if max_rounds is not None and max_rounds <= 0:
            max_rounds = None

        try:
            self.ml_model = train_ml_scorer(
                self.history_df,
                weights=w_bal,
                max_rounds=max_rounds,
            )
        except Exception as e:
            self.ml_model = None
            self.lbl_ai.config(text="AI 세트 평점: 학습 실패(기본 MQLE만 동작)")
            messagebox.showwarning(
                "AI 학습 경고",
                f"세트 평점 AI 학습 실패: {e}",
            )
        else:
            if max_rounds is None:
                used_rounds = len(self.history_df)
            else:
                used_rounds = min(len(self.history_df), max_rounds)
            self.lbl_ai.config(
                text=f"AI 세트 평점: 학습 완료 (최근 {used_rounds}회 사용)"
            )

    def _prepare_history_weights(self):
        strat = self.hist_strategy.get()
        lookback_str = self.hist_lookback.get().strip()
        lookback = None if lookback_str == "" else int(lookback_str)
        excl = max(0, int(self.hist_exclude.get()))
        w, excl_set = compute_weights(
            self.history_df, lookback, strat, exclude_recent=excl
        )
        self.history_weights = w
        self.history_exclude_set = excl_set

    def _gen_dispatch(self):
        mode = self.gen_mode.get()
        n = max(1, self.gen_count.get())
        weights = None
        excl_set: set[int] = set()

        if self.hist_strategy.get() != "사용 안 함":
            if self.history_df is None:
                messagebox.showwarning(
                    "알림", "히스토리 전략 사용 시 CSV를 먼저 불러오세요."
                )
                return
            try:
                self._prepare_history_weights()
            except Exception as e:
                messagebox.showerror("히스토리 가중치 오류", str(e))
                return
            weights = self.history_weights
            excl_set = self.history_exclude_set

        try:
            if mode == "무작위":
                arr = generate_random_sets(
                    n, True, weights, excl_set or None
                )
            elif mode == "패턴":
                even_str = self.gen_even.get().strip()
                even_target = None if even_str == "" else int(even_str)
                arr = generate_pattern_sets(
                    n,
                    even_target=even_target,
                    low_mid_high=(
                        self.gen_low.get(),
                        self.gen_mid.get(),
                        self.gen_high.get(),
                    ),
                    include_multiples=(
                        self.gen_m3.get(),
                        self.gen_m7.get(),
                    ),
                    weights=weights,
                    exclude_set=excl_set or None,
                )
            elif mode == "GI(직관)":
                arr = gen_GI(n, weights=weights, exclude_set=excl_set or None)
            elif mode == "MDA(다차원)":
                arr = gen_MDA(n, weights=weights, exclude_set=excl_set or None)
            elif mode == "CC(창의연결)":
                arr = gen_CC(n, weights=weights, exclude_set=excl_set or None)
            elif mode == "PR(패턴공진)":
                arr = gen_PR(n, weights=weights, exclude_set=excl_set or None)
            elif mode == "IS(혁신시뮬)":
                arr = gen_IS(n, weights=weights, exclude_set=excl_set or None)
            elif mode == "GAP-R(간격공진)":
                arr = gen_GAPR(
                    n,
                    history_df=self.history_df,
                    weights=weights,
                    exclude_set=excl_set or None,
                )
            elif mode == "QH(다속성조화)":
                arr = gen_QH(n, weights=weights, exclude_set=excl_set or None)
            elif mode == "HD(초다양성)":
                base_sets = None
                txt = self.text_sets.get("1.0", tk.END)
                if txt.strip():
                    try:
                        base_sets = parse_sets_from_text(txt)
                    except Exception:
                        base_sets = None
                arr = gen_HD(
                    n, base_sets=base_sets, weights=weights, exclude_set=excl_set or None
                )
            elif mode == "QP-Wave(파동)":
                arr = gen_QP(n, weights=weights, exclude_set=excl_set or None)
            elif mode == "QP-Tunnel(터널링)":
                arr = gen_QP_tunnel(
                    n, weights=weights, exclude_set=excl_set or None
                )
            elif mode == "QP-Entangle(얽힘)":
                arr = gen_QP_entangle(
                    n,
                    history_df=self.history_df,
                    weights=weights,
                    exclude_set=excl_set or None,
                )
            elif mode == "QH-QA(어닐링)":
                arr = gen_QH_QA(n, weights=weights, exclude_set=excl_set or None)
            elif mode == "QP-Jump(모드도약)":
                arr = gen_QP_jump(
                    n,
                    history_df=self.history_df,
                    weights=weights,
                    exclude_set=excl_set or None,
                )
            elif mode == "MQLE(끝판왕)":
                base_sets = None
                txt = self.text_sets.get("1.0", tk.END)
                if txt.strip():
                    try:
                        base_sets = parse_sets_from_text(txt)
                    except Exception:
                        base_sets = None
                q_bal = self.qc_balance.get() / 100.0
                arr = gen_MQLE(
                    n,
                    history_df=self.history_df,
                    weights=weights,
                    exclude_set=excl_set or None,
                    base_sets=base_sets,
                    q_balance=q_bal,
                    ml_model=self.ml_model,
                )
            else:
                arr = []
        except Exception as e:
            messagebox.showerror("번호 생성 오류", str(e))
            return

        self.text_generate.delete("1.0", tk.END)
        self.text_generate.insert("1.0", sets_to_text(arr))

    def _append_to_sets(self):
        try:
            sets_new = parse_sets_from_text(self.text_generate.get("1.0", tk.END))
        except Exception as e:
            messagebox.showerror("오류", str(e))
            return
        current = self.text_sets.get("1.0", tk.END)
        base: list[list[int]] = []
        if current.strip():
            try:
                base = parse_sets_from_text(current)
            except Exception as e:
                messagebox.showerror("오류", f"세트 페이지 오류: {e}")
                return
        merged = [tuple(s) for s in base] + [tuple(s) for s in sets_new]
        uniq = sorted(list({t for t in merged}))
        self.text_sets.delete("1.0", tk.END)
        self.text_sets.insert("1.0", sets_to_text([list(t) for t in uniq]))
        messagebox.showinfo(
            "추가 완료",
            f"세트 {len(sets_new)}개 추가됨 (중복 제거 후 총 {len(uniq)}개)",
        )

    def _clear_generated(self):
        self.text_generate.delete("1.0", tk.END)

    # --- 시뮬레이션 페이지 ---
    def _build_sim_page(self):
        top = self.page_sim

        frm = ttk.Frame(top)
        frm.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(frm, text="총 추첨 횟수(draws)").grid(row=0, column=0, sticky="w")
        self.sim_draws = tk.IntVar(value=2_000_000)
        ttk.Entry(frm, textvariable=self.sim_draws, width=12).grid(
            row=0, column=1, sticky="w", padx=6
        )

        ttk.Label(frm, text="배치(batch)").grid(row=0, column=2, sticky="e")
        self.sim_batch = tk.IntVar(value=200_000)
        ttk.Entry(frm, textvariable=self.sim_batch, width=10).grid(
            row=0, column=3, sticky="w", padx=6
        )

        ttk.Label(frm, text="워커 수(workers, 최대 36)").grid(
            row=0, column=4, sticky="e"
        )
        self.sim_workers = tk.IntVar(value=8)
        ttk.Entry(frm, textvariable=self.sim_workers, width=8).grid(
            row=0, column=5, sticky="w", padx=6
        )

        ttk.Label(frm, text="Seed(선택)").grid(row=1, column=0, sticky="w", pady=6)
        self.sim_seed = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.sim_seed, width=12).grid(
            row=1, column=1, sticky="w"
        )

        btns = ttk.Frame(top)
        btns.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(btns, text="시뮬레이션 실행", command=self._run_sim).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(btns, text="CSV/Excel로 저장", command=self._save_outputs).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(btns, text="가상 조작 시뮬", command=self._open_rigged_dialog).pack(
            side=tk.LEFT, padx=6
        )

        self.progress = ttk.Progressbar(top, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=10, pady=6)
        self.lbl_status = ttk.Label(top, text="대기 중")
        self.lbl_status.pack(anchor="w", padx=10)

        cols = [
            "Set",
            "Numbers",
        ] + [f"match_{m}_count" for m in range(7)] + [
            f"match_{m}_prob" for m in range(7)
        ] + ["match_5plusbonus_count", "match_5plusbonus_prob", "≥3_match_prob"]

        frame_list = ttk.Frame(top)
        frame_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        self.tree = ttk.Treeview(
            frame_list, columns=cols, show="headings", height=16
        )
        vsb = ttk.Scrollbar(frame_list, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(
            frame_list, orient="horizontal", command=self.tree.xview
        )
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        frame_list.rowconfigure(0, weight=1)
        frame_list.columnconfigure(0, weight=1)

        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(
                c, width=110 if c != "Numbers" else 180, anchor="center"
            )

        self.per_set_df: pd.DataFrame | None = None
        self.agg_df: pd.DataFrame | None = None

    def _run_sim(self):
        try:
            sets_ = parse_sets_from_text(self.text_sets.get("1.0", tk.END))
        except Exception as e:
            messagebox.showerror("오류", str(e))
            return
        draws = max(1, self.sim_draws.get())
        batch = max(1, self.sim_batch.get())
        workers = max(1, min(36, self.sim_workers.get()))
        seed_str = self.sim_seed.get().strip()
        seed_val = None if seed_str == "" else int(seed_str)

        def task():
            try:
                self._set_busy(True, "시뮬레이션 실행 중...")
                per_set_df, agg_df = run_simulation(
                    draws, batch, workers, seed_val, sets_
                )
                self.per_set_df = per_set_df
                self.agg_df = agg_df
                self.after(
                    0, lambda: self._populate_tree(per_set_df, agg_df)
                )
                self._set_busy(False, f"완료: draws={draws:,}, workers={workers}, batch={batch:,}")
            except Exception as e_inner:
                self._set_busy(False, "오류 발생")
                messagebox.showerror("오류", str(e_inner))

        threading.Thread(target=task, daemon=True).start()

    def _populate_tree(self, per_set_df: pd.DataFrame, agg_df: pd.DataFrame):
        self.tree.delete(*self.tree.get_children())
        for _, row in per_set_df.iterrows():
            values = [row.get(col, "") for col in self.tree["columns"]]
            self.tree.insert("", tk.END, values=values)
        row = agg_df.iloc[0].to_dict()
        values = [row.get(col, "") for col in self.tree["columns"]]
        self.tree.insert("", tk.END, values=values)

    def _save_outputs(self):
        if self.per_set_df is None or self.agg_df is None:
            messagebox.showwarning("알림", "먼저 시뮬레이션을 실행하세요.")
            return
        folder = filedialog.askdirectory()
        if not folder:
            return
        per_csv = os.path.join(folder, "lotto_per_set.csv")
        agg_csv = os.path.join(folder, "lotto_aggregate.csv")
        self.per_set_df.to_csv(per_csv, index=False)
        self.agg_df.to_csv(agg_csv, index=False)
        try:
            xlsx = os.path.join(folder, "lotto_results.xlsx")
            with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
                self.per_set_df.to_excel(
                    writer, sheet_name="PerSet", index=False
                )
                self.agg_df.to_excel(
                    writer, sheet_name="Aggregate", index=False
                )
        except Exception as e:
            messagebox.showwarning("엑셀 저장 경고", f"엑셀 저장 실패: {e}")
        messagebox.showinfo("저장 완료", f"CSV/엑셀 저장됨: {folder}")

    def _set_busy(self, busy: bool, text: str):
        self.lbl_status.config(text=text)
        if busy:
            self.progress.start(10)
        else:
            self.progress.stop()

    # --- 가상 조작 시뮬 레이어 ---
    def _open_rigged_dialog(self):
        if self.history_df is None or self.history_df.empty:
            messagebox.showwarning(
                "알림",
                "가상 조작 시뮬은 과거 히스토리가 필요합니다.\n먼저 CSV를 로드해 주세요.",
            )
            return

        if self.rig_win is not None and tk.Toplevel.winfo_exists(self.rig_win):
            self.rig_win.lift()
            self.rig_win.focus_force()
            return

        win = tk.Toplevel(self)
        win.title("가상 조작 시뮬레이터 (1등 인원 타겟 + GPU 옵션)")
        win.geometry("640x560")
        self.rig_win = win

        top = ttk.Frame(win)
        top.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(top, text="목표 1등 인원 최소").grid(row=0, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.rig_target_min, width=6).grid(
            row=0, column=1, sticky="w", padx=4
        )

        ttk.Label(top, text="목표 1등 인원 최대").grid(row=0, column=2, sticky="e")
        ttk.Entry(top, textvariable=self.rig_target_max, width=6).grid(
            row=0, column=3, sticky="w", padx=4
        )

        ttk.Label(top, text="샘플링 후보 개수").grid(row=1, column=0, sticky="e", pady=4)
        ttk.Entry(top, textvariable=self.rig_samples, width=10).grid(
            row=1, column=1, sticky="w", padx=4
        )

        btn = ttk.Button(top, text="검색 실행", command=self._run_rigged_search)
        btn.grid(row=1, column=2, columnspan=2, sticky="w", padx=6)

        # ★ 추가: 현실 구매자 수 입력
        self.rig_buyers = tk.IntVar(value=14000000)
        ttk.Label(top, text="현실 구매자 수").grid(row=4, column=0, sticky="e", pady=4)
        ttk.Entry(top, textvariable=self.rig_buyers, width=12).grid(
            row=4, column=1, sticky="w", padx=4
        )
        ttk.Label(top, text="(예: 14,000,000)").grid(row=4, column=2, sticky="w")

        # ★ 추가: 1인당 평균 게임 수 입력
        self.rig_avg_games = tk.DoubleVar(value=8.0)
        ttk.Label(top, text="1인당 평균 게임 수").grid(row=5, column=0, sticky="e", pady=4)
        ttk.Entry(top, textvariable=self.rig_avg_games, width=12).grid(
            row=5, column=1, sticky="w", padx=4
        )
        ttk.Label(top, text="(예: 8 게임)").grid(row=5, column=2, sticky="w")

        # ★ 가상 플레이어 수 입력
        ttk.Label(top, text="가상 플레이어 수").grid(row=2, column=0, sticky="e", pady=4)
        ttk.Entry(top, textvariable=self.rig_virtual_players, width=12).grid(
            row=2, column=1, sticky="w", padx=4
        )
        ttk.Label(top, text="(예: 400000)").grid(row=2, column=2, sticky="w")

        # ★ GPU 사용 옵션
        chk = ttk.Checkbutton(
            top,
            text="GPU 사용 (CuPy, 실험적)",
            variable=self.rig_use_gpu,
        )
        chk.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 2))

        if cp is None:
            chk.state(["disabled"])
            chk.config(text="GPU 사용 (CuPy 미설치)")

        self.rig_status_label = ttk.Label(win, text="대기 중")
        self.rig_status_label.pack(anchor="w", padx=10)

        frame_list = ttk.Frame(win)
        frame_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        cols = ["Rank", "Draw", "예상 1등 인원(λ)"]

        self.rig_tree = ttk.Treeview(
            frame_list, columns=cols, show="headings", height=16
        )
        vsb = ttk.Scrollbar(
            frame_list, orient="vertical", command=self.rig_tree.yview
        )
        hsb = ttk.Scrollbar(
            frame_list, orient="horizontal", command=self.rig_tree.xview
        )
        self.rig_tree.configure(
            yscrollcommand=vsb.set, xscrollcommand=hsb.set
        )

        self.rig_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        frame_list.rowconfigure(0, weight=1)
        frame_list.columnconfigure(0, weight=1)

        for c in cols:
            self.rig_tree.heading(c, text=c)
            self.rig_tree.column(c, width=160, anchor="center")

    def _run_rigged_search(self):
        if self.history_df is None or self.history_df.empty:
            messagebox.showwarning(
                "알림", "먼저 과거 CSV를 로드해야 가상 조작 시뮬이 가능합니다."
            )
            return

        try:
            tmin = max(0, int(self.rig_target_min.get()))
            tmax = max(tmin, int(self.rig_target_max.get()))
            samples = max(1000, int(self.rig_samples.get()))
            sim_players_val = max(1, int(self.rig_virtual_players.get()))
        except Exception:
            messagebox.showerror("오류", "입력 값이 잘못되었습니다.")
            return

        # HM 가중치(Balanced) 구하기
        try:
            w_bal, _ = compute_weights(
                self.history_df,
                lookback=None,
                strategy="Balanced(중립화)",
                exclude_recent=0,
            )
        except Exception:
            w_bal = None

        if self.rig_status_label is not None:
            self.rig_status_label.config(
                text=f"가상 플레이어 풀 구성 + 검색 중... (샘플 {samples:,}개, 가상 플레이어 {sim_players_val:,}명)"
            )

        def task():
            # 세트 편집 탭에서 사용자 세트 읽기 (취향 반영용)
            user_sets = None
            txt_sets = self.text_sets.get("1.0", tk.END)
            if txt_sets.strip():
                try:
                    user_sets = parse_sets_from_text(txt_sets)
                except Exception:
                    user_sets = None

            # 1) HM + 휴먼 버프 섞어서 '현실적 인기 분포' 만들기
            local_w = compute_realistic_popularity_weights(
                self.history_df,
                hm_weights=w_bal,
                user_sets=user_sets,
            )

            # ★ 최근 N회 번호 회피 세트 (예: 최근 20회)
            try:
                recent_N = 20
                tail = self.history_df.tail(recent_N)
                recent_exclude = set(int(v) for v in np.unique(tail.values) if 1 <= int(v) <= 45)
            except Exception:
                recent_exclude = set()

            sim_players = sim_players_val

            # 2) 가상 플레이어 수: 사용자가 지정한 값 그대로 사용
            sim_players = sim_players_val

            # 3) 가상 플레이어 풀 생성 (전구간 36코어 사용)
            ticket_pool = build_synthetic_player_pool(
                sim_players,
                local_w,
                workers=36,   # 36 프로세스 풀
            )

            # 4) 실제 전국 판매량 계산 (구매자수 × 평균게임수)
            buyers = int(self.rig_buyers.get())
            avg_games = float(self.rig_avg_games.get())
            REAL_TICKETS = buyers * avg_games

            scale_factor = REAL_TICKETS / float(sim_players)

            # 5) GPU 사용 여부 결정
            use_gpu = bool(self.rig_use_gpu.get() and cp is not None)

            xs: list[tuple[list[int], float]] = []

            if use_gpu:
                # GPU 경로 (벡터화)
                try:
                    xs = _rigged_candidate_gpu(
                        samples,
                        local_w,
                        ticket_pool,
                        scale_factor,
                        tmin,
                        tmax,
                    )
                except Exception as e:
                    use_gpu = False
                    xs = []
                    if self.rig_status_label is not None:
                        self.after(
                            0,
                            lambda err=e: self.rig_status_label.config(
                                text=f"GPU 경로 실패, CPU로 폴백: {err}"
                            ),
                        )

            if not use_gpu:
                # 기존 CPU 멀티프로세스 경로
                max_workers = 36
                per_worker = samples // max_workers
                remainder = samples % max_workers

                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futures = []
                    for i in range(max_workers):
                        n_i = per_worker + (1 if i < remainder else 0)
                        if n_i <= 0:
                            continue
                        futures.append(
                            ex.submit(
                                _rigged_candidate_chunk,
                                n_i,
                                local_w,
                                ticket_pool,
                                scale_factor,
                                tmin,
                                tmax,
                            )
                        )
                    for fut in as_completed(futures):
                        part = fut.result()
                        if part:
                            xs.extend(part)

            # 후보 정렬 및 상위 200개 선택
            if not xs:
                rows = []
            else:
                center = 0.5 * (tmin + tmax)
                xs_sorted = sorted(xs, key=lambda d: abs(d[1] - center))
                rows = xs_sorted[:200]

            self.after(0, lambda: self._update_rigged_tree(rows, tmin, tmax, samples, sim_players))

        threading.Thread(target=task, daemon=True).start()

    def _update_rigged_tree(
        self,
        rows: list[tuple[list[int], float]],
        tmin: int,
        tmax: int,
        samples: int,
        sim_players: int,
    ):
        if self.rig_tree is None:
            return
        self.rig_tree.delete(*self.rig_tree.get_children())
        for idx, (draw, lam) in enumerate(rows, start=1):
            self.rig_tree.insert(
                "",
                tk.END,
                values=[
                    idx,
                    " ".join(map(str, sorted(draw))),
                    f"{lam:5.2f}",
                ],
            )
        if self.rig_status_label is not None:
            if not rows:
                self.rig_status_label.config(
                    text=f"검색 완료 — 범위 [{tmin}~{tmax}]에 해당하는 후보 없음 "
                         f"(후보 샘플 {samples:,}개, 가상 플레이어 {sim_players:,}명)"
                )
            else:
                self.rig_status_label.config(
                    text=f"검색 완료 — 후보 {len(rows)}개 "
                         f"(후보 샘플 {samples:,}개, 가상 플레이어 {sim_players:,}명, 목표 [{tmin}~{tmax}])"
                )

    # --- 분포 분석 페이지 ---
    def _build_viz_page(self):
        top = self.page_viz
        ctrl = ttk.Frame(top)
        ctrl.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(
            ctrl,
            text="히스토리 빈도(1~45)",
            command=self._plot_history_freq,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            ctrl,
            text="추출 결과 빈도(번호 추출기 탭)",
            command=self._plot_generated_freq,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            ctrl,
            text="세트 편집 빈도",
            command=self._plot_sets_freq,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            ctrl,
            text="짝/홀/구간 분포(세트 편집)",
            command=self._plot_sets_parity_range,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            ctrl,
            text="MQLE 조화/다양성(번호 추출기)",
            command=self._plot_mqle_qh_div,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            ctrl,
            text="3D MQLE 맵(번호 추출기)",
            command=self._show_3d_mqle_map,
        ).pack(side=tk.LEFT, padx=4)

        self.fig_frame = ttk.Frame(top)
        self.fig_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.canvas: FigureCanvasTkAgg | None = None

    def _clear_canvas(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

    def _embed_fig(self, fig: Figure):
        self._clear_canvas()
        self.canvas = FigureCanvasTkAgg(fig, master=self.fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_bar(self, xs, ys, title, xlabel, ylabel):
        fig = Figure(figsize=(9, 4))
        ax = fig.add_subplot(111)
        ax.bar(xs, ys)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if len(xs) > 0 and isinstance(xs[0], int) and len(xs) > 20:
            ax.set_xticks(xs)
        if len(xs) > 20:
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
                tick.set_ha("right")
        self._embed_fig(fig)

    def _parse_text_numbers(self, text_widget: tk.Text):
        try:
            sets_ = parse_sets_from_text(text_widget.get("1.0", tk.END))
        except Exception as e:
            messagebox.showerror("오류", str(e))
            return None
        return sets_

    def _plot_history_freq(self):
        if self.history_df is None or self.history_df.empty:
            messagebox.showwarning("알림", "과거 CSV를 먼저 불러오세요.")
            return
        cnt = np.zeros(46, dtype=int)
        for v in self.history_df.values.flatten():
            if 1 <= v <= 45:
                cnt[int(v)] += 1
        xs = list(range(1, 46))
        ys = [int(cnt[i]) for i in xs]
        self._plot_bar(xs, ys, "히스토리 빈도(번호별 출현 횟수)", "번호", "횟수")

    def _plot_generated_freq(self):
        sets_ = self._parse_text_numbers(self.text_generate)
        if sets_ is None:
            return
        cnt = np.zeros(46, dtype=int)
        for s in sets_:
            for v in s:
                cnt[v] += 1
        xs = list(range(1, 46))
        ys = [int(cnt[i]) for i in xs]
        self._plot_bar(xs, ys, "번호 추출기 결과 빈도", "번호", "출현 수")

    def _plot_sets_freq(self):
        sets_ = self._parse_text_numbers(self.text_sets)
        if sets_ is None:
            return
        cnt = np.zeros(46, dtype=int)
        for s in sets_:
            for v in s:
                cnt[v] += 1
        xs = list(range(1, 46))
        ys = [int(cnt[i]) for i in xs]
        self._plot_bar(xs, ys, "세트 편집 탭 빈도", "번호", "출현 수")

    def _plot_sets_parity_range(self):
        sets_ = self._parse_text_numbers(self.text_sets)
        if sets_ is None:
            return
        even_counts = 0
        odd_counts = 0
        low = mid = high = 0
        for s in sets_:
            for v in s:
                if v % 2 == 0:
                    even_counts += 1
                else:
                    odd_counts += 1
                if 1 <= v <= 20:
                    low += 1
                elif 21 <= v <= 35:
                    mid += 1
                else:
                    high += 1
        xs = ["Even", "Odd", "Low(1-20)", "Mid(21-35)", "High(36-45)"]
        ys = [even_counts, odd_counts, low, mid, high]
        self._plot_bar(xs, ys, "짝/홀/구간 분포 (세트 편집)", "구분", "개수")

    def _plot_mqle_qh_div(self):
        sets_ = self._parse_text_numbers(self.text_generate)
        if sets_ is None:
            return

        weights = getattr(self, "history_weights", None)

        qh_scores = []
        for s in sets_:
            try:
                qh_scores.append(_qh_score(s, weights))
            except Exception:
                qh_scores.append(0.0)

        div_scores = []
        for i, s in enumerate(sets_):
            sset = set(s)
            overlap = 0
            for j, t in enumerate(sets_):
                if i == j:
                    continue
                overlap += len(sset & set(t))
            div_scores.append(overlap)

        labels = [f"S{i+1}" for i in range(len(sets_))]
        xs = np.arange(len(sets_))

        fig = Figure(figsize=(9, 6))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.bar(xs, qh_scores)
        ax1.set_title("MQLE / QH 기반 조화 점수 (번호 추출기 결과)")
        ax1.set_xticks(xs)
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_ylabel("QH Score")

        ax2.bar(xs, div_scores)
        ax2.set_title("세트 간 겹침 정도 (작을수록 다양)")
        ax2.set_xticks(xs)
        ax2.set_xticklabels(labels, rotation=45, ha="right")
        ax2.set_ylabel("겹치는 번호 개수 합")

        self._embed_fig(fig)

    def _show_3d_mqle_map(self):
        sets_ = self._parse_text_numbers(self.text_generate)
        if sets_ is None or len(sets_) == 0:
            messagebox.showwarning(
                "알림", "먼저 번호 추출기 탭에서 세트를 생성하세요."
            )
            return

        weights = getattr(self, "history_weights", None)

        qh_scores = []
        for s in sets_:
            try:
                qh_scores.append(_qh_score(s, weights))
            except Exception:
                qh_scores.append(0.0)

        div_penalty = []
        for i, s in enumerate(sets_):
            sset = set(s)
            overlap = 0
            for j, t in enumerate(sets_):
                if i == j:
                    continue
                overlap += len(sset & set(t))
            div_penalty.append(overlap)

        avg_nums = [sum(s) / len(s) for s in sets_]

        X = np.array(qh_scores)
        Y = np.array(div_penalty)
        Z = np.array(avg_nums)

        labels = [f"S{i+1}" for i in range(len(sets_))]

        win = tk.Toplevel(self)
        win.title("3D MQLE 맵 - QH/다양성/평균 번호")
        win.geometry("800x600")

        fig = Figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(X, Y, Z)

        ax.set_xlabel("QH 조화 점수")
        ax.set_ylabel("다양성 penalty (겹침, 작을수록 다양)")
        ax.set_zlabel("세트 평균 번호")
        ax.set_title("3D MQLE 맵 (번호 추출기 결과)")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def update(frame):
            angle = frame % 360
            ax.view_init(elev=25, azim=angle)
            canvas.draw()
            return scatter,

        ani = FuncAnimation(fig, update, frames=360, interval=80, blit=False)
        win.ani = ani

    # --- HELP 페이지 ---
    def _build_help_page(self):
        top = self.page_help
        frame = ttk.Frame(top)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        txt = tk.Text(frame, wrap="word")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=scroll.set)

        txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        help_text = """
[1. 프로그램 전체 개요]

이 프로그램은 로또 6/45를 대상으로

  1) 내가 원하는 번호 세트를 직접 관리하고,
  2) 여러 가지 '번호 생성 알고리듬'으로 추천 번호를 만들고,
  3) 실제 추첨기를 가정한 몬테카를로(Monte Carlo) 시뮬레이션으로 통계적 성능을 확인하고,
  4) 번호 분포 / 짝·홀 / 구간 / MQLE 조화·다양성을 시각화해서 분석하고,
  5) '만약 조작이 있다면 1등 인원을 어떻게 맞출까?'를 가상으로 실험(리깅 시뮬레이션),
  6) 가상 조작 시뮬의 후보 번호 생성 일부를 GPU(CuPy)로 벡터화 가속

하는 연구/놀이용 도구입니다.

※ 매우 중요
- 실제 로또는 '완전 난수'를 목표로 설계된 시스템입니다.
- 여기 나오는 모든 알고리듬, 양자사운드, AI, 가상조작 시뮬, GPU 가속은
  "수학적인 장난감 + 취향 정리용"일 뿐,
  진짜 수학적 기대값(당첨 확률)을 유의미하게 올려주지 못합니다.
- 반드시 여윳돈 + 재미·연구용으로만 활용하세요.

(이하 HELP 텍스트는 필요하면 자유롭게 확장)
"""
        txt.insert("1.0", help_text)
        txt.config(state="disabled")
        self.help_text_widget = txt


if __name__ == "__main__":
    app = LottoApp()
    app.mainloop()
