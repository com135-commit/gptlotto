#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lotto 6/45 GUI Simulator (KR) — Sets Editor + Number Generator + MP Simulator
-----------------------------------------------------------------------------
- Tkinter GUI: manage sets, generate new ones, and run simulations.
- Monte Carlo engine uses multiprocessing (up to 36 workers).
- Exports CSV/Excel and optional PNG plots.
- Compatible with Python 3.9+; requires numpy, pandas, matplotlib, xlsxwriter.
"""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

# ------------------ Utilities ------------------
def parse_sets_from_text(text: str):
    sets = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        nums = [int(x) for x in line.replace(',', ' ').split()]
        if len(nums) != 6:
            raise ValueError(f"각 줄은 정확히 6개 숫자여야 합니다: '{line}'")
        if any(n < 1 or n > 45 for n in nums):
            raise ValueError(f"숫자는 1~45 범위여야 합니다: '{line}'")
        if len(set(nums)) != 6:
            raise ValueError(f"중복 없는 6개여야 합니다: '{line}'")
        sets.append(sorted(nums))
    if not sets:
        raise ValueError("최소 1개 세트가 필요합니다.")
    return sets

def sets_to_text(sets):
    return '\n'.join(' '.join(map(str, s)) for s in sets)

def default_sets():
    return [
        [7, 14, 21, 27, 33, 41],
        [4, 9, 18, 26, 32, 40],
        [3, 10, 15, 25, 30, 43],
        [6, 13, 22, 29, 35, 44],
        [8, 17, 23, 31, 37, 42],
        [2, 5, 11, 19, 28, 39],
        [5, 8, 13, 21, 34, 45],
        [3, 9, 12, 24, 36, 42],
        [1, 7, 16, 22, 28, 38],
        [4, 11, 18, 27, 35, 43],
    ]

# ------------------ Number Generators ------------------
def generate_random_sets(n_sets:int, avoid_duplicates=True):
    rng = np.random.default_rng()
    result = []
    seen = set()
    while len(result) < n_sets:
        s = tuple(sorted(rng.choice(np.arange(1,46), size=6, replace=False).tolist()))
        if avoid_duplicates and s in seen:
            continue
        seen.add(s)
        result.append(list(s))
    return result

def generate_pattern_sets(n_sets:int, even_target:int|None=None, low_mid_high=(2,2,2), include_multiples=(0,0), avoid_duplicates=True):
    if sum(low_mid_high) != 6:
        raise ValueError("구간 합계가 6이어야 합니다.")
    rng = np.random.default_rng()
    result = []
    seen = set()
    lows = np.arange(1,21)
    mids = np.arange(21,36)
    highs = np.arange(36,46)
    while len(result) < n_sets:
        pool = []
        pool.extend(rng.choice(lows, size=low_mid_high[0], replace=False).tolist())
        pool.extend(rng.choice(mids, size=low_mid_high[1], replace=False).tolist())
        pool.extend(rng.choice(highs, size=low_mid_high[2], replace=False).tolist())
        s = sorted(pool)
        if even_target is not None:
            evens = sum(1 for x in s if x%2==0)
            if evens != even_target:
                continue
        if include_multiples[0] > 0:
            if sum(1 for x in s if x%3==0) < include_multiples[0]:
                continue
        if include_multiples[1] > 0:
            if sum(1 for x in s if x%7==0) < include_multiples[1]:
                continue
        t = tuple(s)
        if avoid_duplicates and t in seen:
            continue
        seen.add(t)
        result.append(s)
    return result

# ------------------ Simulation Core (MP) ------------------
def simulate_chunk(draws:int, batch:int, seed:int, sets_array:list[np.ndarray], include_bonus=True):
    rng = np.random.default_rng(seed)
    S = len(sets_array)
    match_bins = np.zeros((S, 7), dtype=np.int64)
    bonus5_counts = np.zeros(S, dtype=np.int64)
    agg_bins = np.zeros(7, dtype=np.int64)
    agg_bonus5 = 0
    nums_all = np.arange(1, 46, dtype=np.int16)
    remaining = draws
    while remaining > 0:
        b = min(batch, remaining)
        remaining -= b
        draws_batch = np.empty((b, 6), dtype=np.int16)
        for i in range(b):
            draws_batch[i] = rng.choice(nums_all, size=6, replace=False)
        draws_batch.sort(axis=1)
        if include_bonus:
            bonus_batch = np.empty(b, dtype=np.int16)
            for i in range(b):
                while True:
                    c = rng.integers(1, 46, dtype=np.int16)
                    row = draws_batch[i]
                    pos = np.searchsorted(row, c)
                    if not (pos < 6 and row[pos] == c):
                        bonus_batch[i] = c
                        break
        else:
            bonus_batch = None
        for idx, s in enumerate(sets_array):
            matches_vec = np.isin(draws_batch, s).sum(axis=1)
            counts = np.bincount(matches_vec, minlength=7)
            match_bins[idx, :7] += counts
            agg_bins[:7] += counts
            if include_bonus:
                bonus_in_set = np.isin(bonus_batch, s)
                not_in_row = np.ones(b, dtype=bool)
                for j in range(b):
                    pos = np.searchsorted(draws_batch[j], bonus_batch[j])
                    not_in_row[j] = not (pos < 6 and draws_batch[j][pos] == bonus_batch[j])
                cond = (matches_vec == 5) & bonus_in_set & not_in_row
                bonus5_counts[idx] += int(cond.sum())
                agg_bonus5 += int(cond.sum())
    return match_bins, bonus5_counts, agg_bins, agg_bonus5

def run_simulation(draws:int, batch:int, workers:int, seed:int|None, sets:list[list[int]]):
    W = max(1, min(36, workers))
    sets_array = [np.array(s, dtype=np.int16) for s in sets]
    per_worker = draws // W
    remainder = draws % W
    parts = []
    with ProcessPoolExecutor(max_workers=W) as ex:
        futures = []
        for i in range(W):
            draws_i = per_worker + (1 if i < remainder else 0)
            if draws_i == 0:
                continue
            seed_i = None if seed is None else (seed + i*9973)
            futures.append(ex.submit(simulate_chunk, draws_i, batch, seed_i, sets_array, True))
        for fut in as_completed(futures):
            parts.append(fut.result())
    match_bins = None; bonus5_counts=None; agg_bins=None; agg_bonus5=0
    for mb, b5, ab, a5 in parts:
        match_bins = mb if match_bins is None else (match_bins + mb)
        bonus5_counts = b5 if bonus5_counts is None else (bonus5_counts + b5)
        agg_bins = ab if agg_bins is None else (agg_bins + ab)
        agg_bonus5 += a5
    total = draws
    rows = []
    for idx, s in enumerate(sets):
        row = {"Set": f"Set_{idx+1:02d}", "Numbers": " ".join(map(str, s))}
        for m in range(7):
            row[f"match_{m}_count"] = int(match_bins[idx, m])
            row[f"match_{m}_prob"]  = match_bins[idx, m] / total
        row["match_5plusbonus_count"] = int(bonus5_counts[idx])
        row["match_5plusbonus_prob"]  = bonus5_counts[idx] / total
        row["≥3_match_prob"] = match_bins[idx, 3:7].sum() / total
        rows.append(row)
    per_set_df = pd.DataFrame(rows)
    agg_row = {"Set": "AGG_ALL", "Numbers": "—"}
    for m in range(7):
        agg_row[f"match_{m}_count"] = int(agg_bins[m])
        agg_row[f"match_{m}_prob"]  = agg_bins[m] / (total * len(sets))
    agg_row["match_5plusbonus_count"] = int(agg_bonus5)
    agg_row["match_5plusbonus_prob"]  = agg_bonus5 / (total * len(sets))
    agg_row["≥3_match_prob"] = agg_bins[3:7].sum() / (total * len(sets))
    agg_df = pd.DataFrame([agg_row])
    return per_set_df, agg_df

# ------------------ GUI ------------------
class LottoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lotto 6/45 Simulator (GUI, up to 36 cores)")
        self.geometry("940x700")
        self.resizable(True, True)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.page_sets = ttk.Frame(self.notebook)
        self.page_generate = ttk.Frame(self.notebook)
        self.page_sim = ttk.Frame(self.notebook)

        self.notebook.add(self.page_sets, text="세트 편집")
        self.notebook.add(self.page_generate, text="번호 추출기")
        self.notebook.add(self.page_sim, text="시뮬레이션")

        self._build_sets_page()
        self._build_generate_page()
        self._build_sim_page()

        self.text_sets.insert("1.0", sets_to_text(default_sets()))

    def _build_sets_page(self):
        top = self.page_sets
        ttk.Label(top, text="세트 목록 (한 줄에 6개 숫자, 공백/쉼표 구분)").pack(anchor="w", padx=10, pady=6)
        self.text_sets = tk.Text(top, height=20, wrap="none")
        self.text_sets.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        btn_frame = ttk.Frame(top); btn_frame.pack(fill=tk.X, padx=10, pady=6)
        ttk.Button(btn_frame, text="불러오기(.txt)", command=self._load_sets).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="저장하기(.txt)", command=self._save_sets_txt).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="정렬/중복제거", command=self._normalize_sets).pack(side=tk.LEFT, padx=4)

    def _load_sets(self):
        path = filedialog.askopenfilename(filetypes=[("Text", "*.txt"), ("All","*.*")])
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            self.text_sets.delete("1.0", tk.END)
            self.text_sets.insert("1.0", f.read())

    def _save_sets_txt(self):
        try:
            sets = parse_sets_from_text(self.text_sets.get("1.0", tk.END))
        except Exception as e:
            messagebox.showerror("오류", str(e)); return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text","*.txt")])
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            f.write(sets_to_text(sets))
        messagebox.showinfo("저장 완료", f"세트 {len(sets)}개 저장")

    def _normalize_sets(self):
        try:
            sets = parse_sets_from_text(self.text_sets.get("1.0", tk.END))
        except Exception as e:
            messagebox.showerror("오류", str(e)); return
        uniq = sorted({tuple(s) for s in sets})
        self.text_sets.delete("1.0", tk.END)
        self.text_sets.insert("1.0", sets_to_text([list(s) for s in uniq]))
        messagebox.showinfo("정리 완료", f"세트 {len(uniq)}개")

    def _build_generate_page(self):
        top = self.page_generate
        frm = ttk.Frame(top); frm.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(frm, text="생성 개수").grid(row=0, column=0, sticky="w")
        self.gen_count = tk.IntVar(value=10)
        ttk.Entry(frm, textvariable=self.gen_count, width=8).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(frm, text="짝수 개수(선택):").grid(row=0, column=2, sticky="e")
        self.gen_even = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.gen_even, width=6).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(frm, text="구간 분포 (저/중/고)").grid(row=1, column=0, sticky="w", pady=6)
        self.gen_low = tk.IntVar(value=2); self.gen_mid = tk.IntVar(value=2); self.gen_high = tk.IntVar(value=2)
        ttk.Entry(frm, textvariable=self.gen_low, width=5).grid(row=1, column=1, sticky="w")
        ttk.Entry(frm, textvariable=self.gen_mid, width=5).grid(row=1, column=2, sticky="w")
        ttk.Entry(frm, textvariable=self.gen_high, width=5).grid(row=1, column=3, sticky="w")
        ttk.Label(frm, text="배수 포함 (3의/7의 최소개수)").grid(row=2, column=0, sticky="w", pady=6)
        self.gen_m3 = tk.IntVar(value=0); self.gen_m7 = tk.IntVar(value=0)
        ttk.Entry(frm, textvariable=self.gen_m3, width=5).grid(row=2, column=1, sticky="w")
        ttk.Entry(frm, textvariable=self.gen_m7, width=5).grid(row=2, column=2, sticky="w")
        btns = ttk.Frame(top); btns.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(btns, text="무작위 생성", command=self._gen_random).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="패턴 기반 생성", command=self._gen_pattern).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="세트 페이지에 추가", command=self._append_to_sets).pack(side=tk.LEFT, padx=4)
        self.text_generate = tk.Text(top, height=18, wrap="none")
        self.text_generate.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

    def _gen_random(self):
        n = max(1, self.gen_count.get())
        arr = generate_random_sets(n, avoid_duplicates=True)
        self.text_generate.delete("1.0", tk.END)
        self.text_generate.insert("1.0", sets_to_text(arr))

    def _gen_pattern(self):
        n = max(1, self.gen_count.get())
        even_str = self.gen_even.get().strip()
        even_target = None if even_str=="" else int(even_str)
        low, mid, high = self.gen_low.get(), self.gen_mid.get(), self.gen_high.get()
        m3, m7 = self.gen_m3.get(), self.gen_m7.get()
        try:
            arr = generate_pattern_sets(n, even_target=even_target, low_mid_high=(low,mid,high), include_multiples=(m3,m7))
        except Exception as e:
            messagebox.showerror("오류", str(e)); return
        self.text_generate.delete("1.0", tk.END)
        self.text_generate.insert("1.0", sets_to_text(arr))

    def _append_to_sets(self):
        try:
            sets_new = parse_sets_from_text(self.text_generate.get("1.0", tk.END))
        except Exception as e:
            messagebox.showerror("오류", str(e)); return
        current = self.text_sets.get("1.0", tk.END)
        base = []
        if current.strip():
            try:
                base = parse_sets_from_text(current)
            except Exception as e:
                messagebox.showerror("오류", f"세트 페이지 오류: {e}"); return
        merged = [tuple(s) for s in base] + [tuple(s) for s in sets_new]
        uniq = sorted(list({t for t in merged}))
        self.text_sets.delete("1.0", tk.END)
        self.text_sets.insert("1.0", sets_to_text([list(t) for t in uniq]))
        messagebox.showinfo("추가 완료", f"세트 {len(sets_new)}개 추가됨 (중복 제거 후 총 {len(uniq)}개)")

    def _build_sim_page(self):
        top = self.page_sim
        frm = ttk.Frame(top); frm.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(frm, text="총 추첨 횟수(draws)").grid(row=0, column=0, sticky="w")
        self.sim_draws = tk.IntVar(value=2_000_000)
        ttk.Entry(frm, textvariable=self.sim_draws, width=12).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(frm, text="배치(batch)").grid(row=0, column=2, sticky="e")
        self.sim_batch = tk.IntVar(value=200_000)
        ttk.Entry(frm, textvariable=self.sim_batch, width=10).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Label(frm, text="워커 수(workers, 최대 36)").grid(row=0, column=4, sticky="e")
        self.sim_workers = tk.IntVar(value=8)
        ttk.Entry(frm, textvariable=self.sim_workers, width=8).grid(row=0, column=5, sticky="w", padx=6)
        ttk.Label(frm, text="Seed(선택)").grid(row=1, column=0, sticky="w", pady=6)
        self.sim_seed = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.sim_seed, width=12).grid(row=1, column=1, sticky="w")
        btns = ttk.Frame(top); btns.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(btns, text="시뮬레이션 실행", command=self._run_sim).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="CSV/Excel로 저장", command=self._save_outputs).pack(side=tk.LEFT, padx=6)
        self.progress = ttk.Progressbar(top, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=10, pady=6)
        self.lbl_status = ttk.Label(top, text="대기 중"); self.lbl_status.pack(anchor="w", padx=10)
        cols = ["Set","Numbers"] + [f"match_{m}_count" for m in range(7)] + [f"match_{m}_prob" for m in range(7)] + ["match_5plusbonus_count","match_5plusbonus_prob","≥3_match_prob"]
        self.tree = ttk.Treeview(top, columns=cols, show="headings", height=16)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=110 if c!="Numbers" else 180, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        self.per_set_df = None; self.agg_df = None

    def _run_sim(self):
        try:
            sets = parse_sets_from_text(self.text_sets.get("1.0", tk.END))
        except Exception as e:
            messagebox.showerror("오류", str(e)); return
        draws = max(1, self.sim_draws.get())
        batch = max(1, self.sim_batch.get())
        workers = max(1, min(36, self.sim_workers.get()))
        seed = self.sim_seed.get().strip()
        seed_val = None if seed=="" else int(seed)
        def task():
            try:
                self._set_busy(True, "시뮬레이션 실행 중...")
                per_set_df, agg_df = run_simulation(draws, batch, workers, seed_val, sets)
                self.per_set_df = per_set_df; self.agg_df = agg_df
                self._populate_tree(per_set_df, agg_df)
                self._set_busy(False, f"완료: draws={draws:,}, workers={workers}, batch={batch:,}")
            except Exception as e:
                self._set_busy(False, "오류 발생"); messagebox.showerror("오류", str(e))
        threading.Thread(target=task, daemon=True).start()

    def _populate_tree(self, per_set_df:pd.DataFrame, agg_df:pd.DataFrame):
        self.tree.delete(*self.tree.get_children())
        for _, row in per_set_df.iterrows():
            values = [row.get(col, "") for col in self.tree["columns"]]
            self.tree.insert("", tk.END, values=values)
        row = agg_df.iloc[0].to_dict()
        values = [row.get(col, "") for col in self.tree["columns"]]
        self.tree.insert("", tk.END, values=values)

    def _save_outputs(self):
        if self.per_set_df is None or self.agg_df is None:
            messagebox.showwarning("알림", "먼저 시뮬레이션을 실행하세요."); return
        folder = filedialog.askdirectory()
        if not folder: return
        per_csv = os.path.join(folder, "lotto_per_set.csv")
        agg_csv = os.path.join(folder, "lotto_aggregate.csv")
        self.per_set_df.to_csv(per_csv, index=False)
        self.agg_df.to_csv(agg_csv, index=False)
        try:
            xlsx = os.path.join(folder, "lotto_results.xlsx")
            with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
                self.per_set_df.to_excel(writer, sheet_name="PerSet", index=False)
                self.agg_df.to_excel(writer, sheet_name="Aggregate", index=False)
        except Exception as e:
            messagebox.showwarning("엑셀 저장 경고", f"엑셀 저장 실패: {e}")
        messagebox.showinfo("저장 완료", f"CSV/엑셀 저장됨: {folder}")

    def _set_busy(self, busy:bool, text:str):
        self.lbl_status.config(text=text)
        if busy: self.progress.start(10)
        else: self.progress.stop()

if __name__ == "__main__":
    app = LottoApp()
    app.mainloop()
