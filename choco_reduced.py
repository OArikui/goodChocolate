#!/usr/bin/env python3
"""
choco_reduced.py  -  円錐台チョコ形状最適化(手のパラメータ縮約版) v0.2

■ 縮約の考え方
  詳細モデル(v0.1)の溶解量は、ヘルツ接触 A ∝ (N R_eff / E*)^k と
  非定常熱伝導 M ∝ A ΔT √t / (1/e_c + 1/e_s) から

      M = S_hand · [ cosφ / sin(α+φ) ]^k · R_eff^k  ·  (m g SF/2)^k · √t / E_melt
          ~~~~~~~   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          手のスケール   形状 × (φ, R_f) だけに依存

  φ = arctan μ(摩擦角)、R_eff = (1/R_f + 1/r_m)^-1、r_m = 側面の平均半径。
  E, t_s, T_s, f_b(弾性率・軟組織厚・皮膚温度・血流)は定数 S_hand にまとまり、
  形状の「順位」には影響しない(絶対量を掛け算するだけ)。

  → 手の情報は (φ, R_f) の2つ + 絶対量用の1点較正(基準円柱の実測溶解量 M_ref0)に縮約。
  → 評価指標は「同体積の基準円柱に対する溶解量比 M_rel」(S_hand が消える)。

角度 α: 側面の鉛直からの傾き。α > 0 で上が広い(逆台形)。
"""
import csv
import math
import warnings

import numpy as np
import matplotlib
from matplotlib.figure import Figure

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
except ImportError:
    tk = None

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Yu Gothic", "Meiryo", "Hiragino Sans", "Noto Sans CJK JP",
    "IPAexGothic", "TakaoPGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

DEFAULT_P = dict(V=15.0, rmin=5.0, rmax=25.0, k=2 / 3, ref_h=0.0, Mref0=0.010)
DEFAULT_HAND = dict(mu_m=0.5, mu_cv=25.0, Rf_m=8.0, Rf_cv=15.0, ns=500)
DEFAULT_GRID = dict(a0=-10.0, a1=35.0, na=46, h0=8.0, h1=40.0, nh=33)


# ============================ 計算コア ======================================
def frustum_geometry(V, h, alpha):
    """体積 V 一定・高さ h・傾き α から下底 rb、上底 rt を求める。"""
    d = h * np.tan(alpha)
    K = 3 * V / (np.pi * h)
    disc = 12 * K - 3 * d ** 2
    with np.errstate(invalid="ignore"):
        rb = (-3 * d + np.sqrt(disc)) / 6
    return rb, rb + d, disc


def g_index(rm, alpha, mu, Rf, k):
    """形状×手(φ, R_f)のみに依存する溶解量の相対尺度(手のスケールは含まない)。"""
    den = np.sin(alpha) + mu * np.cos(alpha)        # = sin(α+φ)/cosφ
    Reff = 1.0 / (1.0 / Rf + 1.0 / rm)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0.02, (Reff / den) ** k, np.nan)


def ref_radius(p):
    V = p["V"] * 1e-6
    if p["ref_h"] > 0:                               # 指定高さの円柱
        return math.sqrt(V / (math.pi * p["ref_h"] * 1e-3))
    return (V / (2 * math.pi)) ** (1 / 3)            # 高さ = 直径 の円柱


def sample_hand(h, rng, mean_only):
    n = 1 if mean_only else int(h["ns"])
    def ln(m, cv):
        if n == 1 or cv <= 0:
            return np.full(n, float(m))
        s2 = math.log(1 + (cv / 100) ** 2)
        return rng.lognormal(math.log(m) - s2 / 2, math.sqrt(s2), n)
    return ln(h["mu_m"], h["mu_cv"]), ln(h["Rf_m"], h["Rf_cv"]) * 1e-3


def relative_index(p, mu, Rf, alphas_deg, hs_mm):
    """M_rel(基準円柱=1)を (ns, nh, na) で返す。"""
    al = np.deg2rad(np.asarray(alphas_deg, float))[None, :]
    h = (np.asarray(hs_mm, float) * 1e-3)[:, None]
    rb, rt, disc = frustum_geometry(p["V"] * 1e-6, h, al)
    rmin, rmax = p["rmin"] * 1e-3, p["rmax"] * 1e-3
    valid = ((disc > 0) & (rb >= rmin) & (rt >= rmin)
             & (np.maximum(rb, rt) <= rmax))
    rb, rt = np.where(valid, rb, np.nan), np.where(valid, rt, np.nan)
    rm = (rb + rt) / 2
    mu3, Rf3 = mu[:, None, None], Rf[:, None, None]
    g = g_index(rm[None], al[None], mu3, Rf3, p["k"])
    g0 = g_index(ref_radius(p), 0.0, mu, Rf, p["k"])
    return g / g0[:, None, None], rb, rt


def run_all(p, hand, grid, mean_only, objective, seed=0):
    rng = np.random.default_rng(seed)
    mu, Rf = sample_hand(hand, rng, mean_only)
    al = np.linspace(grid["a0"], grid["a1"], int(grid["na"]))
    hs = np.linspace(grid["h0"], grid["h1"], int(grid["nh"]))
    M, rb, rt = relative_index(p, mu, Rf, al, hs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate = np.isfinite(M).mean(axis=0)
        mean = np.nanmean(M, axis=0)
        p95 = np.nanpercentile(M, 95, axis=0) if len(mu) > 1 else mean.copy()
    ok = rate >= 0.9
    mean, p95 = np.where(ok, mean, np.nan), np.where(ok, p95, np.nan)
    score = mean if objective == "mean" else p95
    if not np.isfinite(score).any():
        raise ValueError("実行可能な形状がありません。半径制限や探索範囲を見直してください。")
    ih, ia = np.unravel_index(np.nanargmin(score), score.shape)
    return dict(M=M, rb=rb, rt=rt, al=al, hs=hs, mean=mean, p95=p95, score=score,
                ih=ih, ia=ia, mu=mu, Rf=Rf, objective=objective, p=p)


# ============================ 作図・出力 ====================================
def draw(fig, o):
    fig.clear()
    p, al, hs, ih, ia = o["p"], o["al"], o["hs"], o["ih"], o["ia"]
    lab = "平均" if o["objective"] == "mean" else "95%点"
    ax1, ax2, ax3, ax4 = fig.subplots(2, 2).ravel()

    pc = ax1.pcolormesh(al, hs, o["score"], shading="auto", cmap="viridis")
    fig.colorbar(pc, ax=ax1, label=f"M_rel {lab}(基準円柱=1)")
    ax1.plot(al[ia], hs[ih], "r*", ms=14)
    ax1.set(xlabel="傾き α [deg](+ で上が広い)", ylabel="高さ h [mm]", title="形状マップ(★=最適)")

    M = o["M"][:, ih, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo, md, hi = (np.nanpercentile(M, q, axis=0) for q in (5, 50, 95))
    ax2.fill_between(al, lo, hi, alpha=0.25, label="5–95%(手の個人差)")
    ax2.plot(al, md, label="中央値")
    ax2.axhline(1, color="gray", ls=":", label="基準円柱")
    ax2.axvline(al[ia], color="r", ls="--")
    ax2.set(xlabel="傾き α [deg]", ylabel="M_rel", title=f"h = {hs[ih]:.1f} mm のスライス")
    ax2.legend(fontsize=8)

    # 手の軸(摩擦角 φ)に沿った性能: 最適形状は誰の手に有利か
    phi = np.linspace(10, 45, 40)
    mu_ax = np.tan(np.deg2rad(phi))
    rm = (o["rb"][ih, ia] + o["rt"][ih, ia]) / 2
    a_opt = np.deg2rad(al[ia])
    for f in (0.7, 1.0, 1.3):
        Rf = np.full_like(mu_ax, np.mean(o["Rf"]) * f)
        g = g_index(rm, a_opt, mu_ax, Rf, p["k"]) / g_index(ref_radius(p), 0.0, mu_ax, Rf, p["k"])
        ax3.plot(phi, g, label=f"R_f = {np.mean(o['Rf'])*f*1e3:.1f} mm")
    ax3.axhline(1, color="gray", ls=":")
    ax3.set(xlabel="摩擦角 φ = arctan μ [deg]", ylabel="M_rel", title="最適形状の効果(手の2軸 φ, R_f)")
    ax3.legend(fontsize=8)

    rb, rt, h = o["rb"][ih, ia] * 1e3, o["rt"][ih, ia] * 1e3, hs[ih]
    r0, h0 = ref_radius(p) * 1e3, p["V"] * 1e3 / (math.pi * (ref_radius(p) * 1e3) ** 2)
    ax4.fill([-rb, rb, rt, -rt], [0, 0, h, h], color="#7b4a2b", alpha=0.85, label="最適形状")
    ax4.plot([-r0, r0, r0, -r0, -r0], [0, 0, h0, h0, 0], "k--", lw=1, label="基準円柱")
    ax4.set_aspect("equal")
    ax4.legend(fontsize=8)
    ax4.set(title=f"上底 r={rt:.1f} / 下底 r={rb:.1f} / h={h:.1f} mm", xlabel="mm", ylabel="mm")
    fig.tight_layout()


def summary_text(o):
    p, ih, ia = o["p"], o["ih"], o["ia"]
    mean, p95 = o["mean"][ih, ia], o["p95"][ih, ia]
    s = [f"■ 最適形状({'平均' if o['objective']=='mean' else '95%点'}最小)",
         f"  α = {o['al'][ia]:.1f}°   h = {o['hs'][ih]:.1f} mm",
         f"  上底半径 {o['rt'][ih, ia]*1e3:.1f} mm / 下底半径 {o['rb'][ih, ia]*1e3:.1f} mm",
         f"  M_rel 平均 {mean:.3f} / 95%点 {p95:.3f}(基準円柱=1)",
         f"  → 基準円柱比で約 {100*(1-mean):.0f}% 減(平均)",
         f"  推定溶解量(基準実測 {p['Mref0']:.4f} g × M_rel)",
         f"     平均 {p['Mref0']*mean:.4f} g / 95%点 {p['Mref0']*p95:.4f} g",
         "",
         "■ 必要な手の計測(これだけ)",
         "  ① 摩擦係数 μ(傾斜板で滑り出し角→ μ=tanθ)",
         "  ② 指先の曲率半径 R_f(指先の側面写真)",
         "  ③ 基準円柱の溶解量 M_ref0 を1点実測(手の熱・弾性を一括較正)"]
    if ia in (0, len(o["al"]) - 1) or ih in (0, len(o["hs"]) - 1):
        s.append("\n※ 最適点が探索範囲の端です。制約(半径・高さ)を見直してください。")
    return "\n".join(s)


def export_csv(path, o):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["alpha_deg", "h_mm", "r_top_mm", "r_bottom_mm", "Mrel_mean", "Mrel_p95"])
        for i, h in enumerate(o["hs"]):
            for j, a in enumerate(o["al"]):
                w.writerow([f"{a:.2f}", f"{h:.2f}", f"{o['rt'][i,j]*1e3:.3f}",
                            f"{o['rb'][i,j]*1e3:.3f}", f"{o['mean'][i,j]:.5f}", f"{o['p95'][i,j]:.5f}"])


# ============================ GUI ==========================================
class App:
    def __init__(self, root):
        root.title("チョコレート形状最適化(手の縮約モデル) v0.2")
        root.geometry("1300x800")
        self.vars, self.o = {}, None
        left = ttk.Frame(root, padding=6)
        left.pack(side="left", fill="y")

        def box(title, rows, defaults):
            fr = ttk.LabelFrame(left, text=title, padding=6)
            fr.pack(fill="x", pady=4)
            for i, (lab, k) in enumerate(rows):
                ttk.Label(fr, text=lab).grid(row=i, column=0, sticky="w", pady=1)
                self.vars[k] = tk.StringVar(value=str(defaults[k]))
                ttk.Entry(fr, textvariable=self.vars[k], width=9).grid(row=i, column=1, padx=4)

        box("チョコ・モデル", [("体積 V [mL]", "V"), ("最小半径 [mm]", "rmin"), ("最大半径 [mm]", "rmax"),
                              ("接触指数 k(ヘルツ=0.667)", "k"),
                              ("基準円柱の高さ [mm](0=直径と同じ)", "ref_h"),
                              ("基準円柱の実測溶解量 M_ref0 [g]", "Mref0")], DEFAULT_P)
        box("手(縮約: φ と R_f だけ)", [("摩擦係数 μ 平均", "mu_m"), ("μ のばらつき CV[%]", "mu_cv"),
                                        ("指先曲率半径 R_f 平均 [mm]", "Rf_m"),
                                        ("R_f のばらつき CV[%]", "Rf_cv"), ("手のサンプル数", "ns")],
            DEFAULT_HAND)
        box("探索範囲", [("α 最小 [deg]", "a0"), ("α 最大 [deg]", "a1"), ("α 分割数", "na"),
                         ("h 最小 [mm]", "h0"), ("h 最大 [mm]", "h1"), ("h 分割数", "nh")], DEFAULT_GRID)
        self.mean_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="個人差を無視(平均手のみ)", variable=self.mean_only).pack(anchor="w")
        self.obj = tk.StringVar(value="p95")
        ttk.Radiobutton(left, text="95%点を最小化(ロバスト)", variable=self.obj, value="p95").pack(anchor="w")
        ttk.Radiobutton(left, text="平均を最小化", variable=self.obj, value="mean").pack(anchor="w")
        bf = ttk.Frame(left)
        bf.pack(fill="x", pady=6)
        ttk.Button(bf, text="計算・最適化", command=self.run).pack(side="left", padx=2)
        ttk.Button(bf, text="CSV保存", command=self.save_csv).pack(side="left", padx=2)
        ttk.Button(bf, text="図を保存", command=self.save_fig).pack(side="left", padx=2)
        self.text = tk.Text(left, width=48, height=17, font=("TkFixedFont", 9))
        self.text.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(9, 7), dpi=100)
        right = ttk.Frame(root)
        right.pack(side="right", fill="both", expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        NavigationToolbar2Tk(self.canvas, right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def read(self):
        f = lambda k: float(self.vars[k].get())
        p = {k: f(k) for k in DEFAULT_P}
        hand = {k: f(k) for k in DEFAULT_HAND}
        grid = {k: f(k) for k in DEFAULT_GRID}
        return p, hand, grid

    def run(self):
        try:
            p, hand, grid = self.read()
            self.o = run_all(p, hand, grid, self.mean_only.get(), self.obj.get())
        except Exception as e:
            messagebox.showerror("エラー", str(e))
            return
        draw(self.fig, self.o)
        self.canvas.draw()
        self.text.delete("1.0", "end")
        self.text.insert("end", summary_text(self.o))

    def save_csv(self):
        if self.o:
            path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if path:
                export_csv(path, self.o)

    def save_fig(self):
        if self.o:
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
            if path:
                self.fig.savefig(path, dpi=150)


def main():
    if tk is None:
        raise SystemExit("tkinter が利用できません")
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
