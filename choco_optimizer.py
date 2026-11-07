#!/usr/bin/env python3
"""
choco_optimizer.py  -  台形回転体(円錐台)チョコレートの形状最適化 GUI  v0.1

モデルの流れ
  形状(α, h; 体積一定) -> 保持に必要な押し付け力 N -> 皮膚変形による接触面積 A
  -> 平均圧力 P = N/A -> 接触熱伝達 h_c(P) -> 熱流入 -> 溶解量指標 M [g]

手の個人差は θ = (E, t_s, R_f, μ, T_s, f_b) に集約し、平均とばらつきから
モンテカルロでサンプリングして「平均」または「95%点(ロバスト)」で最適化する。

角度 α: 側面の鉛直からの傾き。α > 0 で上が広い(逆台形)、α < 0 で下が広い。
        指でつまむとき、α > 0 の方が側面の法線成分が重さを支える(くさび効果)。

※ 係数(CONST)や接触式は仮定。実験A〜Cの結果で較正して使うこと。
"""
import csv
import math
import warnings

import numpy as np
from matplotlib.figure import Figure
import matplotlib

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
except ImportError:  # ヘッドレス環境(計算のみ使う場合)
    tk = None

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Yu Gothic", "Meiryo", "Hiragino Sans", "Noto Sans CJK JP",
    "IPAexGothic", "TakaoPGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

G = 9.80665

# ---- 物性などの定数(必要に応じて較正) -------------------------------------
CONST = dict(
    nu=0.45,          # 皮膚のポアソン比
    k_c=0.30,         # チョコの熱伝導率 [W/mK]
    c_c=1800.0,       # チョコの比熱 [J/kgK]
    L_fus=1.2e5,      # 融解潜熱 [J/kg]
    T_melt=32.0,      # 融点(代表値) [°C]
    e_s0=1180.0,      # 皮膚の熱浸透率 [W s^0.5/m2K]
    hc_min=300.0,     # 接触熱伝達 低圧側 [W/m2K]
    hc_max=3000.0,    # 接触熱伝達 高圧側 [W/m2K]
    P_half=5e3,       # h_c が中間値になる圧力 [Pa]
    side_frac=0.25,   # 1面あたりの接触が側面積に占める上限
    finger_frac=0.8,  # 指腹の最大接触面積 = frac * pi * R_f^2
)

# 手パラメータ: キー, 表示名, 平均, ばらつき, 種別('cv'=CV%, 'sd'=標準偏差), 単位
HAND_PARAMS = [
    ("E",  "皮膚の弾性率 E",      100.0, 30.0, "cv", "kPa"),
    ("ts", "指腹の軟組織厚 t_s",    4.0, 25.0, "cv", "mm"),
    ("Rf", "指先の曲率半径 R_f",    8.0, 15.0, "cv", "mm"),
    ("mu", "摩擦係数 μ",            0.5, 25.0, "cv", "-"),
    ("Ts", "皮膚表面温度 T_s",     33.0,  1.5, "sd", "°C"),
    ("fb", "熱供給係数(血流)f_b",   1.0, 15.0, "cv", "-"),
]


# ============================ 計算コア ======================================
def sample_theta(spec, ns, rng):
    """spec: {key: (mean, spread, kind)} -> {key: ndarray(ns)}(SI 換算前の GUI 単位)"""
    out = {}
    for k, (m, s, kind) in spec.items():
        if ns == 1 or s <= 0:
            out[k] = np.full(ns, float(m))
        elif kind == "sd":
            out[k] = rng.normal(m, s, ns)
        else:
            cv = s / 100.0
            sig2 = math.log(1 + cv * cv)
            out[k] = rng.lognormal(math.log(m) - sig2 / 2, math.sqrt(sig2), ns)
    return out


def frustum_geometry(V, h, alpha):
    """体積 V 一定、高さ h、傾き α から下底 rb・上底 rt を求める。"""
    d = h * np.tan(alpha)
    K = 3 * V / (np.pi * h)
    disc = 12 * K - 3 * d ** 2
    with np.errstate(invalid="ignore"):
        rb = (-3 * d + np.sqrt(disc)) / 6
    rt = rb + d
    return rb, rt, d, disc


def simulate(p, th, alphas_deg, hs_mm):
    """格子 (h × α) と θ サンプル (ns) について溶解量指標などを計算する。"""
    C = CONST
    al = np.deg2rad(np.asarray(alphas_deg, float))[None, :]
    h = (np.asarray(hs_mm, float) * 1e-3)[:, None]
    V = p["V"] * 1e-6
    m = p["rho"] * 1e3 * V
    rmin = p["rmin"] * 1e-3

    rb, rt, d, disc = frustum_geometry(V, h, al)
    valid = (disc > 0) & (rb >= rmin) & (rt >= rmin)
    rb = np.where(valid, rb, np.nan)
    rt = np.where(valid, rt, np.nan)
    rm = (rb + rt) / 2
    s = np.sqrt(h ** 2 + d ** 2)
    A_lat = np.pi * (rb + rt) * s

    def T(k, scale=1.0):  # (ns,1,1) へ
        return (th[k] * scale)[:, None, None]
    E, ts, Rf = T("E", 1e3), T("ts", 1e-3), T("Rf", 1e-3)
    mu, Ts, fb = T("mu"), T("Ts"), T("fb")
    al3, rm3, A_lat3 = al[None], rm[None], A_lat[None]

    # --- 保持力: 両側からつまみ、側面の傾きで重さの一部を支える ---
    den = np.sin(al3) + mu * np.cos(al3)
    with np.errstate(divide="ignore", invalid="ignore"):
        N = np.where(den > 0.02, p["SF"] * m * G / (2 * den), np.nan)

    # --- 皮膚変形による接触面積(ヘルツ + 薄層による硬化 + 飽和) ---
    Reff = 1.0 / (1.0 / Rf + 1.0 / rm3)
    Es = E / (1 - C["nu"] ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (3 * N * Reff / (4 * Es)) ** (1 / 3)
        for _ in range(30):
            Ee = Es * (1 + (a / ts) ** 1.5)
            a = 0.5 * a + 0.5 * (3 * N * Reff / (4 * Ee)) ** (1 / 3)
        A_h = np.pi * a ** 2
        A_cap = np.minimum(C["side_frac"] * A_lat3, C["finger_frac"] * np.pi * Rf ** 2)
        A = (A_h ** -4 + A_cap ** -4) ** (-0.25)
        P = N / A
    feas = np.isfinite(P) & (P <= p["Pmax"] * 1e3)

    # --- 熱流入: 接触熱抵抗 + 半無限体2つの非定常伝導 ---
    e_c = math.sqrt(C["k_c"] * p["rho"] * 1e3 * C["c_c"])
    e_s = C["e_s0"] * fb
    hc = C["hc_min"] + (C["hc_max"] - C["hc_min"]) * P / (P + C["P_half"])
    R = 1.0 / hc
    b = math.sqrt(math.pi) * (1.0 / e_c + 1.0 / e_s)
    U = math.sqrt(p["t"])
    I = (2 / b) * (U - (R / b) * np.log((R + b * U) / R))      # ∫dt/(R+b√t)
    dT = np.maximum(Ts - p["T0"], 0.0)
    E_melt = C["L_fus"] + C["c_c"] * max(C["T_melt"] - p["T0"], 0.0)
    M = p["kappa"] * (2 * A) * dT * I / E_melt * 1e3             # [g]
    M = np.where(feas, M, np.nan)
    return dict(M=M, N=np.where(feas, N, np.nan), A=np.where(feas, 2 * A, np.nan),
                P=np.where(feas, P, np.nan), feas=feas, rb=rb, rt=rt,
                alphas=np.asarray(alphas_deg, float), hs=np.asarray(hs_mm, float))


def aggregate(res, min_rate):
    M = res["M"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate = np.isfinite(M).mean(axis=0)
        mean = np.nanmean(M, axis=0)
        p95 = np.nanpercentile(M, 95, axis=0) if M.shape[0] > 1 else mean.copy()
        Nm, Am, Pm = (np.nanmean(res[k], axis=0) for k in ("N", "A", "P"))
    ok = rate >= min_rate
    return dict(rate=rate, mean=np.where(ok, mean, np.nan),
                p95=np.where(ok, p95, np.nan), N=Nm, A=Am, P=Pm)


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or np.std(x[ok]) == 0:
        return 0.0
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def run_all(p, spec, grid, use_mean_only, objective, ns, seed=0):
    rng = np.random.default_rng(seed)
    n = 1 if use_mean_only else ns
    th = sample_theta(spec, n, rng)
    alphas = np.linspace(grid["a0"], grid["a1"], grid["na"])
    hs = np.linspace(grid["h0"], grid["h1"], grid["nh"])
    res = simulate(p, th, alphas, hs)
    agg = aggregate(res, p["min_rate"])
    score = agg["mean"] if objective == "mean" else agg["p95"]
    if not np.isfinite(score).any():
        raise ValueError("実行可能な形状がありません。許容圧力・最小半径・探索範囲を見直してください。")
    ih, ia = np.unravel_index(np.nanargmin(score), score.shape)
    sens = {k: spearman(th[k], res["M"][:, ih, ia]) for k in th} if n >= 20 else {}
    return dict(res=res, agg=agg, th=th, score=score, ih=ih, ia=ia, sens=sens,
                objective=objective, n=n)


# ============================ 作図 =========================================
def draw(fig, out, p):
    fig.clear()
    res, agg, score = out["res"], out["agg"], out["score"]
    ih, ia = out["ih"], out["ia"]
    al, hs = res["alphas"], res["hs"]
    lab = "平均" if out["objective"] == "mean" else "95%点"
    ax1, ax2, ax3, ax4 = fig.subplots(2, 2).ravel()

    pc = ax1.pcolormesh(al, hs, score, shading="auto", cmap="viridis")
    fig.colorbar(pc, ax=ax1, label=f"溶解量指標 {lab} [g]")
    ax1.plot(al[ia], hs[ih], "r*", ms=14)
    ax1.set(xlabel="傾き α [deg](+ で上が広い)", ylabel="高さ h [mm]",
            title="形状マップ(★=最適)")

    M = res["M"][:, ih, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lo, md, hi = (np.nanpercentile(M, q, axis=0) for q in (5, 50, 95))
    ax2.fill_between(al, lo, hi, alpha=0.25, label="5–95%(手の個人差)")
    ax2.plot(al, md, label="中央値")
    ax2.axvline(al[ia], color="r", ls="--")
    ax2.set(xlabel="傾き α [deg]", ylabel="溶解量指標 [g]",
            title=f"h = {hs[ih]:.1f} mm でのスライス")
    ax2.legend(fontsize=8)

    names = {k: n for k, n, *_ in HAND_PARAMS}
    if out["sens"]:
        ks = list(out["sens"])
        vals = [out["sens"][k] for k in ks]
        ax3.barh([names[k].split(" ")[0] for k in ks], vals,
                 color=["tab:red" if v > 0 else "tab:blue" for v in vals])
        ax3.axvline(0, color="k", lw=0.5)
        ax3.set(xlim=(-1, 1), xlabel="順位相関(溶解量との関係)", title="最適形状での手パラメータ感度")
    else:
        ax3.text(0.5, 0.5, "「平均手のみ」では感度は出せません", ha="center", va="center")
        ax3.set_axis_off()

    rb, rt, h = res["rb"][ih, ia] * 1e3, res["rt"][ih, ia] * 1e3, hs[ih]
    ax4.fill([-rb, rb, rt, -rt], [0, 0, h, h], color="#7b4a2b", alpha=0.85)
    for sgn in (-1, 1):
        xm = sgn * (rb + rt) / 4 * 1.0
        ax4.annotate("", xy=(xm, h / 2), xytext=(sgn * (rb + rt) / 2 + sgn * 12, h / 2),
                     arrowprops=dict(arrowstyle="->", color="tab:red"))
    ax4.set_aspect("equal")
    ax4.set(title=f"最適断面: 上底 r={rt:.1f}, 下底 r={rb:.1f}, h={h:.1f} mm",
            xlabel="mm", ylabel="mm")
    fig.tight_layout()


def summary_text(out, p):
    res, agg = out["res"], out["agg"]
    ih, ia = out["ih"], out["ia"]
    s = [f"■ 最適形状({'平均' if out['objective']=='mean' else '95%点'}最小)",
         f"  α = {res['alphas'][ia]:.1f}°  h = {res['hs'][ih]:.1f} mm",
         f"  上底半径 {res['rt'][ih, ia]*1e3:.1f} mm / 下底半径 {res['rb'][ih, ia]*1e3:.1f} mm",
         f"  保持力 N(片側)= {agg['N'][ih, ia]:.2f} N",
         f"  総接触面積 = {agg['A'][ih, ia]*1e6:.0f} mm²",
         f"  平均圧力 P = {agg['P'][ih, ia]/1e3:.1f} kPa",
         f"  溶解量指標: 平均 {agg['mean'][ih, ia]:.3f} g / 95%点 {agg['p95'][ih, ia]:.3f} g",
         f"  実行可能率 {agg['rate'][ih, ia]*100:.0f}%(θ サンプル {out['n']})"]
    at_edge = ia in (0, len(res["alphas"]) - 1) or ih in (0, len(res["hs"]) - 1)
    if at_edge:
        s.append("※ 最適点が探索範囲の端です。範囲や制約(最小半径等)を見直してください。")
    return "\n".join(s)


def export_csv(path, out):
    res, agg = out["res"], out["agg"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["alpha_deg", "h_mm", "r_top_mm", "r_bottom_mm", "N_N", "A_total_mm2",
                    "P_kPa", "M_mean_g", "M_p95_g", "feasible_rate"])
        for i, h in enumerate(res["hs"]):
            for j, a in enumerate(res["alphas"]):
                w.writerow([f"{a:.2f}", f"{h:.2f}", f"{res['rt'][i,j]*1e3:.3f}",
                            f"{res['rb'][i,j]*1e3:.3f}", f"{agg['N'][i,j]:.4f}",
                            f"{agg['A'][i,j]*1e6:.2f}", f"{agg['P'][i,j]/1e3:.3f}",
                            f"{agg['mean'][i,j]:.5f}", f"{agg['p95'][i,j]:.5f}",
                            f"{agg['rate'][i,j]:.3f}"])


DEFAULT_P = dict(V=15.0, rho=1.2, t=60.0, T0=22.0, SF=3.0, Pmax=60.0,
                 rmin=5.0, kappa=1.0, min_rate=0.9)
DEFAULT_GRID = dict(a0=-10.0, a1=35.0, na=46, h0=8.0, h1=40.0, nh=33)


# ============================ GUI ==========================================
class App:
    def __init__(self, root):
        self.root = root
        root.title("チョコレート形状最適化 v0.1")
        root.geometry("1320x820")
        self.vars = {}
        self.out = None

        left = ttk.Frame(root, padding=6)
        left.pack(side="left", fill="y")
        nb = ttk.Notebook(left)
        nb.pack(fill="x")
        t1, t2, t3 = ttk.Frame(nb, padding=6), ttk.Frame(nb, padding=6), ttk.Frame(nb, padding=6)
        nb.add(t1, text="チョコ・把持")
        nb.add(t2, text="手(θ)")
        nb.add(t3, text="探索・実行")

        rows = [("体積 V [mL]", "V"), ("密度 ρ [g/mL]", "rho"), ("保持時間 t [s]", "t"),
                ("初期温度 T0 [°C]", "T0"), ("把持の安全率 SF", "SF"),
                ("許容圧力 P_max [kPa]", "Pmax"), ("最小半径 [mm]", "rmin"),
                ("較正係数 κ(実験Cで合わせる)", "kappa"),
                ("実行可能率の下限(0–1)", "min_rate")]
        for i, (lab, k) in enumerate(rows):
            ttk.Label(t1, text=lab).grid(row=i, column=0, sticky="w", pady=2)
            self.vars[k] = tk.StringVar(value=str(DEFAULT_P[k]))
            ttk.Entry(t1, textvariable=self.vars[k], width=9).grid(row=i, column=1, padx=4)

        ttk.Label(t2, text="パラメータ").grid(row=0, column=0)
        ttk.Label(t2, text="平均").grid(row=0, column=1)
        ttk.Label(t2, text="ばらつき(CV% / SD)").grid(row=0, column=2)
        for i, (k, name, m, s, kind, unit) in enumerate(HAND_PARAMS, start=1):
            ttk.Label(t2, text=f"{name} [{unit}]").grid(row=i, column=0, sticky="w", pady=2)
            self.vars[k + "_m"] = tk.StringVar(value=str(m))
            self.vars[k + "_s"] = tk.StringVar(value=str(s))
            ttk.Entry(t2, textvariable=self.vars[k + "_m"], width=8).grid(row=i, column=1, padx=3)
            ttk.Entry(t2, textvariable=self.vars[k + "_s"], width=8).grid(row=i, column=2, padx=3)
        ttk.Label(t2, text="※ T_s のばらつきは SD[°C]、他は CV[%]",
                  foreground="gray").grid(row=8, column=0, columnspan=3, sticky="w", pady=6)

        grid_rows = [("α 最小 [deg]", "a0"), ("α 最大 [deg]", "a1"), ("α 分割数", "na"),
                     ("h 最小 [mm]", "h0"), ("h 最大 [mm]", "h1"), ("h 分割数", "nh"),
                     ("θ サンプル数", "ns")]
        for i, (lab, k) in enumerate(grid_rows):
            ttk.Label(t3, text=lab).grid(row=i, column=0, sticky="w", pady=2)
            default = 500 if k == "ns" else DEFAULT_GRID[k]
            self.vars[k] = tk.StringVar(value=str(default))
            ttk.Entry(t3, textvariable=self.vars[k], width=9).grid(row=i, column=1, padx=4)
        self.mean_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(t3, text="個人差を無視(平均手のみ)", variable=self.mean_only
                        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=4)
        self.objective = tk.StringVar(value="p95")
        ttk.Radiobutton(t3, text="95%点を最小化(ロバスト)", variable=self.objective,
                        value="p95").grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(t3, text="平均を最小化", variable=self.objective,
                        value="mean").grid(row=9, column=0, columnspan=2, sticky="w")

        bf = ttk.Frame(left)
        bf.pack(fill="x", pady=8)
        ttk.Button(bf, text="計算・最適化", command=self.run).pack(side="left", padx=2)
        ttk.Button(bf, text="CSV保存", command=self.save_csv).pack(side="left", padx=2)
        ttk.Button(bf, text="図を保存", command=self.save_fig).pack(side="left", padx=2)
        self.text = tk.Text(left, width=46, height=14, font=("TkFixedFont", 9))
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
        spec = {k: (f(k + "_m"), f(k + "_s"), kind) for k, _, _, _, kind, _ in HAND_PARAMS}
        grid = dict(a0=f("a0"), a1=f("a1"), na=int(f("na")),
                    h0=f("h0"), h1=f("h1"), nh=int(f("nh")))
        return p, spec, grid, int(f("ns"))

    def run(self):
        try:
            p, spec, grid, ns = self.read()
            self.out = run_all(p, spec, grid, self.mean_only.get(), self.objective.get(), ns)
        except Exception as e:  # 入力ミス・実行不能
            messagebox.showerror("エラー", str(e))
            return
        self.p = p
        draw(self.fig, self.out, p)
        self.canvas.draw()
        self.text.delete("1.0", "end")
        self.text.insert("end", summary_text(self.out, p))

    def save_csv(self):
        if not self.out:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            export_csv(path, self.out)

    def save_fig(self):
        if not self.out:
            return
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
