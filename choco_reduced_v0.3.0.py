#!/usr/bin/env python3
"""必要圧力の計算・側面角 alpha の評価ツール v0.3.0

モデルの範囲
------------
- 形状変数は側面角 alpha [deg] のみ。
- 計算する評価値は、指がチョコレートを保持するために必要な平均圧力 P [kPa]。
- 左右対称に2本の指でつまみ、各指の接触面積 A は一定と仮定する。
- ヘルツ接触、皮膚の弾性変形、指先曲率、熱伝導、溶解量は扱わない。

力のつり合い:
    2 N (sin(alpha) + mu cos(alpha)) = SF * m * g

必要圧力:
    P = N / A
      = SF * m * g / (2 A (sin(alpha) + mu cos(alpha)))

注意:
- A は「片側の指」の接触面積 [mm^2]。
- A は alpha によらず一定とする。この仮定では P は alpha と mu から決まり、
  P 自体は独立した探索変数ではなく、alpha に対する計算結果である。
- このモデルは必要圧力を比較するものであり、溶けやすさや溶解量は予測しない。
"""

from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path

import numpy as np
import matplotlib
from matplotlib.figure import Figure

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except ImportError:  # GUI が使えない環境でも計算関数は import できる
    tk = None

__version__ = "0.3.0"

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "Yu Gothic", "Meiryo", "Hiragino Sans", "Noto Sans CJK JP",
    "IPAexGothic", "TakaoPGothic", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

G = 9.80665  # 重力加速度 [m/s^2]
DENOMINATOR_MIN = 0.02  # 支持不能・数値的不安定な条件を除く閾値

DEFAULT_MODEL = {
    "mass_g": 15.0,
    "contact_area_mm2": 100.0,
    "safety_factor": 3.0,
}
DEFAULT_HAND = {
    "mu_mean": 0.5,
    "mu_cv": 25.0,
    "sample_count": 500,
}
DEFAULT_ALPHA = {
    "alpha_min": -10.0,
    "alpha_max": 35.0,
    "alpha_steps": 91,
}


def sample_friction(hand: dict, rng: np.random.Generator, mean_only: bool) -> np.ndarray:
    """摩擦係数を平均値または対数正規分布からサンプリングする。"""
    n = 1 if mean_only else int(hand["sample_count"])
    mean = float(hand["mu_mean"])
    cv = float(hand["mu_cv"])

    if mean < 0:
        raise ValueError("摩擦係数 μ の平均は 0 以上にしてください。")
    if cv < 0:
        raise ValueError("μ のばらつき CV は 0 以上にしてください。")
    if n < 1:
        raise ValueError("サンプル数は 1 以上にしてください。")

    if n == 1 or cv == 0 or mean == 0:
        return np.full(n, mean, dtype=float)

    # 平均値と変動係数(CV)を指定した正の対数正規分布
    cv_ratio = cv / 100.0
    sigma2 = math.log1p(cv_ratio**2)
    return rng.lognormal(
        mean=math.log(mean) - sigma2 / 2.0,
        sigma=math.sqrt(sigma2),
        size=n,
    )


def required_pressure_kpa(
    mass_g: float,
    contact_area_mm2: float,
    safety_factor: float,
    alpha_deg: np.ndarray | float,
    mu: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """必要な片側押し付け力 N [N] と圧力 P [kPa] を計算する。

    alpha_deg と mu は NumPy のブロードキャスト規則に従う。
    分母が DENOMINATOR_MIN 以下になる条件は NaN とする。
    """
    if mass_g <= 0:
        raise ValueError("チョコレートの質量は 0 より大きくしてください。")
    if contact_area_mm2 <= 0:
        raise ValueError("片側の接触面積は 0 より大きくしてください。")
    if safety_factor <= 0:
        raise ValueError("安全率 SF は 0 より大きくしてください。")

    alpha_rad = np.deg2rad(np.asarray(alpha_deg, dtype=float))
    friction = np.asarray(mu, dtype=float)
    denominator = np.sin(alpha_rad) + friction * np.cos(alpha_rad)

    mass_kg = mass_g / 1000.0
    area_m2 = contact_area_mm2 * 1e-6
    numerator = safety_factor * mass_kg * G

    valid = denominator > DENOMINATOR_MIN
    with np.errstate(divide="ignore", invalid="ignore"):
        normal_force_n = np.where(valid, numerator / (2.0 * denominator), np.nan)
        pressure_kpa = normal_force_n / area_m2 / 1000.0
    return normal_force_n, pressure_kpa


def run_analysis(
    model: dict,
    hand: dict,
    alpha_settings: dict,
    mean_only: bool,
    objective: str,
    seed: int = 0,
) -> dict:
    """alpha ごとの必要圧力分布を計算し、最小圧力となる alpha を選ぶ。"""
    alpha_min = float(alpha_settings["alpha_min"])
    alpha_max = float(alpha_settings["alpha_max"])
    alpha_steps = int(alpha_settings["alpha_steps"])

    if not (-89.0 < alpha_min < alpha_max < 89.0):
        raise ValueError("角度範囲は -89° より大きく、89° より小さい範囲で設定してください。")
    if alpha_steps < 2:
        raise ValueError("角度の分割数は 2 以上にしてください。")
    if objective not in {"mean", "p95"}:
        raise ValueError("評価方法は mean または p95 を指定してください。")

    rng = np.random.default_rng(seed)
    mu_samples = sample_friction(hand, rng, mean_only)
    alphas = np.linspace(alpha_min, alpha_max, alpha_steps)

    # shape: (摩擦係数サンプル数, alpha の数)
    normal_force_n, pressure_kpa = required_pressure_kpa(
        mass_g=float(model["mass_g"]),
        contact_area_mm2=float(model["contact_area_mm2"]),
        safety_factor=float(model["safety_factor"]),
        alpha_deg=alphas[None, :],
        mu=mu_samples[:, None],
    )

    valid_rate = np.isfinite(pressure_kpa).mean(axis=0)
    with warnings.catch_warnings():
        # 一部の alpha で全サンプルが無効でも、他の角度の計算は継続する。
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_kpa = np.nanmean(pressure_kpa, axis=0)
        median_kpa = np.nanpercentile(pressure_kpa, 50, axis=0)
        p05_kpa = np.nanpercentile(pressure_kpa, 5, axis=0)
        p95_kpa = np.nanpercentile(pressure_kpa, 95, axis=0)

    # 摩擦係数サンプルの90%以上で保持可能な角度のみ候補とする
    feasible = valid_rate >= 0.90
    mean_kpa = np.where(feasible, mean_kpa, np.nan)
    median_kpa = np.where(feasible, median_kpa, np.nan)
    p05_kpa = np.where(feasible, p05_kpa, np.nan)
    p95_kpa = np.where(feasible, p95_kpa, np.nan)

    score = mean_kpa if objective == "mean" else p95_kpa
    if not np.isfinite(score).any():
        raise ValueError(
            "設定範囲内に計算可能な角度がありません。摩擦係数または alpha の範囲を確認してください。"
        )

    best_index = int(np.nanargmin(score))
    best_alpha = float(alphas[best_index])
    best_pressures = pressure_kpa[:, best_index]
    best_valid = best_pressures[np.isfinite(best_pressures)]
    best_force, _ = required_pressure_kpa(
        float(model["mass_g"]),
        float(model["contact_area_mm2"]),
        float(model["safety_factor"]),
        best_alpha,
        mu_samples,
    )
    best_force_valid = best_force[np.isfinite(best_force)]

    return {
        "alphas": alphas,
        "mu_samples": mu_samples,
        "pressure_samples_kpa": pressure_kpa,
        "normal_force_samples_n": normal_force_n,
        "valid_rate": valid_rate,
        "mean_kpa": mean_kpa,
        "median_kpa": median_kpa,
        "p05_kpa": p05_kpa,
        "p95_kpa": p95_kpa,
        "score": score,
        "best_index": best_index,
        "best_alpha": best_alpha,
        "best_mean_kpa": float(np.mean(best_valid)),
        "best_median_kpa": float(np.median(best_valid)),
        "best_p95_kpa": float(np.percentile(best_valid, 95)),
        "best_force_mean_n": float(np.mean(best_force_valid)),
        "best_valid_rate": float(valid_rate[best_index]),
        "objective": objective,
        "model": dict(model),
        "hand": dict(hand),
        "alpha_settings": dict(alpha_settings),
    }


def draw_figure(fig: Figure, result: dict) -> None:
    """必要圧力 P と側面角 alpha の関係を描画する。"""
    fig.clear()
    ax = fig.subplots(1, 1)

    alphas = result["alphas"]
    ax.fill_between(
        alphas,
        result["p05_kpa"],
        result["p95_kpa"],
        alpha=0.22,
        label="5–95% (摩擦係数のばらつき)",
    )
    ax.plot(alphas, result["mean_kpa"], linewidth=2, label="平均必要圧力")
    ax.plot(alphas, result["p95_kpa"], linestyle="--", label="必要圧力の95%点")
    ax.axvline(
        result["best_alpha"],
        color="red",
        linestyle=":",
        linewidth=1.8,
        label=f"選択した alpha = {result['best_alpha']:.2f}°",
    )
    ax.scatter(
        [result["best_alpha"]],
        [result["score"][result["best_index"]]],
        color="red",
        zorder=5,
    )
    ax.set_title("側面角 α と必要圧力 P の関係")
    ax.set_xlabel("側面角 α [deg]（正の値：上側が広い）")
    ax.set_ylabel("片側に必要な圧力 P [kPa]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()


def make_summary(result: dict) -> str:
    model = result["model"]
    objective_label = "平均圧力" if result["objective"] == "mean" else "95%点の圧力"
    alpha = result["best_alpha"]
    lower_or_upper = result["best_index"] in (0, len(result["alphas"]) - 1)

    lines = [
        "■ 必要圧力が最小となる条件",
        f"  側面角 α = {alpha:.2f}°",
        f"  平均必要圧力 = {result['best_mean_kpa']:.3f} kPa",
        f"  必要圧力の中央値 = {result['best_median_kpa']:.3f} kPa",
        f"  必要圧力の95%点 = {result['best_p95_kpa']:.3f} kPa",
        f"  最適化基準 = {objective_label}",
        f"  計算可能なサンプル割合 = {result['best_valid_rate'] * 100:.1f}%",
        "",
        "■ 固定条件",
        f"  チョコレート質量 = {model['mass_g']:.3f} g",
        f"  片側の接触面積 = {model['contact_area_mm2']:.3f} mm²",
        f"  安全率 SF = {model['safety_factor']:.3f}",
        f"  摩擦係数 μ の平均 = {result['hand']['mu_mean']:.3f}",
        f"  μ の変動係数 CV = {result['hand']['mu_cv']:.1f}%",
        "",
        "■ モデルの仮定",
        "  ・左右2本の指で対称につまむ",
        "  ・接触面積は片側ごとに指定し、alpha によらず一定",
        "  ・皮膚の変形、ヘルツ接触、熱量、溶解量は計算しない",
        "  ・P は alpha から求める値で、独立変数ではない",
    ]
    if lower_or_upper:
        lines.extend([
            "",
            "※ 最小値が探索範囲の端にあります。",
            "  その端が物理的・製品設計上も許されるか確認してください。",
        ])
    return "\n".join(lines)


def export_csv(path: str | Path, result: dict) -> None:
    """alpha ごとの必要圧力を CSV に保存する。"""
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "alpha_deg",
            "pressure_mean_kPa",
            "pressure_median_kPa",
            "pressure_p05_kPa",
            "pressure_p95_kPa",
            "valid_sample_rate",
        ])
        for i, alpha in enumerate(result["alphas"]):
            values = [
                alpha,
                result["mean_kpa"][i],
                result["median_kpa"][i],
                result["p05_kpa"][i],
                result["p95_kpa"][i],
                result["valid_rate"][i],
            ]
            writer.writerow([
                f"{values[0]:.3f}",
                *("" if not np.isfinite(v) else f"{v:.6f}" for v in values[1:5]),
                f"{values[5]:.4f}",
            ])


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title(f"チョコレート保持に必要な圧力の評価 v{__version__}")
        root.geometry("1180x760")
        root.minsize(980, 640)
        self.vars: dict[str, tk.StringVar] = {}
        self.result: dict | None = None

        left = ttk.Frame(root, padding=8)
        left.pack(side="left", fill="y")
        right = ttk.Frame(root, padding=(4, 8, 8, 8))
        right.pack(side="right", fill="both", expand=True)

        def add_input_group(title: str, fields: list[tuple[str, str]], defaults: dict) -> None:
            frame = ttk.LabelFrame(left, text=title, padding=7)
            frame.pack(fill="x", pady=4)
            for row, (label, key) in enumerate(fields):
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
                var = tk.StringVar(value=str(defaults[key]))
                self.vars[key] = var
                ttk.Entry(frame, textvariable=var, width=11).grid(row=row, column=1, padx=(8, 2), pady=2)

        add_input_group(
            "チョコレート・接触条件",
            [
                ("質量 m [g]", "mass_g"),
                ("片側の接触面積 A [mm²]", "contact_area_mm2"),
                ("安全率 SF", "safety_factor"),
            ],
            DEFAULT_MODEL,
        )
        add_input_group(
            "摩擦条件",
            [
                ("摩擦係数 μ の平均", "mu_mean"),
                ("μ の変動係数 CV [%]", "mu_cv"),
                ("サンプル数", "sample_count"),
            ],
            DEFAULT_HAND,
        )
        add_input_group(
            "側面角 α の探索範囲",
            [
                ("α 最小 [deg]", "alpha_min"),
                ("α 最大 [deg]", "alpha_max"),
                ("分割数", "alpha_steps"),
            ],
            DEFAULT_ALPHA,
        )

        self.mean_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="摩擦係数の個人差を無視（平均値のみ）", variable=self.mean_only).pack(anchor="w", pady=(5, 2))

        self.objective = tk.StringVar(value="p95")
        objective_frame = ttk.LabelFrame(left, text="最適化基準", padding=6)
        objective_frame.pack(fill="x", pady=4)
        ttk.Radiobutton(objective_frame, text="95%点の圧力を最小化（余裕を重視）", variable=self.objective, value="p95").pack(anchor="w")
        ttk.Radiobutton(objective_frame, text="平均圧力を最小化", variable=self.objective, value="mean").pack(anchor="w")

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=7)
        ttk.Button(buttons, text="計算・評価", command=self.run).pack(side="left", padx=2)
        ttk.Button(buttons, text="CSV保存", command=self.save_csv).pack(side="left", padx=2)
        ttk.Button(buttons, text="図を保存", command=self.save_figure).pack(side="left", padx=2)

        self.output = tk.Text(left, width=49, height=21, wrap="word", font=("TkFixedFont", 9))
        self.output.pack(fill="both", expand=True, pady=(4, 0))
        self.output.insert(
            "end",
            "接触面積 A は片側の指について入力してください。\n"
            "この面積は alpha によらず一定と仮定します。\n"
            "計算されるのは保持に必要な圧力であり、溶解量ではありません。",
        )

        self.figure = Figure(figsize=(8.2, 6.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right)

    def read_inputs(self) -> tuple[dict, dict, dict]:
        def number(key: str) -> float:
            raw = self.vars[key].get().strip()
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError(f"'{key}' には数値を入力してください。") from exc

        model = {key: number(key) for key in DEFAULT_MODEL}
        hand = {key: number(key) for key in DEFAULT_HAND}
        alpha_settings = {key: number(key) for key in DEFAULT_ALPHA}
        hand["sample_count"] = int(hand["sample_count"])
        alpha_settings["alpha_steps"] = int(alpha_settings["alpha_steps"])
        return model, hand, alpha_settings

    def run(self) -> None:
        try:
            model, hand, alpha_settings = self.read_inputs()
            self.result = run_analysis(
                model,
                hand,
                alpha_settings,
                self.mean_only.get(),
                self.objective.get(),
            )
        except Exception as exc:
            messagebox.showerror("計算エラー", str(exc), parent=self.root)
            return

        draw_figure(self.figure, self.result)
        self.canvas.draw()
        self.output.delete("1.0", "end")
        self.output.insert("end", make_summary(self.result))

    def save_csv(self) -> None:
        if self.result is None:
            messagebox.showinfo("未計算", "先に計算・評価を実行してください。", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".csv",
            filetypes=[("CSV ファイル", "*.csv")],
            initialfile="required_pressure_by_alpha.csv",
        )
        if path:
            try:
                export_csv(path, self.result)
            except OSError as exc:
                messagebox.showerror("保存エラー", str(exc), parent=self.root)

    def save_figure(self) -> None:
        if self.result is None:
            messagebox.showinfo("未計算", "先に計算・評価を実行してください。", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".png",
            filetypes=[("PNG 画像", "*.png")],
            initialfile="required_pressure_by_alpha.png",
        )
        if path:
            try:
                self.figure.savefig(path, dpi=160)
            except OSError as exc:
                messagebox.showerror("保存エラー", str(exc), parent=self.root)


def main() -> None:
    if tk is None:
        raise SystemExit("tkinter が利用できません。この環境では GUI を起動できません。")
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
