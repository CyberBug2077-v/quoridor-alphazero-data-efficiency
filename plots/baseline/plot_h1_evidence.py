"""Plot the frozen Baseline plateau and H1 evidence package.

The script deliberately reads only the merged ``h1_v1_1`` artifacts.  It does
not reopen training, replay, hold-out, or fixed-basket source logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE_DIR = (
    ROOT
    / "baseline"
    / "outputs"
    / "baseline_seed1001_4090_analysis"
    / "h1_v1_1"
)
DEFAULT_OUTPUT_DIR = ROOT / "plots" / "figures" / "baseline"

DEEP_BLUE = "#24557A"
LIGHT_BLUE = "#8FB8D5"
PALE_BLUE = "#D8E7F1"
ORANGE = "#C96F24"
PALE_ORANGE = "#E4B083"
CHARCOAL = "#30343B"
MID_GREY = "#667085"
LIGHT_GREY = "#A7ADB5"
GRID_GREY = "#D9DDE3"
POST_ONSET_GREY = "#F4F5F7"
CONFIRMATION_GREY = "#E2E5E9"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite number, got {value!r}")
    return result


def _required_float(value: object, name: str) -> float:
    result = _optional_float(value)
    if result is None:
        raise ValueError(f"Missing required numeric field: {name}")
    return result


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    if not number.is_integer():
        raise ValueError(f"Expected an integer, got {value!r}")
    return int(number)


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _load_aligned(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in _read_csv(path):
        checkpoint = _optional_int(raw.get("checkpoint"))
        if checkpoint is None or checkpoint in seen:
            raise ValueError(f"Invalid or duplicate checkpoint in {path}: {checkpoint}")
        seen.add(checkpoint)
        gpu_hours = _optional_float(raw.get("training_gpu_hours"))
        if checkpoint == 0 and gpu_hours is None:
            gpu_hours = 0.0
        row: dict[str, Any] = dict(raw)
        row.update(
            checkpoint=checkpoint,
            gpu_hours=_required_float(gpu_hours, "training_gpu_hours"),
            score=_required_float(
                raw.get("fixed_basket_macro_score"),
                "fixed_basket_macro_score",
            ),
            ci_low=_required_float(
                raw.get("fixed_basket_score_ci95_low"),
                "fixed_basket_score_ci95_low",
            ),
            ci_high=_required_float(
                raw.get("fixed_basket_score_ci95_high"),
                "fixed_basket_score_ci95_high",
            ),
            is_pretrained=_truth(raw.get("is_pretrained")),
            is_best_observed=_truth(raw.get("is_best_observed")),
            is_final=_truth(raw.get("is_final")),
            is_max_drawdown_peak=_truth(raw.get("is_max_drawdown_peak")),
            is_max_drawdown_trough=_truth(raw.get("is_max_drawdown_trough")),
        )
        if not row["ci_low"] <= row["score"] <= row["ci_high"]:
            raise ValueError(f"Score is outside its CI at checkpoint {checkpoint}")
        rows.append(row)
    return sorted(rows, key=lambda item: item["checkpoint"])


def _load_diagnostics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in _read_csv(path):
        row: dict[str, Any] = dict(raw)
        row.update(
            iteration=_optional_int(raw.get("iteration")),
            checkpoint=_optional_int(raw.get("checkpoint")),
            gpu_hours=_optional_float(raw.get("gpu_hours")),
            value=_optional_float(raw.get("value")),
            observable=_truth(raw.get("observable")),
            used_in_h1_trend=_truth(raw.get("used_in_h1_trend")),
            is_at_or_before_plateau=_truth(
                raw.get("is_at_or_before_plateau")
            ),
            is_after_plateau=_truth(raw.get("is_after_plateau")),
        )
        rows.append(row)
    return rows


def _load_effects(path: Path) -> dict[str, dict[str, Any]]:
    effects: dict[str, dict[str, Any]] = {}
    numeric_fields = (
        "start_iteration",
        "end_iteration",
        "start_checkpoint",
        "end_checkpoint",
        "valid_points",
        "ols_slope_per_iteration",
        "slope_per_20_iterations",
        "slope_per_gpu_hour",
    )
    for raw in _read_csv(path):
        metric = raw.get("metric", "")
        if not metric:
            raise ValueError(f"Effect row without a metric in {path}")
        row: dict[str, Any] = dict(raw)
        for field in numeric_fields:
            row[field] = _optional_float(raw.get(field))
        effects[metric] = row
    return effects


def _checkpoint_gpu_lookup(rows: Sequence[dict[str, Any]]) -> dict[int, float]:
    return {int(row["checkpoint"]): float(row["gpu_hours"]) for row in rows}


def _checkpoint_x(
    checkpoint: int | float | None,
    lookup: dict[int, float],
    field: str,
) -> float:
    if checkpoint is None or int(checkpoint) not in lookup:
        raise ValueError(f"{field} checkpoint {checkpoint!r} is absent from aligned data")
    return lookup[int(checkpoint)]


def _style_axis(ax: Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID_GREY, linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MID_GREY)
    ax.spines["bottom"].set_color(MID_GREY)
    ax.tick_params(colors=CHARCOAL, labelsize=9)


def _annotate_point(
    ax: Axes,
    row: dict[str, Any],
    text: str,
    *,
    xytext: tuple[float, float],
    ha: str = "left",
) -> None:
    ax.annotate(
        text,
        xy=(row["gpu_hours"], row["score"]),
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va="bottom",
        fontsize=9,
        color=CHARCOAL,
        arrowprops={"arrowstyle": "-", "color": MID_GREY, "linewidth": 0.8},
    )


def add_uncertainty_bars(ax: Axes, rows: Sequence[dict[str, Any]]) -> None:
    """Add discrete 95% bootstrap intervals for every evaluated checkpoint."""
    xs = np.asarray([row["gpu_hours"] for row in rows], dtype=float)
    scores = np.asarray([row["score"] for row in rows], dtype=float)
    low = np.asarray([row["ci_low"] for row in rows], dtype=float)
    high = np.asarray([row["ci_high"] for row in rows], dtype=float)
    ax.errorbar(
        xs,
        scores,
        yerr=np.vstack((scores - low, high - scores)),
        fmt="o-",
        markersize=5.2,
        markerfacecolor=DEEP_BLUE,
        markeredgecolor="white",
        markeredgewidth=0.7,
        color=DEEP_BLUE,
        ecolor=DEEP_BLUE,
        elinewidth=1.1,
        capsize=3.5,
        capthick=1.1,
        linewidth=1.35,
        zorder=4,
    )


def mark_pretrained(ax: Axes, row: dict[str, Any]) -> None:
    """Mark the checkpoint-0 pretrained starting point."""
    ax.scatter(
        row["gpu_hours"],
        row["score"],
        marker="D",
        s=86,
        facecolor="white",
        edgecolor=DEEP_BLUE,
        linewidth=1.7,
        zorder=7,
    )
    _annotate_point(ax, row, "Pretrained, ckpt 0", xytext=(8, -29))


def mark_best_observed(ax: Axes, row: dict[str, Any]) -> None:
    """Mark the largest observed fixed-basket macro score."""
    ax.scatter(
        row["gpu_hours"],
        row["score"],
        marker="*",
        s=180,
        facecolor=DEEP_BLUE,
        edgecolor="white",
        linewidth=0.8,
        zorder=8,
    )


def mark_final(ax: Axes, row: dict[str, Any]) -> None:
    """Mark the final evaluated checkpoint."""
    ax.scatter(
        row["gpu_hours"],
        row["score"],
        marker="s",
        s=76,
        facecolor=DEEP_BLUE,
        edgecolor="white",
        linewidth=0.9,
        zorder=7,
    )
    _annotate_point(
        ax,
        row,
        f"Final, ckpt {row['checkpoint']}",
        xytext=(-7, 22),
        ha="right",
    )


def _shade_time_context(
    ax: Axes,
    rows: Sequence[dict[str, Any]],
    plateau: dict[str, Any],
    *,
    label_onset: bool = False,
    label_confirmation: bool = False,
) -> tuple[float, float, float]:
    lookup = _checkpoint_gpu_lookup(rows)
    onset_checkpoint = int(plateau["plateau_start_checkpoint"])
    confirmation_start = int(plateau["confirmation_window_start"])
    confirmation_end = int(plateau["confirmation_window_end"])
    onset_x = _checkpoint_x(onset_checkpoint, lookup, "plateau onset")
    confirmation_start_x = _checkpoint_x(
        confirmation_start, lookup, "confirmation start"
    )
    confirmation_end_x = _checkpoint_x(
        confirmation_end, lookup, "confirmation end"
    )
    final_x = max(lookup.values())

    ax.axvspan(onset_x, final_x, color=POST_ONSET_GREY, zorder=-5)
    ax.axvspan(
        confirmation_start_x,
        confirmation_end_x,
        color=CONFIRMATION_GREY,
        zorder=-4,
    )
    ax.axvline(
        onset_x,
        color=CHARCOAL,
        linestyle=(0, (5, 4)),
        linewidth=1.25,
        zorder=3,
    )
    if label_onset:
        onset_label_x = onset_x + 0.006 * (final_x - min(lookup.values()))
        ax.text(
            onset_label_x,
            0.985,
            f"Detected plateau onset: ckpt {onset_checkpoint}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            rotation_mode="anchor",
            ha="right",
            va="top",
            fontsize=7.8,
            color=CHARCOAL,
        )
    if label_confirmation:
        ax.text(
            (confirmation_start_x + confirmation_end_x) / 2,
            0.035,
            f"Confirmation window\nckpt {confirmation_start}–{confirmation_end}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=MID_GREY,
        )
    return onset_x, confirmation_start_x, confirmation_end_x


def shade_plateau(
    ax: Axes,
    rows: Sequence[dict[str, Any]],
    plateau: dict[str, Any],
) -> None:
    """Shade the post-onset region and the detector's confirmation window."""
    _shade_time_context(
        ax,
        rows,
        plateau,
        label_onset=True,
        label_confirmation=True,
    )


def annotate_max_drawdown(
    ax: Axes, rows: Sequence[dict[str, Any]]
) -> tuple[float, int, int]:
    """Draw the frozen peak-to-trough arrow and return its summary."""
    peaks = [row for row in rows if row["is_max_drawdown_peak"]]
    troughs = [row for row in rows if row["is_max_drawdown_trough"]]
    if len(peaks) != 1 or len(troughs) != 1:
        raise ValueError("Expected one maximum-drawdown peak and one trough")
    peak, trough = peaks[0], troughs[0]
    change = float(trough["score"]) - float(peak["score"])
    bracket_x = float(trough["gpu_hours"]) + 0.45
    ax.annotate(
        "",
        xy=(bracket_x, trough["score"]),
        xytext=(bracket_x, peak["score"]),
        arrowprops={
            "arrowstyle": "|-|",
            "color": MID_GREY,
            "linewidth": 0.85,
            "linestyle": (0, (3, 2.5)),
        },
        zorder=5,
    )
    return change, int(peak["checkpoint"]), int(trough["checkpoint"])


def _find_single_marker(
    rows: Sequence[dict[str, Any]], field: str
) -> dict[str, Any]:
    matches = [row for row in rows if row[field]]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {field} row, found {len(matches)}")
    return matches[0]


def plot_baseline_strength(
    aligned_path: Path = DEFAULT_PACKAGE_DIR / "aligned_checkpoint_metrics.csv",
    plateau_path: Path = DEFAULT_PACKAGE_DIR / "plateau.json",
    output_path: Path = DEFAULT_OUTPUT_DIR / "figure_5_1_baseline_strength.png",
    *,
    dpi: int = 300,
) -> Path:
    """Render Figure 5.1 from the frozen aligned checkpoint package."""
    rows = _load_aligned(Path(aligned_path))
    plateau = _read_json(Path(plateau_path))
    if len(rows) != 12:
        raise ValueError(f"Figure 5.1 requires 12 checkpoints, found {len(rows)}")
    if not plateau.get("plateau_detected"):
        raise ValueError("Figure 5.1 requires a reproduced plateau")

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": CHARCOAL,
            "text.color": CHARCOAL,
        }
    ):
        fig, ax = plt.subplots(figsize=(12.2, 6.5))
        shade_plateau(ax, rows, plateau)
        add_uncertainty_bars(ax, rows)
        mark_pretrained(ax, _find_single_marker(rows, "is_pretrained"))
        best_observed = _find_single_marker(rows, "is_best_observed")
        mark_best_observed(ax, best_observed)
        mark_final(ax, _find_single_marker(rows, "is_final"))
        drawdown, drawdown_peak, drawdown_trough = annotate_max_drawdown(ax, rows)

        fig.suptitle(
            "Baseline playing strength and detected plateau",
            x=0.10,
            y=0.965,
            ha="left",
            fontsize=15,
            fontweight="bold",
        )
        fig.text(
            0.10,
            0.885,
            "Fixed-basket macro score with discrete checkpoint uncertainty",
            fontsize=10,
            color=MID_GREY,
            ha="left",
            va="top",
        )
        fig.text(
            0.975,
            0.885,
            (
                f"★  Best observed: ckpt {best_observed['checkpoint']}  ·  "
                f"{best_observed['gpu_hours']:.2f} GPU-h  ·  "
                f"score {best_observed['score']:.3f}"
            ),
            fontsize=9.2,
            color=DEEP_BLUE,
            fontweight="bold",
            ha="right",
            va="top",
        )
        fig.text(
            0.975,
            0.825,
            (
                "Observed maximum drawdown = "
                f"\N{MINUS SIGN}{abs(drawdown):.3f}  ·  "
                f"ckpt {drawdown_peak} → {drawdown_trough}"
            ),
            fontsize=8.8,
            color=CHARCOAL,
            ha="right",
            va="top",
        )
        ax.set_xlabel("Cumulative GPU-hours", fontsize=10.5, labelpad=8)
        ax.set_ylabel("Fixed-basket macro score", fontsize=10.5)
        ax.set_xlim(-0.45, max(row["gpu_hours"] for row in rows) + 0.65)
        y_low = min(row["ci_low"] for row in rows) - 0.025
        y_high = max(row["ci_high"] for row in rows) + 0.035
        ax.set_ylim(y_low, y_high)
        _style_axis(ax)

        top = ax.secondary_xaxis("top")
        top.set_xticks([row["gpu_hours"] for row in rows])
        top.set_xticklabels([str(row["checkpoint"]) for row in rows], fontsize=8)
        top.set_xlabel("Checkpoint", color=MID_GREY, fontsize=9, labelpad=6)
        top.tick_params(axis="x", colors=MID_GREY, length=3, pad=2)
        top.spines["top"].set_color(GRID_GREY)

        fig.text(
            0.10,
            0.045,
            (
                "Fixed-basket macro score assigns win = 1, draw = 0.5, "
                "and loss = 0, then equally weights the four registered opponents. "
                "Error bars are 95% stratified-bootstrap confidence intervals; "
                "evaluation occurs only at the plotted checkpoints."
            ),
            ha="left",
            va="bottom",
            fontsize=8.8,
            color=MID_GREY,
            wrap=True,
        )
        fig.subplots_adjust(left=0.10, right=0.975, top=0.76, bottom=0.20)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor="white")
        plt.close(fig)
    return output_path


def _metric_points(
    diagnostics: Sequence[dict[str, Any]],
    metric: str,
    checkpoint_gpu: dict[int, float],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in diagnostics:
        if row.get("metric") != metric or not row["observable"]:
            continue
        if row["value"] is None:
            continue
        point = dict(row)
        if row["checkpoint"] is not None:
            point["x"] = _checkpoint_x(
                row["checkpoint"], checkpoint_gpu, f"{metric}"
            )
        elif row["gpu_hours"] is not None:
            point["x"] = row["gpu_hours"]
        else:
            raise ValueError(f"Observable {metric} row has no time coordinate")
        points.append(point)
    return sorted(points, key=lambda item: item["x"])


def _plot_pre_post_series(
    ax: Axes,
    points: Sequence[dict[str, Any]],
    *,
    label: str,
    post_label: str | None = None,
    pre_color: str = DEEP_BLUE,
    post_color: str = LIGHT_BLUE,
    marker: str = "o",
    linestyle: str = "-",
    linewidth: float = 1.15,
    connect_pre: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pre = [point for point in points if point["is_at_or_before_plateau"]]
    post = [point for point in points if point["is_after_plateau"]]
    if pre:
        ax.plot(
            [point["x"] for point in pre],
            [point["value"] for point in pre],
            color=pre_color,
            linestyle=linestyle if connect_pre else "None",
            linewidth=linewidth,
            marker=marker,
            markersize=3.4,
            markerfacecolor=pre_color,
            markeredgecolor="white",
            markeredgewidth=0.35,
            label=label,
            zorder=4,
        )
    if post:
        connector = ([pre[-1]] if pre else []) + post
        ax.plot(
            [point["x"] for point in connector],
            [point["value"] for point in connector],
            color=post_color,
            linestyle=linestyle,
            linewidth=max(0.8, linewidth - 0.15),
            marker=marker,
            markersize=3.4,
            markerfacecolor="white",
            markeredgecolor=post_color,
            markeredgewidth=0.8,
            alpha=0.82,
            label=post_label,
            zorder=3,
        )
    return pre, post


def _add_descriptive_fit(
    ax: Axes,
    points: Sequence[dict[str, Any]],
    effect: dict[str, Any],
    *,
    color: str,
    show_annotation: bool = True,
) -> None:
    used = [point for point in points if point["used_in_h1_trend"]]
    if len(used) < 4 or effect.get("availability_status") != "available":
        return
    xs = np.asarray([point["x"] for point in used], dtype=float)
    ys = np.asarray([point["value"] for point in used], dtype=float)
    slope = _required_float(effect.get("slope_per_gpu_hour"), "slope_per_gpu_hour")
    intercept = float(np.mean(ys) - slope * np.mean(xs))
    fit_x = np.asarray([float(xs.min()), float(xs.max())])
    ax.plot(
        fit_x,
        intercept + slope * fit_x,
        color=color,
        linewidth=2.0,
        linestyle=(0, (6, 3)),
        zorder=6,
    )
    if show_annotation:
        sign = ">" if slope > 0 else "<" if slope < 0 else "="
        ax.text(
            0.985,
            0.08,
            f"eligible descriptive OLS (n={len(used)}): β {sign} 0",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.2,
            color=color,
        )


def _status_box(
    ax: Axes,
    text: str,
    *,
    x: float = 0.985,
    y: float = 0.94,
    ha: str = "right",
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=8.1,
        color=CHARCOAL,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": GRID_GREY,
            "linewidth": 0.6,
            "alpha": 0.9,
        },
        zorder=10,
        clip_on=False,
    )


def _plot_supply_inset(
    ax: Axes,
    points: Sequence[dict[str, Any]],
    aligned: Sequence[dict[str, Any]],
    plateau: dict[str, Any],
) -> None:
    """Magnify the low-valued supply observations around plateau onset."""
    checkpoint_gpu = _checkpoint_gpu_lookup(aligned)
    onset_checkpoint = int(plateau["plateau_start_checkpoint"])
    onset_x = _checkpoint_x(onset_checkpoint, checkpoint_gpu, "plateau onset")
    inset = ax.inset_axes([0.42, 0.43, 0.27, 0.31], zorder=12)
    inset.set_facecolor("white")
    inset.axvspan(onset_x, 5.0, color=POST_ONSET_GREY, zorder=-5)
    inset.axvline(
        onset_x,
        color=CHARCOAL,
        linestyle=(0, (4, 3)),
        linewidth=0.8,
        zorder=3,
    )
    _plot_pre_post_series(
        inset,
        points,
        label="_nolegend_",
        post_label="_nolegend_",
        pre_color=DEEP_BLUE,
        post_color=LIGHT_BLUE,
        marker="o",
        linewidth=0.8,
    )
    inset.set_xlim(1.0, 5.0)
    inset.set_ylim(0.0, 100.0)
    inset.set_title(
        "Local view (1–5 GPU-h; 0–100)",
        loc="left",
        fontsize=6.9,
        pad=3,
    )
    inset.text(
        onset_x + 0.05,
        0.985,
        f"ckpt {onset_checkpoint}",
        transform=inset.get_xaxis_transform(),
        fontsize=6.4,
        color=CHARCOAL,
        va="top",
    )
    _style_axis(inset)
    inset.tick_params(axis="both", labelsize=6.3, length=2.5)


def _plot_turnover_inset(
    ax: Axes,
    turnover: Sequence[dict[str, Any]],
) -> None:
    """Magnify the small post-onset turnover fractions."""
    if not turnover:
        return
    inset = ax.inset_axes([0.42, 0.08, 0.25, 0.28], zorder=12)
    inset.set_facecolor("white")
    _plot_post_onset_only(
        inset,
        turnover,
        label="_nolegend_",
        marker="s",
        linestyle="--",
    )
    inset.set_xlim(
        min(point["x"] for point in turnover) - 0.25,
        max(point["x"] for point in turnover) + 0.20,
    )
    inset.set_ylim(0.0, 0.02)
    inset.set_yticks((0.0, 0.01, 0.02))
    inset.set_title(
        "Turnover detail (0–0.02)",
        loc="left",
        fontsize=6.9,
        pad=3,
    )
    _style_axis(inset)
    inset.tick_params(axis="both", labelsize=6.3, length=2.5)


def _plot_strength_panel(
    ax: Axes,
    rows: Sequence[dict[str, Any]],
    onset_checkpoint: int,
) -> None:
    pre = [row for row in rows if row["checkpoint"] <= onset_checkpoint]
    post = [row for row in rows if row["checkpoint"] > onset_checkpoint]
    for subset, color, filled, label in (
        (pre, DEEP_BLUE, True, "Eligible (ckpt ≤ 40)"),
        (post, LIGHT_BLUE, False, "Post-onset description"),
    ):
        scores = np.asarray([row["score"] for row in subset], dtype=float)
        low = np.asarray([row["ci_low"] for row in subset], dtype=float)
        high = np.asarray([row["ci_high"] for row in subset], dtype=float)
        ax.errorbar(
            [row["gpu_hours"] for row in subset],
            scores,
            yerr=np.vstack((scores - low, high - scores)),
            fmt="o-" if filled else "o--",
            color=color,
            ecolor=color,
            markerfacecolor=color if filled else "white",
            markeredgecolor=color,
            markersize=4.5,
            linewidth=1.1,
            elinewidth=0.9,
            capsize=2.8,
            label=label,
            zorder=4,
        )
    ax.legend(loc="lower right", frameon=False, fontsize=7.8, ncol=2)


def _plot_post_onset_only(
    ax: Axes,
    points: Sequence[dict[str, Any]],
    *,
    label: str,
    marker: str,
    linestyle: str,
) -> None:
    if not points:
        return
    ax.plot(
        [point["x"] for point in points],
        [point["value"] for point in points],
        color=LIGHT_GREY,
        linestyle=linestyle,
        linewidth=1.0,
        marker=marker,
        markersize=3.7,
        markerfacecolor="white",
        markeredgecolor=MID_GREY,
        markeredgewidth=0.8,
        alpha=0.85,
        label=label,
        zorder=4,
    )


def _decorate_diagnostic_axes(
    axes: Iterable[Axes],
    aligned: Sequence[dict[str, Any]],
    plateau: dict[str, Any],
) -> None:
    final_x = max(row["gpu_hours"] for row in aligned)
    for ax in axes:
        _shade_time_context(ax, aligned, plateau)
        _style_axis(ax)
        ax.set_xlim(-0.25, final_x + 0.45)


def plot_temporally_aligned_diagnostics(
    diagnostic_path: Path = DEFAULT_PACKAGE_DIR / "diagnostic_time_series.csv",
    aligned_path: Path = DEFAULT_PACKAGE_DIR / "aligned_checkpoint_metrics.csv",
    effects_path: Path = DEFAULT_PACKAGE_DIR / "h1_effects.csv",
    plateau_path: Path = DEFAULT_PACKAGE_DIR / "plateau.json",
    decision_path: Path = DEFAULT_PACKAGE_DIR / "h1_decision.json",
    output_path: Path = DEFAULT_OUTPUT_DIR / "figure_5_2_h1_evidence.png",
    *,
    dpi: int = 300,
) -> Path:
    """Render Figure 5.2 from the frozen long-form H1 evidence package."""
    diagnostics = _load_diagnostics(Path(diagnostic_path))
    aligned = _load_aligned(Path(aligned_path))
    effects = _load_effects(Path(effects_path))
    plateau = _read_json(Path(plateau_path))
    decision = _read_json(Path(decision_path))
    if len(aligned) != 12:
        raise ValueError(f"Figure 5.2 requires 12 checkpoints, found {len(aligned)}")
    if not plateau.get("plateau_detected"):
        raise ValueError("Figure 5.2 requires a reproduced plateau")
    checkpoint_gpu = _checkpoint_gpu_lookup(aligned)
    onset_checkpoint = int(plateau["plateau_start_checkpoint"])

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": CHARCOAL,
            "text.color": CHARCOAL,
        }
    ):
        fig = plt.figure(figsize=(12.6, 17.6))
        grid = fig.add_gridspec(
            7,
            1,
            height_ratios=(1.35, 1.20, 0.95, 1.06, 1.02, 0.98, 0.98),
            hspace=0.34,
        )
        axes: list[Axes] = []
        for index in range(7):
            axes.append(
                fig.add_subplot(grid[index, 0], sharex=axes[0] if axes else None)
            )
        ax_a, ax_b, ax_c1, ax_c2, ax_c3, ax_d1, ax_d2 = axes
        _decorate_diagnostic_axes(axes, aligned, plateau)

        _plot_strength_panel(ax_a, aligned, onset_checkpoint)
        ax_a.set_title(
            "(A) Playing strength context — fixed-basket macro score",
            loc="left",
            fontsize=11.2,
            fontweight="bold",
            pad=6,
        )
        ax_a.set_ylabel("Macro score", fontsize=9.2)
        _status_box(ax_a, "Plateau context")
        _shade_time_context(
            ax_a,
            aligned,
            plateau,
            label_onset=True,
            label_confirmation=True,
        )

        supply = _metric_points(
            diagnostics, "fresh_states_per_update", checkpoint_gpu
        )
        eligible_supply, _ = _plot_pre_post_series(
            ax_b,
            supply,
            label="Eligible (ckpt ≤ 40)",
            post_label="Post-onset continuation",
            pre_color=DEEP_BLUE,
            post_color=LIGHT_BLUE,
            marker="o",
        )
        ax_b.set_ylim(bottom=0)
        ax_b.set_title(
            "(B) Fresh-state supply",
            loc="left",
            fontsize=11.2,
            fontweight="bold",
            pad=6,
        )
        ax_b.set_ylabel("Fresh states\nper update", fontsize=9.2)
        _status_box(
            ax_b,
            (
                f"Eligible observations (ckpt ≤ 40): n={len(eligible_supply)}\n"
                "Descriptive OLS slope < 0"
            ),
        )

        exposure = _metric_points(
            diagnostics, "mean_sample_exposure", checkpoint_gpu
        )
        _plot_pre_post_series(
            ax_c1,
            exposure,
            label="Mean sample exposure",
            pre_color=ORANGE,
            post_color=PALE_ORANGE,
            marker="o",
        )
        _add_descriptive_fit(
            ax_c1,
            exposure,
            effects["mean_sample_exposure"],
            color=ORANGE,
            show_annotation=False,
        )
        ax_c1.set_title(
            "(C1) Replay reuse — mean sample exposure",
            loc="left",
            fontsize=10.4,
            fontweight="bold",
            pad=5,
        )
        ax_c1.set_ylabel("Exposures", fontsize=9.0)
        _status_box(ax_c1, "Eligible trend assessable (ckpt ≤ 40, n=40)")

        mean_age = _metric_points(diagnostics, "mean_sample_age", checkpoint_gpu)
        p90_age = _metric_points(diagnostics, "p90_sample_age", checkpoint_gpu)
        eligible_mean_age, _ = _plot_pre_post_series(
            ax_c2,
            mean_age,
            label="Mean sample age",
            pre_color=DEEP_BLUE,
            post_color=LIGHT_BLUE,
            marker="o",
            linestyle="-",
        )
        eligible_p90_age, _ = _plot_pre_post_series(
            ax_c2,
            p90_age,
            label="P90 sample age",
            pre_color=MID_GREY,
            post_color=LIGHT_GREY,
            marker="^",
            linestyle="--",
        )
        ax_c2.set_title(
            "(C2) Replay age",
            loc="left",
            fontsize=10.4,
            fontweight="bold",
            pad=5,
        )
        ax_c2.set_ylabel("Age\n(iterations)", fontsize=9.0)
        ax_c2.legend(loc="lower right", frameon=False, fontsize=7.7, ncol=2)
        _status_box(
            ax_c2,
            (
                f"Eligible observations: n={min(len(eligible_mean_age), len(eligible_p90_age))}\n"
                "Mean age slope > 0; P90 age slope > 0"
            ),
            x=0.015,
            ha="left",
        )

        unique_ratio = _metric_points(
            diagnostics, "incoming_unique_state_ratio", checkpoint_gpu
        )
        turnover = _metric_points(diagnostics, "turnover_fraction", checkpoint_gpu)
        _plot_post_onset_only(
            ax_c3,
            unique_ratio,
            label="Incoming unique-state ratio",
            marker="o",
            linestyle="-",
        )
        _plot_post_onset_only(
            ax_c3,
            turnover,
            label="Turnover fraction",
            marker="s",
            linestyle="--",
        )
        ax_c3.set_title(
            "(C3) Replay snapshot diversity and buffer turnover",
            loc="left",
            fontsize=10.4,
            fontweight="bold",
            pad=5,
        )
        ax_c3.set_ylabel("Fraction", fontsize=9.0)
        ax_c3.set_ylim(-0.035, 1.02)
        ax_c3.legend(loc="center right", frameon=False, fontsize=7.7)
        _status_box(ax_c3, "Post-onset only")
        snapshot_x = next(
            row["gpu_hours"]
            for row in diagnostics
            if row["metric"] == "incoming_unique_state_ratio"
            and row["iteration"] == 61
        )
        turnover_x = next(
            point["x"] for point in turnover if point["iteration"] == 151
        )
        ax_c3.text(
            snapshot_x + 0.15,
            0.94,
            "observable from iteration 61",
            fontsize=7.8,
            color=MID_GREY,
            va="top",
        )
        ax_c3.text(
            turnover_x + 0.12,
            0.16,
            "turnover observable\nfrom iteration 151",
            fontsize=7.8,
            color=MID_GREY,
            va="bottom",
        )

        for ax, metric, title, ylabel in (
            (
                ax_d1,
                "approx_policy_gap",
                "(D1) Approximate online train–hold-out policy-loss gap",
                "Policy gap",
            ),
            (
                ax_d2,
                "approx_value_gap",
                "(D2) Approximate online train–hold-out value-loss gap",
                "Value gap",
            ),
        ):
            points = _metric_points(diagnostics, metric, checkpoint_gpu)
            pre, _ = _plot_pre_post_series(
                ax,
                points,
                label="Eligible checkpoints (ckpt ≤ 40)",
                post_label="Post-onset descriptive",
                pre_color=DEEP_BLUE,
                post_color=LIGHT_BLUE,
                marker="o",
                connect_pre=False,
            )
            ax.set_title(
                title,
                loc="left",
                fontsize=10.4,
                fontweight="bold",
                pad=5,
            )
            ax.set_ylabel(ylabel, fontsize=9.0)
            _status_box(
                ax,
                f"Eligible checkpoints (ckpt ≤ 40): n={len(pre)}",
            )

        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)
        ax_d2.set_xlabel("Cumulative GPU-hours", fontsize=10.5, labelpad=7)

        fig.suptitle(
            "Figure 5.2  Temporally aligned diagnostic evidence for H1",
            x=0.105,
            y=0.988,
            ha="left",
            fontsize=15,
            fontweight="bold",
        )
        fig.text(
            0.105,
            0.964,
            (
                "Filled solid marks are eligible through checkpoint 40; "
                "open or grey marks are post-onset descriptive observations."
            ),
            ha="left",
            va="top",
            fontsize=9.2,
            color=MID_GREY,
        )
        decision_label = str(decision.get("status", "unknown")).replace("_", " ")
        fig.text(
            0.105,
            0.018,
            (
                f"H1 decision: {decision_label}. The shaded region begins at "
                f"checkpoint {onset_checkpoint}; the darker band is the detector "
                f"confirmation window (checkpoints "
                f"{plateau['confirmation_window_start']}–"
                f"{plateau['confirmation_window_end']}). Approximate gaps subtract "
                "same-iteration logged training loss from hold-out loss; no formal "
                "eligible-set gap regression is reported from two checkpoints."
            ),
            ha="left",
            va="bottom",
            fontsize=8.4,
            color=MID_GREY,
            wrap=True,
        )
        fig.subplots_adjust(left=0.105, right=0.98, top=0.945, bottom=0.075)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor="white")
        plt.close(fig)
    return output_path


def _load_panel_data(
    diagnostic_path: Path,
    aligned_path: Path,
    effects_path: Path,
    plateau_path: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    diagnostics = _load_diagnostics(Path(diagnostic_path))
    aligned = _load_aligned(Path(aligned_path))
    effects = _load_effects(Path(effects_path))
    plateau = _read_json(Path(plateau_path))
    if len(aligned) != 12:
        raise ValueError(f"Figure 5.2 requires 12 checkpoints, found {len(aligned)}")
    if not plateau.get("plateau_detected"):
        raise ValueError("Figure 5.2 requires a reproduced plateau")
    return diagnostics, aligned, effects, plateau


def _add_checkpoint_axis(ax: Axes, aligned: Sequence[dict[str, Any]]) -> None:
    top = ax.secondary_xaxis("top")
    top.set_xticks([row["gpu_hours"] for row in aligned])
    top.set_xticklabels([str(row["checkpoint"]) for row in aligned], fontsize=8)
    top.set_xlabel("Checkpoint", color=MID_GREY, fontsize=9, labelpad=5)
    top.tick_params(axis="x", colors=MID_GREY, length=3, pad=2)
    top.spines["top"].set_color(GRID_GREY)


def _label_plateau_onset(
    ax: Axes,
    aligned: Sequence[dict[str, Any]],
    plateau: dict[str, Any],
) -> None:
    lookup = _checkpoint_gpu_lookup(aligned)
    onset_checkpoint = int(plateau["plateau_start_checkpoint"])
    onset_x = _checkpoint_x(onset_checkpoint, lookup, "plateau onset")
    label_x = onset_x + 0.006 * (
        max(row["gpu_hours"] for row in aligned)
        - min(row["gpu_hours"] for row in aligned)
    )
    ax.text(
        label_x,
        0.94,
        f"Plateau onset: ckpt {onset_checkpoint}",
        transform=ax.get_xaxis_transform(),
        rotation=90,
        rotation_mode="anchor",
        ha="right",
        va="top",
        fontsize=7.8,
        color=CHARCOAL,
    )


def plot_h1_panel_a(
    aligned_path: Path = DEFAULT_PACKAGE_DIR / "aligned_checkpoint_metrics.csv",
    plateau_path: Path = DEFAULT_PACKAGE_DIR / "plateau.json",
    output_path: Path = DEFAULT_OUTPUT_DIR / "figure_5_2_a_playing_strength.png",
    *,
    dpi: int = 300,
) -> Path:
    """Render the standalone Figure 5.2(a) playing-strength panel."""
    aligned = _load_aligned(Path(aligned_path))
    plateau = _read_json(Path(plateau_path))
    if len(aligned) != 12 or not plateau.get("plateau_detected"):
        raise ValueError("Figure 5.2(a) requires 12 checkpoints and a plateau")
    onset_checkpoint = int(plateau["plateau_start_checkpoint"])

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": CHARCOAL,
            "text.color": CHARCOAL,
        }
    ):
        fig, ax = plt.subplots(figsize=(11.8, 5.8))
        _shade_time_context(
            ax,
            aligned,
            plateau,
            label_onset=True,
            label_confirmation=True,
        )
        _plot_strength_panel(ax, aligned, onset_checkpoint)
        _style_axis(ax)
        ax.set_xlim(-0.35, max(row["gpu_hours"] for row in aligned) + 0.45)
        ax.set_ylim(
            min(row["ci_low"] for row in aligned) - 0.012,
            max(row["ci_high"] for row in aligned) + 0.012,
        )
        ax.set_xlabel("Cumulative GPU-hours", fontsize=10.2, labelpad=7)
        ax.set_ylabel("Fixed-basket macro score", fontsize=10.2)
        _add_checkpoint_axis(ax, aligned)

        fig.suptitle(
            "(a) Playing strength and plateau context",
            x=0.10,
            y=0.965,
            ha="left",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            0.10,
            0.895,
            (
                "Fixed-basket macro score (win = 1, draw = 0.5, loss = 0; "
                "four opponents equally weighted). Error bars show 95% "
                "stratified-bootstrap confidence intervals."
            ),
            ha="left",
            va="top",
            fontsize=9.0,
            color=MID_GREY,
        )
        fig.subplots_adjust(left=0.10, right=0.98, top=0.73, bottom=0.15)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor="white")
        plt.close(fig)
    return output_path


def plot_h1_panel_b(
    diagnostic_path: Path = DEFAULT_PACKAGE_DIR / "diagnostic_time_series.csv",
    aligned_path: Path = DEFAULT_PACKAGE_DIR / "aligned_checkpoint_metrics.csv",
    effects_path: Path = DEFAULT_PACKAGE_DIR / "h1_effects.csv",
    plateau_path: Path = DEFAULT_PACKAGE_DIR / "plateau.json",
    output_path: Path = DEFAULT_OUTPUT_DIR / "figure_5_2_b_fresh_state_supply.png",
    *,
    dpi: int = 300,
) -> Path:
    """Render the standalone Figure 5.2(b) supply panel."""
    diagnostics, aligned, effects, plateau = _load_panel_data(
        diagnostic_path, aligned_path, effects_path, plateau_path
    )
    checkpoint_gpu = _checkpoint_gpu_lookup(aligned)
    supply = _metric_points(
        diagnostics, "fresh_states_per_update", checkpoint_gpu
    )
    supply_slope = _required_float(
        effects["fresh_states_per_update"].get("slope_per_gpu_hour"),
        "fresh_states_per_update slope_per_gpu_hour",
    )
    supply_slope_sign = ">" if supply_slope > 0 else "<" if supply_slope < 0 else "="

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": CHARCOAL,
            "text.color": CHARCOAL,
        }
    ):
        fig, ax = plt.subplots(figsize=(11.8, 5.5))
        _shade_time_context(ax, aligned, plateau, label_onset=True)
        pre, _ = _plot_pre_post_series(
            ax,
            supply,
            label="Eligible (ckpt ≤ 40)",
            post_label="Post-onset continuation",
            pre_color=DEEP_BLUE,
            post_color=LIGHT_BLUE,
            marker="o",
        )
        _plot_supply_inset(ax, supply, aligned, plateau)
        _style_axis(ax)
        ax.set_xlim(-0.25, max(row["gpu_hours"] for row in aligned) + 0.45)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Cumulative GPU-hours", fontsize=10.2, labelpad=7)
        ax.set_ylabel("Fresh states per update", fontsize=10.2)
        ax.legend(loc="upper right", frameon=False, fontsize=8.2, ncol=2)
        _status_box(
            ax,
            (
                f"Eligible observations (ckpt ≤ 40): n = {len(pre)}; "
                f"descriptive OLS slope {supply_slope_sign} 0"
            ),
            y=0.27,
        )

        fig.suptitle(
            "(b) Fresh-state supply around plateau onset",
            x=0.10,
            y=0.965,
            ha="left",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            0.10,
            0.895,
            (
                "Iteration-level fresh_states_per_update. Filled points are eligible "
                "observations through checkpoint 40; open points are descriptive "
                "continuation.\nThe inset magnifies the low-valued onset region; the darker band marks "
                "the checkpoint 60–120 confirmation window."
            ),
            ha="left",
            va="top",
            fontsize=9.0,
            color=MID_GREY,
        )
        fig.subplots_adjust(left=0.10, right=0.98, top=0.79, bottom=0.15)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor="white")
        plt.close(fig)
    return output_path


def plot_h1_panel_c(
    diagnostic_path: Path = DEFAULT_PACKAGE_DIR / "diagnostic_time_series.csv",
    aligned_path: Path = DEFAULT_PACKAGE_DIR / "aligned_checkpoint_metrics.csv",
    effects_path: Path = DEFAULT_PACKAGE_DIR / "h1_effects.csv",
    plateau_path: Path = DEFAULT_PACKAGE_DIR / "plateau.json",
    output_path: Path = DEFAULT_OUTPUT_DIR / "figure_5_2_c_replay_diagnostics.png",
    *,
    dpi: int = 300,
) -> Path:
    """Render standalone Figure 5.2(c) as three same-x narrow strips."""
    diagnostics, aligned, effects, plateau = _load_panel_data(
        diagnostic_path, aligned_path, effects_path, plateau_path
    )
    checkpoint_gpu = _checkpoint_gpu_lookup(aligned)
    exposure = _metric_points(
        diagnostics, "mean_sample_exposure", checkpoint_gpu
    )
    mean_age = _metric_points(diagnostics, "mean_sample_age", checkpoint_gpu)
    p90_age = _metric_points(diagnostics, "p90_sample_age", checkpoint_gpu)
    unique_ratio = _metric_points(
        diagnostics, "incoming_unique_state_ratio", checkpoint_gpu
    )
    turnover = _metric_points(diagnostics, "turnover_fraction", checkpoint_gpu)
    mean_age_slope = _required_float(
        effects["mean_sample_age"].get("slope_per_gpu_hour"),
        "mean_sample_age slope_per_gpu_hour",
    )
    p90_age_slope = _required_float(
        effects["p90_sample_age"].get("slope_per_gpu_hour"),
        "p90_sample_age slope_per_gpu_hour",
    )
    mean_age_sign = ">" if mean_age_slope > 0 else "<" if mean_age_slope < 0 else "="
    p90_age_sign = ">" if p90_age_slope > 0 else "<" if p90_age_slope < 0 else "="
    snapshot_x = next(
        row["gpu_hours"]
        for row in diagnostics
        if row["metric"] == "incoming_unique_state_ratio"
        and row["iteration"] == 61
    )
    turnover_x = next(
        row["gpu_hours"]
        for row in diagnostics
        if row["metric"] == "turnover_fraction" and row["iteration"] == 151
    )

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": CHARCOAL,
            "text.color": CHARCOAL,
        }
    ):
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(11.8, 9.2),
            sharex=True,
            gridspec_kw={"height_ratios": (1.0, 1.08, 1.12), "hspace": 0.34},
        )
        ax_c1, ax_c2, ax_c3 = axes
        _decorate_diagnostic_axes(axes, aligned, plateau)
        _label_plateau_onset(ax_c1, aligned, plateau)

        pre_exposure, _ = _plot_pre_post_series(
            ax_c1,
            exposure,
            label="Mean sample exposure",
            pre_color=ORANGE,
            post_color=PALE_ORANGE,
            marker="o",
        )
        _add_descriptive_fit(
            ax_c1,
            exposure,
            effects["mean_sample_exposure"],
            color=ORANGE,
            show_annotation=False,
        )
        ax_c1.set_title(
            "(C1) Replay reuse — mean sample exposure",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            pad=5,
        )
        ax_c1.set_ylabel("Exposures", fontsize=9.2)
        _status_box(
            ax_c1,
            f"Eligible trend assessable (ckpt ≤ 40, n={len(pre_exposure)})",
        )

        pre_age, _ = _plot_pre_post_series(
            ax_c2,
            mean_age,
            label="Mean sample age",
            pre_color=DEEP_BLUE,
            post_color=LIGHT_BLUE,
            marker="o",
            linestyle="-",
        )
        pre_p90_age, _ = _plot_pre_post_series(
            ax_c2,
            p90_age,
            label="P90 sample age",
            pre_color=MID_GREY,
            post_color=LIGHT_GREY,
            marker="^",
            linestyle="--",
        )
        ax_c2.set_title(
            "(C2) Replay age",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            pad=5,
        )
        ax_c2.set_ylabel("Age (iterations)", fontsize=9.2)
        ax_c2.legend(loc="lower right", frameon=False, fontsize=7.8, ncol=2)
        _status_box(
            ax_c2,
            (
                f"Eligible observations: n={min(len(pre_age), len(pre_p90_age))}\n"
                f"Mean age slope {mean_age_sign} 0; "
                f"P90 age slope {p90_age_sign} 0"
            ),
            x=0.30,
            y=0.94,
            ha="left",
        )
        _plot_post_onset_only(
            ax_c3,
            unique_ratio,
            label="Incoming unique-state ratio",
            marker="o",
            linestyle="-",
        )
        _plot_post_onset_only(
            ax_c3,
            turnover,
            label="Turnover fraction",
            marker="s",
            linestyle="--",
        )
        for boundary_x in (snapshot_x, turnover_x):
            ax_c3.axvline(
                boundary_x,
                color=MID_GREY,
                linestyle=(0, (1.5, 2.5)),
                linewidth=0.9,
                zorder=2,
            )
        ax_c3.text(
            snapshot_x + 0.12,
            0.95,
            "Observable from iteration 61",
            fontsize=8.0,
            color=MID_GREY,
            va="top",
        )
        ax_c3.text(
            turnover_x + 0.12,
            0.08,
            "Turnover observable\nfrom iteration 151",
            fontsize=8.0,
            color=MID_GREY,
            va="bottom",
        )
        ax_c3.set_title(
            "(C3) Replay snapshot diversity and buffer turnover",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            pad=5,
        )
        ax_c3.set_ylabel("Fraction", fontsize=9.2)
        ax_c3.set_ylim(-0.035, 1.02)
        ax_c3.legend(
            loc="center right",
            bbox_to_anchor=(1.0, 0.35),
            frameon=False,
            fontsize=7.8,
        )
        _status_box(ax_c3, "Post-onset supplementary evidence only")
        _plot_turnover_inset(ax_c3, turnover)
        ax_c3.set_xlabel("Cumulative GPU-hours", fontsize=10.2, labelpad=7)
        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)

        fig.suptitle(
            "(c) Replay reuse, age, diversity, and turnover",
            x=0.10,
            y=0.982,
            ha="left",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            0.10,
            0.945,
            (
                "Iteration-level replay proxies are observed from training start. "
                "Filled/solid marks are eligible observations through checkpoint 40; "
                "open or grey marks are post-onset.\nThe darker band is the "
                "checkpoint 60–120 confirmation window."
            ),
            ha="left",
            va="top",
            fontsize=8.9,
            color=MID_GREY,
        )
        fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.09)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor="white")
        plt.close(fig)
    return output_path


def _annotate_eligible_checkpoints(
    ax: Axes, points: Sequence[dict[str, Any]]
) -> None:
    for point in points:
        is_onset = int(point["checkpoint"]) == 40
        ax.annotate(
            f"ckpt {point['checkpoint']}",
            xy=(point["x"], point["value"]),
            xytext=(-8, 6) if is_onset else (5, 6),
            textcoords="offset points",
            fontsize=7.8,
            color=DEEP_BLUE,
            ha="right" if is_onset else "left",
            va="bottom",
        )


def plot_h1_panel_d(
    diagnostic_path: Path = DEFAULT_PACKAGE_DIR / "diagnostic_time_series.csv",
    aligned_path: Path = DEFAULT_PACKAGE_DIR / "aligned_checkpoint_metrics.csv",
    plateau_path: Path = DEFAULT_PACKAGE_DIR / "plateau.json",
    output_path: Path = DEFAULT_OUTPUT_DIR / "figure_5_2_d_generalisation_gap.png",
    *,
    dpi: int = 300,
) -> Path:
    """Render standalone Figure 5.2(d) as two gap strips."""
    diagnostics = _load_diagnostics(Path(diagnostic_path))
    aligned = _load_aligned(Path(aligned_path))
    plateau = _read_json(Path(plateau_path))
    if len(aligned) != 12 or not plateau.get("plateau_detected"):
        raise ValueError("Figure 5.2(d) requires 12 checkpoints and a plateau")
    checkpoint_gpu = _checkpoint_gpu_lookup(aligned)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": CHARCOAL,
            "text.color": CHARCOAL,
        }
    ):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(11.8, 7.2),
            sharex=True,
            gridspec_kw={"height_ratios": (1, 1), "hspace": 0.36},
        )
        _decorate_diagnostic_axes(axes, aligned, plateau)
        _label_plateau_onset(axes[0], aligned, plateau)
        for index, (ax, metric, title, ylabel) in enumerate(
            (
                (
                    axes[0],
                    "approx_policy_gap",
                    "(D1) Policy-loss gap",
                    "Policy gap",
                ),
                (
                    axes[1],
                    "approx_value_gap",
                    "(D2) Value-loss gap",
                    "Value gap",
                ),
            )
        ):
            points = _metric_points(diagnostics, metric, checkpoint_gpu)
            pre, _ = _plot_pre_post_series(
                ax,
                points,
                label="Eligible checkpoints (ckpt ≤ 40)",
                post_label="Post-onset descriptive",
                pre_color=DEEP_BLUE,
                post_color=LIGHT_BLUE,
                marker="o",
                connect_pre=False,
            )
            _annotate_eligible_checkpoints(ax, pre)
            ax.set_title(
                title,
                loc="left",
                fontsize=10.6,
                fontweight="bold",
                pad=5,
            )
            ax.set_ylabel(ylabel, fontsize=9.2)
            _status_box(
                ax,
                f"Eligible checkpoints (ckpt ≤ 40): n={len(pre)}",
                y=1.10,
            )
            if index == 0:
                ax.legend(loc="lower right", frameon=False, fontsize=8.0, ncol=2)
        axes[0].tick_params(labelbottom=False)
        axes[1].set_xlabel("Cumulative GPU-hours", fontsize=10.2, labelpad=7)

        fig.suptitle(
            "(d) Approximate online train–hold-out generalisation gaps",
            x=0.10,
            y=0.976,
            ha="left",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            0.10,
            0.928,
            (
                "Approximate gap = hold-out loss − same-iteration logged training loss. "
                "Filled points are the eligible checkpoints 20 and 40;\nopen points are "
                "post-onset descriptive observations. The darker band is the "
                "checkpoint 60–120 confirmation window."
            ),
            ha="left",
            va="top",
            fontsize=8.9,
            color=MID_GREY,
        )
        fig.text(
            0.10,
            0.862,
            "Insufficient eligible checkpoints for trend assessment.",
            ha="left",
            va="top",
            fontsize=9.2,
            color=CHARCOAL,
            fontweight="bold",
        )
        fig.subplots_adjust(left=0.10, right=0.98, top=0.79, bottom=0.11)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, facecolor="white")
        plt.close(fig)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Figures 5.1 and 5.2 from the frozen H1 package."
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=DEFAULT_PACKAGE_DIR,
        help="Directory containing the five frozen h1_v1_1 artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Figure 5.1, the appendix composite, and panels a–d.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    package_dir = args.package_dir.resolve()
    output_dir = args.output_dir.resolve()
    figure_5_1 = plot_baseline_strength(
        package_dir / "aligned_checkpoint_metrics.csv",
        package_dir / "plateau.json",
        output_dir / "figure_5_1_baseline_strength.png",
        dpi=args.dpi,
    )
    figure_5_2_appendix = plot_temporally_aligned_diagnostics(
        package_dir / "diagnostic_time_series.csv",
        package_dir / "aligned_checkpoint_metrics.csv",
        package_dir / "h1_effects.csv",
        package_dir / "plateau.json",
        package_dir / "h1_decision.json",
        output_dir / "figure_5_2_h1_evidence.png",
        dpi=args.dpi,
    )
    figure_5_2_a = plot_h1_panel_a(
        package_dir / "aligned_checkpoint_metrics.csv",
        package_dir / "plateau.json",
        output_dir / "figure_5_2_a_playing_strength.png",
        dpi=args.dpi,
    )
    figure_5_2_b = plot_h1_panel_b(
        package_dir / "diagnostic_time_series.csv",
        package_dir / "aligned_checkpoint_metrics.csv",
        package_dir / "h1_effects.csv",
        package_dir / "plateau.json",
        output_dir / "figure_5_2_b_fresh_state_supply.png",
        dpi=args.dpi,
    )
    figure_5_2_c = plot_h1_panel_c(
        package_dir / "diagnostic_time_series.csv",
        package_dir / "aligned_checkpoint_metrics.csv",
        package_dir / "h1_effects.csv",
        package_dir / "plateau.json",
        output_dir / "figure_5_2_c_replay_diagnostics.png",
        dpi=args.dpi,
    )
    figure_5_2_d = plot_h1_panel_d(
        package_dir / "diagnostic_time_series.csv",
        package_dir / "aligned_checkpoint_metrics.csv",
        package_dir / "plateau.json",
        output_dir / "figure_5_2_d_generalisation_gap.png",
        dpi=args.dpi,
    )
    print(f"Wrote {figure_5_1}")
    print(f"Wrote {figure_5_2_appendix}")
    print(f"Wrote {figure_5_2_a}")
    print(f"Wrote {figure_5_2_b}")
    print(f"Wrote {figure_5_2_c}")
    print(f"Wrote {figure_5_2_d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
