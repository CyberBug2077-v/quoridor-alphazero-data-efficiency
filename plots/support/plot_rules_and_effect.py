"""Create the two-panel Quoridor rules and wall-effect illustration.

The figure is intentionally independent of the reproduction and experiment
pipelines.  Run it directly; the output path is resolved relative to this file.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


BOARD_SIZE = 9
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "support"
    / "rules_and_effect.png"
)

P1_COLOR = "#2F6B9A"
P2_COLOR = "#D9782D"
GRID_COLOR = "#667085"
WALL_COLOR = "#343A40"
NEW_WALL_COLOR = "#B4235A"
BEFORE_COLOR = "#7A8699"
AFTER_COLOR = "#16856B"

Cell = tuple[int, int]
Wall = tuple[str, int, int]


def setup_board(ax: Axes, title: str) -> None:
    """Draw a square 9x9 board with space around it for annotations."""
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 10.0)
    ax.set_ylim(-1.05, 10.0)
    ax.axis("off")

    ax.add_patch(Rectangle((0, 8), 9, 1, color=P1_COLOR, alpha=0.10, zorder=0))
    ax.add_patch(Rectangle((0, 0), 9, 1, color=P2_COLOR, alpha=0.10, zorder=0))
    for coordinate in range(BOARD_SIZE + 1):
        linewidth = 1.6 if coordinate in (0, BOARD_SIZE) else 0.75
        ax.plot(
            [0, BOARD_SIZE],
            [coordinate, coordinate],
            color=GRID_COLOR,
            linewidth=linewidth,
            zorder=1,
        )
        ax.plot(
            [coordinate, coordinate],
            [0, BOARD_SIZE],
            color=GRID_COLOR,
            linewidth=linewidth,
            zorder=1,
        )

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=8)


def draw_pawn(
    ax: Axes,
    cell: Cell,
    color: str,
    label: str,
    *,
    radius: float = 0.26,
) -> None:
    """Draw a labelled pawn at the centre of a board cell."""
    x, y = cell
    centre = (x + 0.5, y + 0.5)
    ax.add_patch(
        Circle(
            centre,
            radius,
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            zorder=7,
        )
    )
    ax.text(
        *centre,
        label,
        color="white",
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=8,
    )


def draw_wall(
    ax: Axes,
    wall: Wall,
    *,
    color: str = WALL_COLOR,
    linewidth: float = 7.0,
    zorder: int = 6,
) -> None:
    """Draw a legal two-cell Quoridor wall.

    Horizontal walls use ("h", x, y), where y is the boundary between rows
    y-1 and y. Vertical walls use ("v", x, y), where x is the boundary between
    columns x-1 and x.
    """
    orientation, x, y = wall
    if orientation == "h":
        coordinates = ([x + 0.06, x + 1.94], [y, y])
    elif orientation == "v":
        coordinates = ([x, x], [y + 0.06, y + 1.94])
    else:
        raise ValueError(f"Unknown wall orientation: {orientation!r}")
    ax.plot(
        *coordinates,
        color=color,
        linewidth=linewidth,
        solid_capstyle="round",
        zorder=zorder,
    )


def blocked_edges(walls: list[Wall]) -> set[frozenset[Cell]]:
    """Translate wall placements into the board edges they block."""
    blocked: set[frozenset[Cell]] = set()
    for orientation, x, y in walls:
        if orientation == "h":
            for column in (x, x + 1):
                blocked.add(frozenset(((column, y - 1), (column, y))))
        elif orientation == "v":
            for row in (y, y + 1):
                blocked.add(frozenset(((x - 1, row), (x, row))))
        else:
            raise ValueError(f"Unknown wall orientation: {orientation!r}")
    return blocked


def shortest_path(start: Cell, goal_row: int, walls: list[Wall]) -> list[Cell]:
    """Return one shortest wall-respecting path to the requested goal row."""
    blocked = blocked_edges(walls)
    frontier: deque[Cell] = deque([start])
    parent: dict[Cell, Cell | None] = {start: None}

    # Prefer forward motion, then lateral motion, to keep tied routes legible.
    vertical_step = 1 if goal_row > start[1] else -1
    directions = ((0, vertical_step), (1, 0), (-1, 0), (0, -vertical_step))

    destination: Cell | None = None
    while frontier:
        current = frontier.popleft()
        if current[1] == goal_row:
            destination = current
            break
        for dx, dy in directions:
            neighbour = (current[0] + dx, current[1] + dy)
            if not (
                0 <= neighbour[0] < BOARD_SIZE
                and 0 <= neighbour[1] < BOARD_SIZE
            ):
                continue
            if neighbour in parent or frozenset((current, neighbour)) in blocked:
                continue
            parent[neighbour] = current
            frontier.append(neighbour)

    if destination is None:
        raise ValueError(f"No legal path from {start} to row {goal_row}.")

    path: list[Cell] = []
    current: Cell | None = destination
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))


def draw_path(
    ax: Axes,
    path: list[Cell],
    *,
    color: str,
    linestyle: str,
    linewidth: float,
    zorder: int,
    alpha: float = 1.0,
) -> None:
    """Draw a route through the centres of its cells."""
    xs = [x + 0.5 for x, _ in path]
    ys = [y + 0.5 for _, y in path]
    ax.plot(
        xs,
        ys,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        dash_capstyle="round",
        solid_capstyle="round",
        alpha=alpha,
        zorder=zorder,
    )


def path_legend_handles() -> list[Line2D]:
    """Return the shared legend entries for the path-effect panel."""
    return [
        Line2D([0], [0], color=BEFORE_COLOR, linewidth=2.0,
               linestyle=(0, (3, 3)), label="P1 shortest path: before"),
        Line2D([0], [0], color=AFTER_COLOR, linewidth=3.0,
               label="P1 detour: after"),
        Line2D([0], [0], color=P2_COLOR, linewidth=1.8,
               linestyle=(0, (1, 2.2)), label="P2 legal path remains"),
        Line2D([0], [0], color=WALL_COLOR, linewidth=5.0,
               label="Existing wall"),
        Line2D([0], [0], color=NEW_WALL_COLOR, linewidth=5.0,
               label="New wall"),
    ]


def draw_panel_a(ax: Axes) -> None:
    setup_board(ax, "(a) Basic game structure")

    ax.text(
        4.5,
        8.83,
        "PLAYER 1 GOAL LINE",
        color=P1_COLOR,
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        4.5,
        0.17,
        "PLAYER 2 GOAL LINE",
        color=P2_COLOR,
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        va="center",
    )

    draw_pawn(ax, (4, 0), P1_COLOR, "P1")
    draw_pawn(ax, (4, 8), P2_COLOR, "P2")

    # A normal one-square pawn move from the initial position.
    ax.add_patch(
        FancyArrowPatch(
            (4.5, 0.82),
            (4.5, 1.42),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=2.2,
            color=P1_COLOR,
            zorder=5,
        )
    )
    ax.annotate(
        "Pawn move\n(one square)",
        xy=(4.5, 1.18),
        xytext=(5.35, 1.55),
        fontsize=8.5,
        color=P1_COLOR,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", color=P1_COLOR, linewidth=0.9),
    )

    # Direction arrows are offset from the pawns so the initial positions remain clear.
    ax.add_patch(
        FancyArrowPatch(
            (-0.55, 1.0),
            (-0.55, 7.9),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.0,
            color=P1_COLOR,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (9.55, 8.0),
            (9.55, 1.1),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=2.0,
            color=P2_COLOR,
        )
    )
    ax.text(-0.73, 4.5, "P1 forward", rotation=90, color=P1_COLOR, fontsize=8.5,
            ha="center", va="center")
    ax.text(9.74, 4.5, "P2 forward", rotation=-90, color=P2_COLOR, fontsize=8.5,
            ha="center", va="center")

    example_wall: Wall = ("h", 1, 3)
    draw_wall(ax, example_wall, color=NEW_WALL_COLOR)
    ax.annotate(
        "Wall-placement action\n(blocks two edges)",
        xy=(2.0, 3.0),
        xytext=(0.65, 3.75),
        fontsize=8.5,
        color=NEW_WALL_COLOR,
        ha="left",
        va="bottom",
        arrowprops=dict(
            arrowstyle="->", color=NEW_WALL_COLOR, linewidth=1.0,
            connectionstyle="arc3,rad=-0.15",
        ),
    )

    # Put each wall inventory on the player's own side of the board.
    for offset in range(3):
        ax.plot(
            [0.25 + 0.10 * offset, 0.85 + 0.10 * offset],
            [0.62 + 0.08 * offset] * 2,
            color=P1_COLOR,
            linewidth=3.0,
            solid_capstyle="round",
        )
        ax.plot(
            [7.95 + 0.10 * offset, 8.55 + 0.10 * offset],
            [8.38 - 0.08 * offset] * 2,
            color=P2_COLOR,
            linewidth=3.0,
            solid_capstyle="round",
        )
    ax.text(1.18, 0.70, "10 walls", color=P1_COLOR, fontsize=8.5,
            fontweight="bold", ha="left", va="center")
    ax.text(7.68, 8.30, "10 walls", color=P2_COLOR, fontsize=8.5,
            fontweight="bold", ha="right", va="center")


def draw_panel_b(ax: Axes) -> None:
    setup_board(ax, "(b) How a wall changes the shortest path")

    p1 = (4, 2)
    p2 = (7, 6)
    existing_walls: list[Wall] = [
        ("v", 2, 1),
        ("v", 5, 2),
        ("h", 0, 5),
        ("h", 5, 7),
        ("h", 6, 3),
        ("v", 8, 4),
    ]
    new_wall: Wall = ("h", 3, 5)

    path_before = shortest_path(p1, 8, existing_walls)
    path_after = shortest_path(p1, 8, existing_walls + [new_wall])
    p2_legal_path = shortest_path(p2, 0, existing_walls + [new_wall])

    draw_path(
        ax,
        path_before,
        color=BEFORE_COLOR,
        linestyle=(0, (3, 3)),
        linewidth=2.0,
        zorder=2,
        alpha=0.95,
    )
    draw_path(
        ax,
        path_after,
        color=AFTER_COLOR,
        linestyle="-",
        linewidth=3.1,
        zorder=3,
    )
    draw_path(
        ax,
        p2_legal_path,
        color=P2_COLOR,
        linestyle=(0, (1, 2.2)),
        linewidth=1.8,
        zorder=2,
        alpha=0.90,
    )

    for wall in existing_walls:
        draw_wall(ax, wall)
    draw_wall(ax, new_wall, color=NEW_WALL_COLOR, linewidth=8.0, zorder=7)

    draw_pawn(ax, p1, P1_COLOR, "P1")
    draw_pawn(ax, p2, P2_COLOR, "P2")

    # Mark the precise step of the old route that the new wall removes.
    ax.plot(4.5, 5.0, marker="x", markersize=9, markeredgewidth=2.2,
            color=NEW_WALL_COLOR, zorder=8)
    ax.annotate(
        "New wall blocks\nthe direct step",
        xy=(4.5, 5.0),
        xytext=(2.35, 5.8),
        fontsize=8.5,
        color=NEW_WALL_COLOR,
        fontweight="bold",
        ha="left",
        va="bottom",
        arrowprops=dict(
            arrowstyle="->", color=NEW_WALL_COLOR, linewidth=1.1,
            connectionstyle="arc3,rad=-0.12",
        ),
        zorder=9,
    )

    ax.text(
        4.5,
        8.83,
        "P1 goal",
        color=P1_COLOR,
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        4.5,
        0.17,
        "P2 goal",
        color=P2_COLOR,
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        va="center",
    )

def main() -> None:
    """Render and save the publication-ready support figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "axes.titlecolor": "#20242A",
            "text.color": "#20242A",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.2))
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    fig.legend(
        handles=path_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
        fontsize=12.6,
        ncol=5,
        handlelength=2.2,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.13, top=0.96, wspace=0.12)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
