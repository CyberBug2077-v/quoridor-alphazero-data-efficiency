from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Union
import math
import random

from pyquoridor.board import Board


def board_to_json(board: Board) -> Dict[str, object]:
    """Serialize a Board into the snapshot format expected by the JS bot."""
    return {
        "turn": board.turn,
        "current": board.current_player(),
        "fences_left": dict(board.fences_left),
        "pawns": {
            "white": _coord_to_dict(board.white_pawn.square.location),
            "black": _coord_to_dict(board.black_pawn.square.location),
        },
        "walls": {
            "h": _fences_of_orientation(board, "h"),
            "v": _fences_of_orientation(board, "v"),
        },
    }


def _coord_to_dict(coord: Tuple[int, int]) -> Dict[str, int]:
    row, col = coord
    return {"row": int(row), "col": int(col)}


def _fences_of_orientation(board: Board, orientation: str) -> List[Dict[str, int]]:
    if orientation not in {"h", "v"}:
        raise ValueError("orientation must be 'h' or 'v'")

    grid = board.horizontal_fence_grid if orientation == "h" else board.vertical_fence_grid
    centers = getattr(board.fence_center_grid, "argwhere", lambda: [])()
    anchors: List[Dict[str, int]] = []
    for row, col in centers:
        row_idx = int(row)
        col_idx = int(col)
        try:
            if orientation == "h":
                if grid[(row_idx, col_idx)] and grid[(row_idx, col_idx + 1)]:
                    anchors.append({"row": row_idx, "col": col_idx})
            else:
                if grid[(row_idx, col_idx)] and grid[(row_idx + 1, col_idx)]:
                    anchors.append({"row": row_idx, "col": col_idx})
        except IndexError:
            continue

    anchors.sort(key=lambda item: (item["row"], item["col"]))
    return anchors


def summarize_matches(matches: Iterable[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate high level statistics from match results."""

    matches_list = list(matches)
    wins = defaultdict(int)
    terminations = defaultdict(int)
    fault_trackers = {
        'invalid_move': defaultdict(int),
        'bot_error': defaultdict(int),
    }
    bot_types_seen = {
        'white': set(),
        'black': set(),
    }

    total_match_duration = 0.0
    total_moves = 0
    total_move_time = defaultdict(float)
    total_move_count = defaultdict(int)

    rollout_buckets = defaultdict(lambda: defaultdict(lambda: {
        'matches': 0,
        'total_player_moves': 0,
        'total_move_time': 0.0,
        'total_match_time': 0.0,
        'total_match_moves': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
    }))
    rollout_draw_summary = defaultdict(lambda: {'draws': 0, 'matches': 0})

    for match in matches_list:
        winner = match.get('winner')
        if winner in (None, 'draw'):
            wins['draw'] += 1
        else:
            wins[str(winner)] += 1

        termination = match.get('termination')
        if termination:
            terminations[str(termination)] += 1

        fault_color = match.get('fault')
        if termination in fault_trackers and fault_color:
            fault_trackers[termination][str(fault_color)] += 1

        match_bot_types = match.get('bot_types') or {}
        for color in ('white', 'black'):
            bot_type = match_bot_types.get(color)
            if bot_type:
                bot_types_seen[color].add(str(bot_type))

        duration = float(match.get('match_duration', 0.0) or 0.0)
        total_match_duration += duration

        moves_in_match = match.get('total_moves')
        if moves_in_match is None:
            moves_in_match = len(match.get('moves', []))
        total_moves += int(moves_in_match or 0)

        move_times = match.get('move_times', {}) or {}
        move_counts = match.get('move_counts', {}) or {}
        for color, value in move_times.items():
            total_move_time[str(color)] += float(value or 0.0)
        for color, count in move_counts.items():
            total_move_count[str(color)] += int(count or 0)

        rollouts = match.get('rollouts', {}) or {}
        for color, rollout_value in rollouts.items():
            try:
                rollout_key = int(rollout_value)
            except (TypeError, ValueError):
                continue
            bucket = rollout_buckets[str(color)][rollout_key]
            bucket['matches'] += 1
            bucket['total_player_moves'] += int(move_counts.get(color, 0) or 0)
            bucket['total_move_time'] += float(move_times.get(color, 0.0) or 0.0)
            bucket['total_match_time'] += duration
            bucket['total_match_moves'] += int(moves_in_match or 0)
            rollout_draw_summary[rollout_key]['matches'] += 1
            winner_color = match.get('winner')
            if winner_color == color:
                bucket['wins'] += 1
            elif winner_color in (None, 'draw'):
                bucket['draws'] += 1
                rollout_draw_summary[rollout_key]['draws'] += 1
            else:
                bucket['losses'] += 1

    match_count = len(matches_list)
    colors = (set(total_move_count.keys()) | set(total_move_time.keys()) | {'white', 'black'})
    averages = {
        'match_duration': total_match_duration / match_count if match_count else 0.0,
        'moves': total_moves / match_count if match_count else 0.0,
        'time_per_move': {},
        'moves_per_player': {},
        'win_rate': {},
    }
    for color in colors:
        move_count = total_move_count.get(color, 0)
        averages['time_per_move'][color] = (
            total_move_time.get(color, 0.0) / move_count if move_count else 0.0
        )
        averages['moves_per_player'][color] = (
            move_count / match_count if match_count else 0.0
        )
        averages['win_rate'][color] = wins.get(color, 0) / match_count if match_count else 0.0

    rollout_metrics: Dict[str, Dict[int, Dict[str, float]]] = {}
    for color, buckets in rollout_buckets.items():
        rollout_metrics[color] = {}
        for rollout_key, data in buckets.items():
            matches_for_rollout = data['matches']
            player_moves = data['total_player_moves']
            rollout_metrics[color][rollout_key] = {
                **data,
                'avg_time_per_move': data['total_move_time'] / player_moves if player_moves else 0.0,
                'avg_player_moves': player_moves / matches_for_rollout if matches_for_rollout else 0.0,
                'avg_match_time': data['total_match_time'] / matches_for_rollout if matches_for_rollout else 0.0,
                'avg_match_moves': data['total_match_moves'] / matches_for_rollout if matches_for_rollout else 0.0,
                'win_rate': data['wins'] / matches_for_rollout if matches_for_rollout else 0.0,
                'draw_rate': data['draws'] / matches_for_rollout if matches_for_rollout else 0.0,
            }

    rollout_draws = {
        rollout_key: {
            'matches': data['matches'],
            'draws': data['draws'],
            'draw_rate': data['draws'] / data['matches'] if data['matches'] else 0.0,
        }
        for rollout_key, data in rollout_draw_summary.items()
    }

    summary = {
        'matches': matches_list,
        'wins': dict(wins),
        'terminations': dict(terminations),
        'faults': {key: dict(value) for key, value in fault_trackers.items()},
        'averages': averages,
        'totals': {
            'match_duration': total_match_duration,
            'moves': total_moves,
            'move_times': dict(total_move_time),
            'move_counts': dict(total_move_count),
        },
        'rollout_metrics': rollout_metrics,
        'rollout_draws': rollout_draws,
        'bot_types': {
            color: sorted(types) for color, types in bot_types_seen.items() if types
        },
    }
    return summary


def calculate_elo_change(player_elo: float, opponent_elo: float,
                         score: float, k_factor: float = 32.0) -> Tuple[float, float]:
    """
    Calculate new ELO ratings after a game using the chess ELO formula.
    Args:
        k_factor: (default 32, standard for chess)
    """
    # Calculate expected scores
    expected_player = 1.0 / (1.0 + math.pow(10, (opponent_elo - player_elo) / 400.0))
    expected_opponent = 1.0 - expected_player

    # Calculate new ratings
    new_player_elo = player_elo + k_factor * (score - expected_player)
    new_opponent_elo = opponent_elo + k_factor * ((1.0 - score) - expected_opponent)

    return new_player_elo, new_opponent_elo


def calculate_stable_elo_ratings(game_results: List[Dict], bot_names: List[str],
                                   k_factor: float = 16.0,
                                   k_decay: float = 0.995,
                                   convergence_threshold: float = 0.1,
                                   max_iterations: int = 5000,) -> Dict[str, float]:
    """
     Calculate ELO by simulating infinite games until ratings converge (Bayeselo-style).

    Args:
        game_results: List of game results, each with 'bot1', 'bot2', 'score_bot1'
        bot_names: List of all bot names
        k_factor: ELO K-factor

    Returns:
        Dictionary mapping bot names to final ELO ratings
    """
    elos = {name: 1000.0 for name in bot_names}
    results_copy = game_results.copy()

    for iteration in range(max_iterations):
        prev_elos = elos.copy()
        random.shuffle(results_copy)
        current_k = max(0.5, k_factor * (k_decay ** iteration))

        for game_result in results_copy:
            bot1 = game_result['bot1']
            bot2 = game_result['bot2']
            score_bot1 = game_result['score_bot1']

            new_elo1, new_elo2 = calculate_elo_change(elos[bot1], elos[bot2], score_bot1, current_k)
            elos[bot1] = new_elo1
            elos[bot2] = new_elo2

        # Standardise ELOS to 1000
        mean_elo = sum(elos.values()) / len(elos)
        for name in elos:
            elos[name] -= (mean_elo - 1000)

        max_change = max(abs(elos[name] - prev_elos[name]) for name in bot_names)
        if max_change < convergence_threshold:
            print(f"ELO converged after {iteration + 1} iterations (max change: {max_change:.4f})")
            break
    else:
        print(f"ELO calculation stopped at max iterations. Final max change: {max_change:.4f}")

    return elos


def plot_rollout_stats(
    series_stats: Union[Dict[str, object], Iterable[Dict[str, object]]],
    show: bool = True,
    metadata: Optional[Dict[str, object]] = None,
):
    """Plot rollout-dependent performance metrics derived from summarize_matches."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to plot JS bot statistics.") from exc

    if isinstance(series_stats, dict):
        stats_iterable: Iterable[Dict[str, object]] = [series_stats]
    else:
        stats_iterable = list(series_stats)

    aggregated = defaultdict(lambda: defaultdict(lambda: {
        'matches': 0,
        'total_player_moves': 0,
        'total_move_time': 0.0,
        'total_match_moves': 0,
        'total_match_time': 0.0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
    }))

    for stats in stats_iterable:
        rollout_metrics = stats.get('rollout_metrics', {}) if isinstance(stats, dict) else {}
        if not isinstance(rollout_metrics, dict):
            continue
        for color, color_metrics in rollout_metrics.items():
            for rollout_value, data in color_metrics.items():
                try:
                    roll = int(rollout_value)
                except (TypeError, ValueError):
                    continue
                bucket = aggregated[color][roll]
                bucket['matches'] += data.get('matches', 0)
                bucket['total_player_moves'] += data.get('total_player_moves', 0)
                bucket['total_move_time'] += data.get('total_move_time', 0.0)
                bucket['total_match_moves'] += data.get('total_match_moves', 0)
                bucket['total_match_time'] += data.get('total_match_time', 0.0)
                bucket['wins'] += data.get('wins', 0)
                bucket['losses'] += data.get('losses', 0)
                bucket['draws'] += data.get('draws', 0)

    if not aggregated:
        raise ValueError("No rollout metrics available to plot.")

    fig, axes_arr = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes_arr.reshape(-1)
    axes_titles = [
        "Avg time per move (s)",
        "Avg total moves per match",
        "Avg match duration (s)",
        "Win rate",
    ]
    for ax, title in zip(axes, axes_titles):
        ax.set_title(title)
        ax.set_xlabel("MCTS rollouts")

    combined_draws = defaultdict(lambda: {'draws': 0.0, 'matches': 0})

    rollout_draw_stats = None
    for stats in stats_iterable:
        draws = stats.get('rollout_draws') if isinstance(stats, dict) else None
        if draws:
            rollout_draw_stats = rollout_draw_stats or defaultdict(lambda: {'draws': 0.0, 'matches': 0})
            for rollout_key, data in draws.items():
                rollout_draw_stats[rollout_key]['draws'] += data.get('draws', 0.0)
                rollout_draw_stats[rollout_key]['matches'] += data.get('matches', 0)

    for color, color_metrics in aggregated.items():
        if not color_metrics:
            continue
        rollouts = sorted(color_metrics)
        if not rollouts:
            continue

        avg_time_per_move = []
        avg_total_moves = []
        avg_match_time = []
        win_rates = []
        for roll in rollouts:
            data = color_metrics[roll]
            matches = data['matches'] or 1
            player_moves = data['total_player_moves']
            avg_time_per_move.append(
                data['total_move_time'] / player_moves if player_moves else 0.0
            )
            avg_total_moves.append(data['total_match_moves'] / matches)
            avg_match_time.append(data['total_match_time'] / matches)
            win_rates.append(data['wins'] / matches)
            combined_draws[roll]['draws'] += data['draws']
            combined_draws[roll]['matches'] += data['matches']

        axes[0].plot(rollouts, avg_time_per_move, marker='o', label=color)
        axes[1].plot(rollouts, avg_total_moves, marker='o', label=color)
        axes[2].plot(rollouts, avg_match_time, marker='o', label=color)
        axes[3].plot(rollouts, win_rates, marker='o', label=f"{color} win")

    draw_source = rollout_draw_stats if rollout_draw_stats is not None else combined_draws

    if draw_source:
        rollout_points = sorted(draw_source)
        axes[3].plot(
            rollout_points,
            [
                (draw_source[r]['draws'] / draw_source[r]['matches'])
                if draw_source[r]['matches'] else 0.0
                for r in rollout_points
            ],
            marker='x',
            linestyle='--',
            color='gray',
            label='draw rate',
        )

    for ax in axes:
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax.legend()

    axes[3].set_ylim(0.0, 1.0)

    fig.tight_layout()

    meta_lines: List[str] = []
    if metadata:
        meta_lines.extend(f"{k}: {v}" for k, v in metadata.items() if v not in (None, ""))
    else:
        collected = []
        for stats in stats_iterable:
            series_meta = stats.get('metadata') if isinstance(stats, dict) else None
            if series_meta:
                collected.append(series_meta)
        if collected:
            keys = sorted({key for meta in collected for key in meta})
            for key in keys:
                values = {
                    meta.get(key) for meta in collected
                    if key in meta and meta.get(key) not in (None, "")
                }
                if not values:
                    continue
                value_str = ", ".join(sorted(str(v) for v in values))
                meta_lines.append(f"{key}: {value_str}")
    if meta_lines:
        fig.text(0.01, 0.02, " | ".join(meta_lines), fontsize=9, ha='left', va='bottom')

    if show:
        plt.show()
    return fig, axes


def plot_tournament_results(
    tournament_data: Dict[str, object],
    output_path: str,
    show: bool = False
):
    """
    Create tournament visualization.

    tournament_data: Contains tournament results with keys:
        - 'bot_configs': Dict mapping bot names to their configuration strings
        - 'final_elos': Dict mapping bot names to final ELO ratings
        - 'pairing_stats': Dict with pairings as keys, each containing:
            - 'wins': Dict
            - 'move_times': List
            - 'game_lengths': List
        - 'bot_move_times': Dict
    """

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError("matplotlib, numpy and seaborn are required to plot tournament results.") from exc


    sns.set_style("whitegrid")
    sns.set_palette("husl")

    bot_configs = tournament_data.get('bot_configs', {})
    final_elos = tournament_data.get('final_elos', {})
    pairing_stats = tournament_data.get('pairing_stats', {})
    bot_move_times = tournament_data.get('bot_move_times', {})

    # Consistent ordering and colors
    bot_names = sorted(final_elos.keys(), key=lambda x: final_elos[x], reverse=True)
    n_bots = len(bot_names)

    bot_colors = {}
    color_palette = sns.color_palette('Set2', n_bots)
    for i, name in enumerate(bot_names):
        bot_colors[name] = color_palette[i]

    # Calculate overall win rate statistics
    total_white_wins = 0
    total_black_wins = 0
    total_draws = 0
    total_timeout_draws = 0
    total_crash_draws = 0
    all_results = tournament_data.get('all_results', [])
    for result in all_results:
        if result.winner == 'white':
            total_white_wins += 1
        elif result.winner == 'black':
            total_black_wins += 1
        else:
            total_draws += 1
            if result.termination == 'max_turns':
                total_timeout_draws += 1
            elif result.termination in ('bot_error', 'invalid_move'):
                total_crash_draws += 1
            else:
                total_timeout_draws += 1

    total_games = len(all_results)

    # Create figure
    fig = plt.figure(figsize=(18, 10 + n_bots * 0.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.4, 2], hspace=0.4, wspace=0.3,
                          left=0.08, right=0.95, top=0.92, bottom=0.12)

    # LEFT SIDE - Top: ELO ratings
    ax_elo = fig.add_subplot(gs[0, 0])
    elos = [final_elos[name] for name in bot_names]
    colors_for_elo = [bot_colors[name] for name in bot_names]
    bars = ax_elo.bar(range(n_bots), elos, color=colors_for_elo, edgecolor='black', linewidth=1)
    # ax_elo.set_xlabel('Bot', fontsize=11)
    ax_elo.set_ylabel('ELO Rating', fontsize=11)
    ax_elo.set_title('Final ELO Ratings', fontsize=13, fontweight='bold')
    ax_elo.set_xticks(range(n_bots))
    ax_elo.set_xticklabels(bot_names, rotation=45, ha='right')
    ax_elo.axhline(y=1000, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Initial ELO')
    ax_elo.legend()
    ax_elo.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, elo) in enumerate(zip(bars, elos)):
        height = bar.get_height()
        ax_elo.text(bar.get_x() + bar.get_width()/2., height,
                f'{elo:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # LEFT SIDE - Middle: White vs Black statistics
    ax_color = fig.add_subplot(gs[1, 0])

    if total_games > 0:
        white_rate = total_white_wins / total_games
        timeout_rate = total_timeout_draws / total_games
        crash_rate = total_crash_draws / total_games
        black_rate = total_black_wins / total_games

        # Create stacked horizontal bar
        bar_height = 0.6
        ax_color.barh(0, white_rate, height=bar_height, color='#F5F5DC', edgecolor='black', linewidth=1.5, label='White Wins')
        ax_color.barh(0, timeout_rate, left=white_rate, height=bar_height, color='#C0C0C0', edgecolor='black', linewidth=1.5, label='Draws (Timeout)')
        ax_color.barh(0, crash_rate, left=white_rate + timeout_rate, height=bar_height, color='#8B5A5A', edgecolor='black', linewidth=1.5, label='Draws (Crash)')
        ax_color.barh(0, black_rate, left=white_rate + timeout_rate + crash_rate, height=bar_height, color='#2F2F2F', edgecolor='black', linewidth=1.5, label='Black Wins')

        # Add percentage labels
        if white_rate > 0.05:
            ax_color.text(white_rate/2, 0, f'{white_rate*100:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')
        if timeout_rate > 0.05:
            ax_color.text(white_rate + timeout_rate/2, 0, f'{timeout_rate*100:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')
        if crash_rate > 0.05:
            ax_color.text(white_rate + timeout_rate + crash_rate/2, 0, f'{crash_rate*100:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold')
        if black_rate > 0.05:
            ax_color.text(white_rate + timeout_rate + crash_rate + black_rate/2, 0, f'{black_rate*100:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    ax_color.set_xlim(0, 1)
    ax_color.set_ylim(-0.8, 0.8)  # More space for labels
    ax_color.set_xticks([])
    ax_color.set_yticks([])
    ax_color.grid(False)
    ax_color.set_xlabel('')
    ax_color.set_title('Overall: White vs Black vs Draw', fontsize=13, fontweight='bold', pad=-9)
    ax_color.legend(loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=8)
    # Remove outer box
    ax_color.spines['left'].set_visible(False)
    ax_color.spines['right'].set_visible(False)
    ax_color.spines['top'].set_visible(False)
    ax_color.spines['bottom'].set_visible(False)
    ax_color.set_facecolor('white')

    # LEFT SIDE - Bottom: Average move time per bot
    ax_time = fig.add_subplot(gs[2, 0])
    avg_times = []
    std_times = []
    for name in bot_names:
        times = bot_move_times.get(name, [])
        if times:
            avg_times.append(np.mean(times))
            std_times.append(np.std(times))
        else:
            avg_times.append(0)
            std_times.append(0)

    colors_for_time = [bot_colors[name] for name in bot_names]
    bars = ax_time.barh(range(n_bots), avg_times, xerr=std_times,
                        capsize=5, color=colors_for_time, edgecolor='black')
    ax_time.set_ylabel('Bot', fontsize=11)
    ax_time.set_xlabel('Average Move Time (s)', fontsize=11)
    ax_time.set_title('Average Move Time per Bot', fontsize=13, fontweight='bold')
    ax_time.set_yticks(range(n_bots))
    ax_time.set_yticklabels(bot_names)
    ax_time.invert_yaxis()

    # RIGHT SIDE - Matchup results (spanning all rows)
    ax_matchups = fig.add_subplot(gs[:, 1])
    ax_matchups.set_xlim(-0.15, 1.15)
    ax_matchups.set_ylim(-0.5, len(pairing_stats) + 0.5)

    # Calculate total games across all pairings for subtitle
    total_pairing_games = sum(sum(stats.get('wins', {}).values()) for stats in pairing_stats.values())

    # Set title and subtitle properly
    ax_matchups.set_title('Head-to-Head Matchups (Aggregated)\n' +
                          f'Win rate and average game length over {total_pairing_games} games',
                          fontsize=13, fontweight='bold', pad=15)
    ax_matchups.set_xlabel('')
    ax_matchups.set_xticks([])
    ax_matchups.set_yticks([])
    ax_matchups.grid(False)

    # Collect all pairings and sort for consistent display
    sorted_pairings = sorted(pairing_stats.keys())
    bar_height = 0.6
    text_offset = bar_height / 2 + 0.06

    y_pos = 0
    for pairing_key in sorted_pairings:
        stats = pairing_stats[pairing_key]
        wins = stats.get('wins', {})
        game_lengths = stats.get('game_lengths', [])

        # Extract bot names from pairing key
        bot1, bot2 = pairing_key.split('_vs_')

        # Calculate totals
        total = sum(wins.values())
        if total == 0:
            continue

        bot1_wins = wins.get(bot1, 0)
        bot2_wins = wins.get(bot2, 0)
        draws = wins.get('draw', 0)
        avg_game_length = np.mean(game_lengths) if game_lengths else 0

        # Calculate rates
        bot1_rate = bot1_wins / total
        draw_rate = draws / total
        bot2_rate = bot2_wins / total

        # Draw stacked horizontal bar with bot-specific colors
        bar_height = 0.6
        ax_matchups.barh(y_pos, bot1_rate, height=bar_height, color=bot_colors[bot1], edgecolor='black', linewidth=1)
        ax_matchups.barh(y_pos, draw_rate, left=bot1_rate, height=bar_height, color='lightgray', edgecolor='black', linewidth=1)
        ax_matchups.barh(y_pos, bot2_rate, left=bot1_rate + draw_rate, height=bar_height, color=bot_colors[bot2], edgecolor='black', linewidth=1)

        # Add labels inside bars
        if bot1_rate > 0.08:
            ax_matchups.text(bot1_rate/2, y_pos, f'{bot1_rate*100:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold')
        if draw_rate > 0.08:
            ax_matchups.text(bot1_rate + draw_rate/2, y_pos, f'{draw_rate*100:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold')
        if bot2_rate > 0.08:
            ax_matchups.text(bot1_rate + draw_rate + bot2_rate/2, y_pos, f'{bot2_rate*100:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold')

        # Add bot names on left and right of bar, as well as average game length
        ax_matchups.text(-0.02, y_pos, bot1, ha='right', va='center', fontsize=9, fontweight='bold', color=bot_colors[bot1])
        ax_matchups.text(1.02, y_pos, bot2, ha='left', va='center', fontsize=9, fontweight='bold', color=bot_colors[bot2])
        ax_matchups.text(0.5, y_pos + text_offset, f'{avg_game_length:.0f} moves', ha='center', va='top', fontsize=8, color='navy')

        y_pos += 1

    ax_matchups.invert_yaxis()
    ax_matchups.spines['left'].set_visible(False)
    ax_matchups.spines['right'].set_visible(False)
    ax_matchups.spines['top'].set_visible(False)
    ax_matchups.spines['bottom'].set_visible(False)

    # Save figure with high quality
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"Tournament results saved to: {output_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig
