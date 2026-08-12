from __future__ import annotations
import math
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple
from tqdm import tqdm

from pyquoridor.board import Board
from pyquoridor.exceptions import GameOver, InvalidFence, InvalidMove
from pyquoridor.utils import print_board

from .bot_interface import BotInterface, BotMove, FenceMove, PawnMove, PlayerColor
from .utils import summarize_matches, plot_tournament_results, calculate_stable_elo_ratings


class MatchResult:
    """Record of a finished Quoridor game."""

    def __init__(self,
                 winner=None, termination='',
                 moves=None, turns=0,
                 message=None, fault=None,
                 move_times=None, move_counts=None,
                 match_duration=0.0, rollouts=None,
                 bot_types=None):
        self.winner = winner            # 'white', 'black', or None
        self.termination = termination  # e.g. 'win', 'invalid_move', 'max_turns'
        self.moves = moves or []        # list of PawnMove/FenceMove
        self.turns = turns
        self.message = message
        self.fault = fault              # Color responsible for an error/invalid move
        self.move_times = move_times or {'white': 0.0, 'black': 0.0}
        self.move_counts = move_counts or {'white': 0, 'black': 0}
        self.match_duration = match_duration
        self.rollouts = rollouts or {}
        self.total_moves = len(self.moves)
        self.bot_types = bot_types or {}

    def to_dict(self) -> Dict[str, object]:
        return {
            'winner': self.winner,
            'termination': self.termination,
            'turns': self.turns,
            'message': self.message,
            'moves': [m.to_dict() for m in self.moves],
            'fault': self.fault,
            'move_times': dict(self.move_times),
            'move_counts': dict(self.move_counts),
            'match_duration': self.match_duration,
            'rollouts': dict(self.rollouts),
            'total_moves': self.total_moves,
            'bot_types': dict(self.bot_types),
        }
    

def _validate_move(player: PlayerColor, move: BotMove,
                   legal_pawn_moves, legal_fence_moves) -> None:
    """Check that the move is allowed before applying it."""
    if isinstance(move, PawnMove):
        if move.player != player:
            raise InvalidMove(f'Expected {player}, got {move.player}')
        valid_targets = {sq.location for sq in legal_pawn_moves.get(player, {})}
        if move.target not in valid_targets:
            raise InvalidMove(f'Illegal pawn target {move.target} for {player}')
        return

    if isinstance(move, FenceMove):
        if move.player != player:
            raise InvalidFence(f'Expected {player}, got {move.player}')
        row, col = move.anchor
        orientation = move.orientation.lower()
        if orientation not in ('h', 'v'):
            raise InvalidFence(f'Bad fence orientation: {move.orientation}')

        horiz, vert = legal_fence_moves
        grids = horiz if orientation == 'h' else vert
        grid = grids.get(player)
        if grid is None:
            raise InvalidFence(f'No grid for player {player}')

        try:
            val = grid[(row, col)]
        except IndexError:
            raise InvalidFence(f'Illegal fence placement ({row},{col},{orientation})')

        if val == -1:
            raise InvalidFence(f'Illegal fence placement ({row},{col},{orientation})')
        return

    raise InvalidMove(f'Unsupported move type: {type(move).__name__}')


def _opponent(color: PlayerColor) -> PlayerColor:
    """Return the opposite color."""
    return 'black' if color == 'white' else 'white'


def play_game(white_bot: BotInterface, black_bot: BotInterface, board: Optional[Board] = None,
              max_turns: int = 264, verbose: bool = False, show_board: bool = False) -> MatchResult:
    """Run a full game between two bots and return the result."""

    if white_bot.color != 'white' or black_bot.color != 'black':
        raise ValueError('Bots must be initialised with colors "white" and "black".')

    board = board or Board()
    moves: List[BotMove] = []
    bots = {'white': white_bot, 'black': black_bot}
    rollouts = {
        'white': getattr(white_bot, 'rollouts', None),
        'black': getattr(black_bot, 'rollouts', None),
    }
    bot_types = {
        'white': f"{type(white_bot).__module__}.{type(white_bot).__name__}",
        'black': f"{type(black_bot).__module__}.{type(black_bot).__name__}",
    }
    move_times = {'white': 0.0, 'black': 0.0}
    move_counts = {'white': 0, 'black': 0}
    match_start = perf_counter()

    while board.turn < max_turns:
        color = board.current_player()
        bot = bots[color]

        # Let the bot choose a move
        select_start = perf_counter()
        try:
            move = bot.select_move(board)
        except Exception as e:
            elapsed = perf_counter() - select_start
            move_times[color] += elapsed
            return MatchResult(
                winner=None,
                termination='bot_error',
                moves=moves,
                turns=board.turn,
                message=f'{color} bot crashed: {e}',
                fault=color,
                move_times=move_times,
                move_counts=move_counts,
                match_duration=perf_counter() - match_start,
                rollouts=rollouts,
                bot_types=bot_types,
            )
        else:
            elapsed = perf_counter() - select_start
            move_times[color] += elapsed

        # Validate the move against the rules
        legal_pawn_moves, legal_fence_moves = board.legal_moves(player=color)
        try:
            _validate_move(color, move, legal_pawn_moves, legal_fence_moves)
        except (InvalidMove, InvalidFence) as err:
            return MatchResult(
                winner=None,
                termination='invalid_move',
                moves=moves,
                turns=board.turn,
                message=str(err),
                fault=color,
                move_times=move_times,
                move_counts=move_counts,
                match_duration=perf_counter() - match_start,
                rollouts=rollouts,
                bot_types=bot_types,
            )

        # Try to apply the move
        try:
            bot.apply_move(board, move)
        except GameOver as end:
            move_counts[color] += 1
            moves.append(move)
            if verbose:
                print(f'Game over: {end}')
                print_board(board)
            return MatchResult(
                winner=end.winner,
                termination='win',
                moves=moves,
                turns=board.turn,
                message=str(end),
                move_times=move_times,
                move_counts=move_counts,
                match_duration=perf_counter() - match_start,
                rollouts=rollouts,
                bot_types=bot_types,
            )
        else:
            move_counts[color] += 1
            moves.append(move)

        if verbose:
            print(f'Turn {board.turn}: {move.to_dict()}')
        if show_board:
            print_board(board)

    # Game reached the turn limit
    return MatchResult(
        winner=None,
        termination='max_turns',
        moves=moves,
        turns=board.turn,
        message=f'Max turns ({max_turns}) reached.',
        move_times=move_times,
        move_counts=move_counts,
        match_duration=perf_counter() - match_start,
        rollouts=rollouts,
        bot_types=bot_types,
    )



def run_series(white_factory: Callable[[], BotInterface],
               black_factory: Callable[[], BotInterface],
               games: int, max_turns: int = 512, verbose: bool = False,
               show_board: bool = False) -> Tuple[Dict[str, object], List[MatchResult]]:
    """Play many matches and collect statistics."""

    results: List[MatchResult] = []
    matches_payload: List[Dict[str, object]] = []

    for _ in tqdm(range(games), desc='Quoridor matches'):
        result = play_game(
            white_factory(),
            black_factory(),
            max_turns=max_turns,
            verbose=verbose,
            show_board=show_board,
        )
        results.append(result)
        matches_payload.append(result.to_dict())

    stats = summarize_matches(matches_payload)
    bot_types = stats.get('bot_types', {})
    stats['metadata'] = {
        'games': games,
        'max_turns': max_turns,
        'verbose': verbose,
        'show_board': show_board,
        'white_bot_types': ', '.join(bot_types.get('white', [])) if bot_types.get('white') else None,
        'black_bot_types': ', '.join(bot_types.get('black', [])) if bot_types.get('black') else None,
    }

    return stats, results


@dataclass
class MatchupResult:
    """Result of a matchup between two bots (playing both sides)."""
    bot1_name: str
    bot2_name: str
    bot1_wins: int = 0
    bot2_wins: int = 0
    draws: int = 0

    @property
    def total_games(self) -> int:
        return self.bot1_wins + self.bot2_wins + self.draws

    @property
    def bot1_win_rate(self) -> float:
        return self.bot1_wins / self.total_games if self.total_games > 0 else 0.0

    @property
    def bot2_win_rate(self) -> float:
        return self.bot2_wins / self.total_games if self.total_games > 0 else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.total_games if self.total_games > 0 else 0.0

    def elo_diff_for_bot1(self) -> float:
        """Calculate ELO difference from bot1's perspective."""
        if self.total_games == 0:
            return 0.0
        score = (self.bot1_wins + 0.5 * self.draws) / self.total_games
        score = max(0.001, min(0.999, score))  # Clamp to avoid log(0)
        return -400 * math.log10((1 / score) - 1)


def _prep_bot(bot: BotInterface, color: PlayerColor) -> BotInterface:
    """Prepare a bot instance for a new game by resetting state."""
    bot.color = color
    if hasattr(bot, 'reset_mcts'):
        bot.reset_mcts()
    if hasattr(bot, 'turn_count'):
        bot.turn_count = 0
    if hasattr(bot, 'valid_moves_cache'):
        bot.valid_moves_cache = {}
    return bot


def play_matchup(
    bot1_factory: Callable[[], BotInterface],
    bot2_factory: Callable[[], BotInterface],
    bot1_name: str,
    bot2_name: str,
    games_per_side: int = 5,
    max_turns: int = 120,
    verbose: bool = False,
    return_game_results: bool = False,
    show_progress: bool = True,
) -> MatchupResult:
    """
    Play a matchup between two bots, with each bot playing both colors.

    Args:
        bot1_factory: Factory function to create bot1
        bot2_factory: Factory function to create bot2
        bot1_name: Name of bot1 for result tracking
        bot2_name: Name of bot2 for result tracking
        games_per_side: Number of games per color assignment (total = 2x this)
        max_turns: Maximum turns per game
        verbose: Whether to print detailed game progress
        return_game_results: If True, return (MatchupResult, List[MatchResult])
        show_progress: Whether to show tqdm progress bar

    Returns:
        MatchupResult with win/loss/draw counts (or tuple if return_game_results=True)
    """
    result = MatchupResult(bot1_name=bot1_name, bot2_name=bot2_name)
    game_results: List[MatchResult] = []

    # Create bot instances once and reuse across games
    bot1_instance = bot1_factory()
    bot2_instance = bot2_factory()

    total_games = games_per_side * 2
    pbar = tqdm(total=total_games, desc=f"{bot1_name} vs {bot2_name}", disable=not show_progress)

    def update_progress():
        pbar.set_postfix({
            bot1_name: result.bot1_wins,
            bot2_name: result.bot2_wins,
            'draws': result.draws
        })
        pbar.update(1)

    # bot1 as white
    for _ in range(games_per_side):
        bot1 = _prep_bot(bot1_instance, 'white')
        bot2 = _prep_bot(bot2_instance, 'black')
        game_result = play_game(bot1, bot2, max_turns=max_turns, verbose=verbose)
        game_results.append(game_result)

        if game_result.winner == 'white':
            result.bot1_wins += 1
        elif game_result.winner == 'black':
            result.bot2_wins += 1
        else:
            result.draws += 1
        update_progress()

    # bot1 as black
    for _ in range(games_per_side):
        bot1 = _prep_bot(bot1_instance, 'black')
        bot2 = _prep_bot(bot2_instance, 'white')
        game_result = play_game(bot2, bot1, max_turns=max_turns, verbose=verbose)
        game_results.append(game_result)

        if game_result.winner == 'black':
            result.bot1_wins += 1
        elif game_result.winner == 'white':
            result.bot2_wins += 1
        else:
            result.draws += 1
        update_progress()

    pbar.close()

    # Cleanup bots (e.g., JS bots have subprocess cleanup)
    if hasattr(bot1_instance, 'cleanup'):
        bot1_instance.cleanup()
    if hasattr(bot2_instance, 'cleanup'):
        bot2_instance.cleanup()

    if return_game_results:
        return result, game_results
    return result


def run_tournament(
    bots: Dict[str, Tuple[Callable[[], BotInterface], str]],
    games_per_pairing: int,
    max_turns: int = 264,
    k_factor: float = 32.0,  # ELO K-factor
    output_path: Optional[str] = None,
    show_plot: bool = False,
    verbose: bool = False    # Whether to print match progress
) -> Dict[str, object]:
    """
    Run a round-robin tournament between multiple bots with ELO ratings.
    Args:
        bots: maps bot_name -> (bot_factory, config_plot_string)
    Returns:
        Dictionary containing tournament results.
    """
    from itertools import combinations
    from collections import defaultdict

    bot_names = list(bots.keys())
    n_bots = len(bot_names)

    if n_bots < 2:
        raise ValueError("Tournament requires at least 2 bots")

    # Generate all pairings (round-robin)
    pairings = list(combinations(bot_names, 2))
    total_games = len(pairings) * games_per_pairing * 2

    print(f"Starting tournament with {n_bots} bots")
    print(f"Total pairings: {len(pairings)}")
    print(f"Games per pairing: {games_per_pairing} (x2)")
    print(f"Total games: {total_games}")
    print("=" * 60)

    # Track results
    pairing_results: Dict[str, MatchupResult] = {}
    pairing_game_results: Dict[str, List[MatchResult]] = {}
    game_results_for_elo = []
    all_results: List[MatchResult] = []
    bot_move_times: Dict[str, List[float]] = defaultdict(list)

    # Play each pairing using play_matchup
    for bot1_name, bot2_name in pairings:
        pairing_key = f"{bot1_name}_vs_{bot2_name}"

        print(f"\nPairing: {bot1_name} vs {bot2_name}")
        print("-" * 60)

        bot1_factory, _ = bots[bot1_name]
        bot2_factory, _ = bots[bot2_name]

        # Use shared play_matchup function with game results
        matchup_result, game_results = play_matchup(
            bot1_factory=bot1_factory,
            bot2_factory=bot2_factory,
            bot1_name=bot1_name,
            bot2_name=bot2_name,
            games_per_side=games_per_pairing,
            max_turns=max_turns,
            verbose=verbose,
            return_game_results=True,
        )

        pairing_results[pairing_key] = matchup_result
        pairing_game_results[pairing_key] = game_results
        all_results.extend(game_results)

        # Collect move times per bot
        # First half of games: bot1 is white, bot2 is black
        # Second half: bot1 is black, bot2 is white
        half = len(game_results) // 2
        for i, gr in enumerate(game_results):
            if i < half:
                # bot1 as white, bot2 as black
                if gr.move_counts.get('white', 0) > 0:
                    avg_time = gr.move_times.get('white', 0) / gr.move_counts['white']
                    bot_move_times[bot1_name].append(avg_time)
                if gr.move_counts.get('black', 0) > 0:
                    avg_time = gr.move_times.get('black', 0) / gr.move_counts['black']
                    bot_move_times[bot2_name].append(avg_time)
            else:
                # bot1 as black, bot2 as white
                if gr.move_counts.get('black', 0) > 0:
                    avg_time = gr.move_times.get('black', 0) / gr.move_counts['black']
                    bot_move_times[bot1_name].append(avg_time)
                if gr.move_counts.get('white', 0) > 0:
                    avg_time = gr.move_times.get('white', 0) / gr.move_counts['white']
                    bot_move_times[bot2_name].append(avg_time)

        # Convert to ELO format (one entry per game for proper ELO calculation)
        # bot1 wins
        for _ in range(matchup_result.bot1_wins):
            game_results_for_elo.append({
                'bot1': bot1_name,
                'bot2': bot2_name,
                'score_bot1': 1.0
            })
        # bot2 wins
        for _ in range(matchup_result.bot2_wins):
            game_results_for_elo.append({
                'bot1': bot1_name,
                'bot2': bot2_name,
                'score_bot1': 0.0
            })
        # draws
        for _ in range(matchup_result.draws):
            game_results_for_elo.append({
                'bot1': bot1_name,
                'bot2': bot2_name,
                'score_bot1': 0.5
            })

        # Print pairing summary
        print(f"\n{pairing_key} results:")
        print(f"  {bot1_name}: {matchup_result.bot1_wins} wins ({matchup_result.bot1_win_rate*100:.1f}%)")
        print(f"  {bot2_name}: {matchup_result.bot2_wins} wins ({matchup_result.bot2_win_rate*100:.1f}%)")
        if matchup_result.draws > 0:
            print(f"  draws: {matchup_result.draws} ({matchup_result.draw_rate*100:.1f}%)")

    # Calculate ELO by simulating infinite games until ratings stabilize
    elos = calculate_stable_elo_ratings(game_results_for_elo, bot_names, k_factor=k_factor)

    # Print final results
    print("\n" + "=" * 60)
    print("TOURNAMENT COMPLETE")
    print("=" * 60)
    print("\nFinal ELO Ratings:")
    sorted_bots = sorted(elos.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, elo) in enumerate(sorted_bots, 1):
        print(f"{rank}. {name}: {elo:.0f} (Δ{elo-1000:+.0f})")

    # Build pairing_stats for plotting
    pairing_stats = {}
    for pairing_key, result in pairing_results.items():
        game_results = pairing_game_results[pairing_key]
        game_lengths = [gr.total_moves for gr in game_results]
        move_times = []
        for gr in game_results:
            for color in ('white', 'black'):
                if gr.move_counts.get(color, 0) > 0:
                    move_times.append(gr.move_times.get(color, 0) / gr.move_counts[color])

        pairing_stats[pairing_key] = {
            'wins': defaultdict(int, {
                result.bot1_name: result.bot1_wins,
                result.bot2_name: result.bot2_wins,
                'draw': result.draws
            }),
            'game_lengths': game_lengths,
            'move_times': move_times
        }

    tournament_data = {
        'bot_configs': {name: config for name, (_, config) in bots.items()},
        'final_elos': elos,
        'pairing_stats': pairing_stats,
        'pairing_results': {k: v.__dict__ for k, v in pairing_results.items()},
        'bot_move_times': dict(bot_move_times),
        'all_results': all_results,
        'games_per_pairing': games_per_pairing,
        'total_games': total_games,
        'k_factor': k_factor
    }

    if output_path:
        plot_tournament_results(tournament_data, output_path, show=show_plot)

    return tournament_data
