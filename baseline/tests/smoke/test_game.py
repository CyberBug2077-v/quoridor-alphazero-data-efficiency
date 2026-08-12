from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from quoridor.QuoridorGame import QuoridorGame
from quoridor.QuoridorLogic import Board


def set_pawns(
    board: np.ndarray,
    white: tuple[int, int],
    black: tuple[int, int],
) -> np.ndarray:
    positioned = board.copy()
    positioned[0].fill(0)
    positioned[1].fill(0)
    positioned[0][white] = 1
    positioned[1][black] = 1
    return positioned


def pawn_position(board: np.ndarray, layer: int) -> tuple[int, int]:
    position = np.argwhere(board[layer] == 1)
    assert position.shape == (1, 2)
    return tuple(int(coordinate) for coordinate in position[0])


def test_initial_board_shape(initial_state: np.ndarray) -> None:
    assert initial_state.shape == (4, 17, 17)
    assert initial_state.dtype == np.uint8


def test_action_space_size(game: QuoridorGame) -> None:
    assert game.getActionSize() == 136


def test_initial_player_positions(initial_state: np.ndarray) -> None:
    assert pawn_position(initial_state, 0) == (16, 8)
    assert pawn_position(initial_state, 1) == (0, 8)


def test_initial_legal_actions(
    game: QuoridorGame,
    initial_state: np.ndarray,
    fixed_legal_action: int,
) -> None:
    legal = game.getValidMoves(initial_state, 1)

    assert legal.shape == (136,)
    assert legal[fixed_legal_action] == 1
    assert legal[0] == 1
    assert legal[1] == 0
    assert legal[2] == 1
    assert legal[3] == 1
    assert legal[8] == 1


def test_normal_pawn_move(
    game: QuoridorGame,
    initial_state: np.ndarray,
    fixed_legal_action: int,
) -> None:
    next_state, next_player = game.getNextState(
        initial_state,
        1,
        fixed_legal_action,
    )

    assert next_player == -1
    assert pawn_position(next_state, 0) == (14, 8)
    assert pawn_position(next_state, 1) == (0, 8)


def test_jump_over_adjacent_player(
    game: QuoridorGame,
    initial_state: np.ndarray,
) -> None:
    board = set_pawns(initial_state, white=(8, 8), black=(6, 8))

    assert game.getValidMoves(board, 1)[0] == 1
    next_state, _ = game.getNextState(board, 1, 0)
    assert pawn_position(next_state, 0) == (4, 8)


def test_diagonal_move_when_jump_is_blocked(
    game: QuoridorGame,
    initial_state: np.ndarray,
) -> None:
    board = set_pawns(initial_state, white=(8, 8), black=(6, 8))
    board[2][5, 8] = 1
    legal = game.getValidMoves(board, 1)

    assert legal[0] == 0
    assert legal[4] == 1
    assert legal[7] == 1

    next_state, _ = game.getNextState(board, 1, 4)
    assert pawn_position(next_state, 0) == (6, 6)


def test_legal_horizontal_wall_placement(
    game: QuoridorGame,
    initial_state: np.ndarray,
) -> None:
    horizontal_wall = 8
    assert game.getValidMoves(initial_state, 1)[horizontal_wall] == 1

    next_state, next_player = game.getNextState(
        initial_state,
        1,
        horizontal_wall,
    )

    assert next_player == -1
    assert next_state[2][1, 0] == 1
    assert next_state[2][1, 2] == 1
    assert int(next_state[2].sum()) == 2


def test_overlapping_and_crossing_walls_are_illegal(
    game: QuoridorGame,
    initial_state: np.ndarray,
) -> None:
    state_with_wall, _ = game.getNextState(initial_state, 1, 8)
    legal = game.getValidMoves(state_with_wall, 1)

    crossing_vertical_wall = 72
    assert legal[8] == 0
    assert legal[crossing_vertical_wall] == 0


def test_wall_cannot_block_every_path() -> None:
    board = Board(9)
    placements = [
        (1, 42),
        (-1, 83),
        (1, 124),
        (-1, 120),
        (1, 117),
        (-1, 16),
        (1, 46),
        (-1, 26),
        (1, 89),
        (-1, 84),
        (1, 94),
        (-1, 133),
        (1, 74),
        (-1, 49),
        (1, 28),
    ]
    for color, action in placements:
        assert board.get_legal_moves(color)[action] == 1
        board.execute_move(action, color)

    candidate = 101
    wall_type, (x, y) = board.action_from_index(candidate, 1)
    blocks = board.pieces[2] + board.pieces[3]

    assert wall_type == 9
    assert blocks[x, y] == 0
    assert blocks[x - 2, y] == 0
    assert not (blocks[x - 1, y - 1] and blocks[x - 1, y + 1])
    assert board.has_a_path(1, blocks)
    assert board.has_a_path(-1, blocks)

    blocked = blocks.copy()
    blocked[x, y] = 1
    blocked[x - 2, y] = 1
    assert not (board.has_a_path(1, blocked) and board.has_a_path(-1, blocked))
    assert board.get_legal_moves(-1)[candidate] == 0


def test_terminal_state_detection(
    game: QuoridorGame,
    initial_state: np.ndarray,
) -> None:
    white_wins = set_pawns(initial_state, white=(0, 8), black=(2, 6))
    black_wins = set_pawns(initial_state, white=(14, 6), black=(16, 8))

    assert game.getGameEnded(initial_state, 1) == 0
    assert game.getGameEnded(white_wins, 1) == 1
    assert game.getGameEnded(black_wins, 1) == -1


def test_canonical_form_swaps_and_rotates_for_black(
    game: QuoridorGame,
    initial_state: np.ndarray,
) -> None:
    board = set_pawns(initial_state, white=(12, 4), black=(2, 10))
    board[2][5, 4] = 1
    board[3][8, 7] = 1

    assert game.getCanonicalForm(board, 1) is board
    canonical = game.getCanonicalForm(board, -1)

    np.testing.assert_array_equal(canonical[0], np.flip(board[1], axis=(0, 1)))
    np.testing.assert_array_equal(canonical[1], np.flip(board[0], axis=(0, 1)))
    np.testing.assert_array_equal(canonical[2], np.flip(board[3], axis=(0, 1)))
    np.testing.assert_array_equal(canonical[3], np.flip(board[2], axis=(0, 1)))


def test_player_action_and_value_perspective(
    game: QuoridorGame,
    initial_state: np.ndarray,
) -> None:
    next_state, next_player = game.getNextState(initial_state, -1, 0)
    assert next_player == 1
    assert pawn_position(next_state, 1) == (2, 8)

    black_wins = set_pawns(initial_state, white=(14, 6), black=(16, 8))
    assert game.getGameEnded(black_wins, 1) == -1

    black_perspective = game.getCanonicalForm(black_wins, -1)
    assert game.getGameEnded(black_perspective, 1) == 1


def test_get_next_state_does_not_mutate_input(
    game: QuoridorGame,
    initial_state: np.ndarray,
    fixed_legal_action: int,
) -> None:
    before = initial_state.copy()
    next_state, _ = game.getNextState(initial_state, 1, fixed_legal_action)

    np.testing.assert_array_equal(initial_state, before)
    assert not np.shares_memory(initial_state, next_state)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native-module check")
def test_windows_pathfinder_native_module_imports_and_executes(
    initial_state: np.ndarray,
) -> None:
    pathfinder = importlib.import_module("pathFinder")

    assert pathfinder.__file__.lower().endswith(".pyd")
    pathfinder.setup(9)
    legal_walls = pathfinder.legalWalls(initial_state.tobytes())
    assert len(legal_walls) == 128
    assert all(move in (0, 1) for move in legal_walls)
    assert sum(legal_walls) > 0
