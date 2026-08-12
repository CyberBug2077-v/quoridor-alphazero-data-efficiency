from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Literal, Tuple, Union

from pyquoridor.board import Board

# Type aliases
PlayerColor = Literal['white', 'black']
Orientation = Literal['h', 'v']
Coordinate = Tuple[int, int]


class PawnMove:
    """Represents a pawn move."""
    def __init__(self, player: PlayerColor, target: Coordinate):
        self.player = player
        self.target = target

    def to_dict(self) -> Dict[str, Union[str, int]]:
        row, col = self.target
        return {
            'type': 'pawn',
            'player': self.player,
            'row': row,
            'col': col,
        }


class FenceMove:
    """Represents a fence placement."""
    def __init__(self, player: PlayerColor, anchor: Coordinate, orientation: Orientation):
        self.player = player
        self.anchor = anchor
        self.orientation = orientation

    def to_dict(self) -> Dict[str, Union[str, int]]:
        row, col = self.anchor
        return {
            'type': 'fence',
            'player': self.player,
            'row': row,
            'col': col,
            'orientation': self.orientation,
        }


BotMove = Union[PawnMove, FenceMove]


class BotInterface(ABC):
    """Base class that all bots should inherit from.

    Bots receive the raw pyquoridor.Board object and must return a BotMove.
    """

    def __init__(self, color: PlayerColor):
        self.color = color

    @abstractmethod
    def select_move(self, board: Board) -> BotMove:
        """Return the next move given the current board state."""
        pass

    def apply_move(self, board: Board, move: BotMove) -> None:
        """Apply the chosen move to the game board."""
        if isinstance(move, PawnMove):
            row, col = move.target
            board.move_pawn(player=move.player, target_row=row, target_col=col)

        elif isinstance(move, FenceMove):
            row, col = move.anchor
            orientation = move.orientation.lower()
            if orientation not in ('h', 'v'):
                raise ValueError(f"Invalid fence orientation: {move.orientation}")
            board.place_fence(row=row, col=col, orientation=orientation)

        else:
            raise TypeError(f"Unsupported move type: {type(move).__name__}")
        
    def play_turn(self, board: Board) -> BotMove:
        """Have the bot play one complete turn."""
        move = self.select_move(board)
        self.apply_move(board, move)
        return move

    def legal_moves(self, board: Board):
        """Optional helper: get all legal moves for this bots color."""
        return board.legal_moves(player=self.color)
