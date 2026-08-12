import random

from .bot_interface import BotInterface, BotMove, PawnMove, FenceMove, PlayerColor
from pyquoridor.board import Board


class RandomBot(BotInterface):
    """Bot that plays random legal moves."""

    def __init__(self, color: PlayerColor):
        super().__init__(color)

    def select_move(self, board: Board) -> BotMove:
        legal_pawn, legal_fence = board.legal_moves(player=self.color)

        moves = []

        # Collect pawn moves
        pawn_squares = legal_pawn.get(self.color, {})
        for square in pawn_squares:
            moves.append(PawnMove(self.color, square.location))

        # Collect fence moves
        horiz_grids, vert_grids = legal_fence
        for orient, grids in [('h', horiz_grids), ('v', vert_grids)]:
            grid = grids.get(self.color)
            if grid:
                for row in range(8):
                    for col in range(8):
                        if grid[(row, col)] != -1:
                            moves.append(FenceMove(self.color, (row, col), orient))

        return random.choice(moves) if moves else None
