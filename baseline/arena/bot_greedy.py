from .bot_interface import BotInterface, BotMove, PawnMove, PlayerColor
from pyquoridor.board import Board


class GreedyBot(BotInterface):
    """Bot that always moves its pawn along the shortest path."""

    def __init__(self, color: PlayerColor):
        super().__init__(color)

    def select_move(self, board: Board) -> BotMove:
        pawn_moves = board.valid_pawn_moves(self.color, path_lengths=True)
        if not pawn_moves:
            return None

        min_len = min(pawn_moves.values())
        best = [sq for sq, dist in pawn_moves.items() if dist == min_len]
        best.sort(key=lambda s: s.location)
        return PawnMove(self.color, best[0].location)
