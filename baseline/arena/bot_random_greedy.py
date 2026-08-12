import random

from .bot_interface import BotInterface, BotMove, PlayerColor
from .bot_random import RandomBot
from .bot_greedy import GreedyBot
from pyquoridor.board import Board


class RandomGreedyBot(BotInterface):
    """Bot that plays greedy with a given probability and random otherwise."""

    DEFAULT_GREEDY_PROB = 0.5

    def __init__(self, color: PlayerColor, greedy_prob: float = DEFAULT_GREEDY_PROB):
        super().__init__(color)
        self.greedy_prob = greedy_prob
        self._random = RandomBot(color)
        self._greedy = GreedyBot(color)

    def select_move(self, board: Board) -> BotMove:
        if random.random() < self.greedy_prob:
            return self._greedy.select_move(board)
        return self._random.select_move(board)
