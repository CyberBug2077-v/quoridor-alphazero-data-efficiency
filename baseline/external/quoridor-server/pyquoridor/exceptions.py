class InvalidSquare(Exception):
    def __init__(self, message):
        super().__init__(message)


class InvalidMove(Exception):
    def __init__(self, message):
        super().__init__(message)


class InvalidFence(Exception):
    def __init__(self, message):
        super().__init__(message)


class TrappedPawn(Exception):
    def __init__(self, message):
        super().__init__(message)


class GameOver(Exception):
    def __init__(self, winner, last_move, message):
        super().__init__(message)
        self.last_move = last_move
        self.winner = winner

class InvalidPlayer(Exception):
    def __init__(self, message):
        super().__init__(message)