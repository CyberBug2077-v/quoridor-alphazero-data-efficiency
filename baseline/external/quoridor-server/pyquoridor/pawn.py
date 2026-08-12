class Pawn:
    def __init__(self, square=None, color='white'):
        self.square = square
        self.color = color
        self.square.set_pawn(self)

    def move(self, square):
        self.square.set_pawn(None)
        self.square = square
        self.square.set_pawn(self)

    def wins(self):
        return self.square.winning_square(self.color)