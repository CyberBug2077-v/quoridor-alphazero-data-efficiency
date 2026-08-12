from pyquoridor.exceptions import InvalidSquare
from pyquoridor.utils import coord2str

# Square constants
MIN_ROW = 0
MIN_COL = 0
MAX_ROW = 9
MAX_COL = 9

WHITE_INIT_ROW = 0
WHITE_INIT_COL = 4
BLACK_INIT_ROW = 8
BLACK_INIT_COL = 4

DIRECTIONS = {'down': 0, 'right': 1, 'up': 2, 'left': 3}
OPPOSITE_DIRECTIONS = {'down': 'up', 'right': 'left', 'up': 'down', 'left': 'right'}


class Square:
    def __init__(self, row, col, pawn=None):
        if not self.valid_square(row, col):
            raise InvalidSquare(f'Invalid square: ({row}, {col})')
        self.row = row
        self.col = col
        self.pawn = pawn
        self.neighbours = set()
        self.physical_neighbours = set()  # [None, None, None, None]  # Up, right, down, left

    def set_physical_neighbours(self, neighbours):
        self.physical_neighbours = neighbours

    def physical_neighbours_iter(self):
        for square in self.physical_neighbours:
            if square is not None:
                yield square

    def neighbours_iter(self):
        for square in self.neighbours:
            if square is not None:
                yield square

    def add_or_remove_neighbours(self, neighbours, add=True):
        try:
            iter(neighbours)
        except TypeError:
            neighbours = [neighbours]

        neighbour_set = set(neighbours)
        if add:
            self.neighbours = self.neighbours.union(neighbour_set)
        else:
            self.neighbours = self.neighbours.difference(neighbour_set)

    def add_neighbours(self, neighbours):
        self.add_or_remove_neighbours(neighbours, add=True)

    def remove_neighbours(self, neighbours):
        self.add_or_remove_neighbours(neighbours, add=False)

    def reset_neighbours(self):
        self.neighbours = self.physical_neighbours.copy()

    def add_or_remove_physical_neighbours(self, neighbours, add=True):
        try:
            iter(neighbours)
        except TypeError:
            neighbours = [neighbours]

        neighbour_set = set(neighbours)
        if add:
            self.physical_neighbours = self.physical_neighbours.union(neighbour_set)
        else:
            self.physical_neighbours = self.physical_neighbours.difference(neighbour_set)

    def add_physical_neighbours(self, neighbours):
        self.add_or_remove_physical_neighbours(neighbours, add=True)

    def remove_physical_neighbours(self, neighbours):
        self.add_or_remove_physical_neighbours(neighbours, add=False)

    def has_pawn(self):
        return self.pawn is not None

    @property
    def location(self):
        return self.row, self.col

    @staticmethod
    def valid_square(x, y):
        return (MIN_ROW <= x < MAX_ROW) & (MIN_COL <= y < MAX_COL)

    def adjacent_coords(self, direction):
        x, y = self.row, self.col
        new_x, new_y = x, y
        if direction == 'right':
            new_x = x + 1
        if direction == 'up':
            new_y = y + 1
        if direction == 'left':
            new_x = x - 1
        if direction == 'down':
            new_y = y - 1
        return new_x, new_y

    def winning_square(self, player='white'):
        if player == 'white':
            return self.row == MAX_ROW - 1
        else:
            return self.row == MIN_ROW

    def set_pawn(self, pawn):
        self.pawn = pawn

    def is_busy(self):
        return self.pawn is not None

    def square2str(self):
        return coord2str(self.row, self.col)

    def __str__(self):
        return f'{self.row}, {self.col}'

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return (self.row, self.col) == (other.row, other.col)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)

    def has_neighbour(self, other):
        return other in self.neighbours
