import numpy as np


class Grid:
    def __init__(self, max_x, max_y, init_value=False):
        self.max_x = max_x
        self.max_y = max_y
        self.grid = []
        for x in range(max_x):
            row = [init_value] * max_y
            self.grid.append(row)
        self.grid = np.array(self.grid)

    def __getitem__(self, location):
        row, col = location
        return self.grid[row][col]

    def __setitem__(self, location, value):
        row, col = location
        self.grid[row][col] = value

    def argwhere(self):
        return np.argwhere(self.grid)

    def argwhere_str(self):
        return [coord2str(row, col) for row, col in self.argwhere()]


def coord2str(row, col):
    col_char = chr(col + 65)
    row = row + 1
    return f'{col_char}{row}'


def print_board(board):
    nx = len(board.board)
    ny = len(board.board)
    for y in range(ny - 1, -1, -1):
        # Fence: print(' ---' * nx)
        row_fence_str = '  '

        row_str = str(y) + ' :'
        for x in range(nx):
            # Horizontal fences
            # down_fence = square.get_fence('up')
            if board.fence_center_grid[(y, x - 1)]:
                row_fence_str += '*'
            else:
                row_fence_str += ' '

            if board.horizontal_fence_grid[(y, x)]:  # down_fence is None:
                row_fence_str += ' — ' # '---'
            else:
                row_fence_str += '   '

            # Vertical fences
            # right_fence = square.get_fence('right')
            if board.vertical_fence_grid[(y, x)]:
                right_str = '|'
            else:
                right_str = ' '

            # print(':   ' * nx + ':')
            path_check = ' '
            content = f' {path_check} '
            if board[y, x].is_busy():
                pawn_color = board[y, x].pawn.color
                pawn_char = 'k' if pawn_color == 'black' else 'w'
                content = f'{path_check}{pawn_char} '
            row_str += f'{content}{right_str}'
        if y == ny - 1:
            print('  ' + ' ···' * nx)
        else:
            print(row_fence_str)
        # print('  ' + ' ···' * nx)
        print(row_str + ':')
        # Fence: print('|   ' * nx + '|')
    print('  ' + ' ···' * nx)
    print('  ' + ''.join([f'  {i} ' for i in range(nx)]))
