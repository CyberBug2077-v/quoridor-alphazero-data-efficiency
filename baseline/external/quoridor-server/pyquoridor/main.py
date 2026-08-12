from pyquoridor.board import Board
from pyquoridor.utils import print_board


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    board = Board()

    board.place_fence(row=0, col=0, orientation='h')
    board.place_fence(row=7, col=0, orientation='h')
    board.place_fence(row=3, col=2, orientation='v')
    board.place_fence(row=1, col=0, orientation='v')

    board.move_pawn(player='white', target_row=1, target_col=4)
    print_board(board)


