from pyquoridor.board import Board


def board_setup1():
    board = Board()
    board.place_fence(row=0, col=0, orientation='h')
    board.place_fence(row=7, col=0, orientation='h')
    board.place_fence(row=3, col=2, orientation='v')
    board.place_fence(row=1, col=0, orientation='v')
    board._set_pawn_location('white', target_row=3, target_col=4)
    board._set_pawn_location('black', target_row=5, target_col=4)
    return board
