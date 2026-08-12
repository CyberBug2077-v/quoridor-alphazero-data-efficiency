"""
Conversion between the AlphaZero board representation (4-channel uint8 numpy,
17×17 grid) and pyquoridor Board objects.

The 17×17 grid encodes a 9×9 board by interleaving cell and wall positions:
  - Pawns sit at even (row, col) indices: pawn at board square (r,c) → grid[r*2][c*2]
  - Horizontal wall anchored at (r,c) → odd row r*2+1, cells (r*2+1, c*2) and (r*2+1, c*2+2)
  - Vertical wall anchored at (r,c) → odd col c*2+1, cells (r*2, c*2+1) and (r*2+2, c*2+1)
  - Channel 0/1: white/black pawn positions
  - Channel 2/3: white/black wall markers

The grid is stored flipped vertically (row 0 = top of board in AZ = player 1's goal row).
pyquoridor also uses row 0 = top, so pyquoridor_to_az flips to align, and az_to_pyquoridor
un-flips.
"""

import numpy as np

try:
    from pyquoridor.board import Board as PyqBoard
    _PYQUORIDOR_AVAILABLE = True
except ImportError:
    _PYQUORIDOR_AVAILABLE = False


def pyquoridor_to_az(board, board_size: int = 9) -> np.ndarray:
    """Convert a pyquoridor Board into the 4-channel AZ numpy array."""
    n = board_size
    dim = 2 * n - 1
    az = np.zeros((4, dim, dim), dtype='uint8')

    wr, wc = board.white_pawn.square.location
    az[0][wr * 2][wc * 2] = 1

    br, bc = board.black_pawn.square.location
    az[1][br * 2][bc * 2] = 1

    for row in range(n - 1):
        for col in range(n - 1):
            try:
                if not board.fence_center_grid[(row, col)]:
                    continue

                is_h = board.horizontal_fence_grid[(row, col)]
                is_v = board.vertical_fence_grid[(row, col)]

                if is_h:
                    owner = board.horizontal_fence_owner.get((row, col))
                    channels = [2] if owner == 'white' else ([3] if owner == 'black' else [2, 3])
                    az_row = row * 2 + 1
                    az_col = col * 2
                    for ch in channels:
                        az[ch][az_row][az_col]     = 1
                        az[ch][az_row][az_col + 2] = 1

                elif is_v:
                    owner = board.vertical_fence_owner.get((row, col))
                    channels = [2] if owner == 'white' else ([3] if owner == 'black' else [2, 3])
                    az_col = col * 2 + 1
                    az_row = row * 2
                    for ch in channels:
                        az[ch][az_row][az_col]     = 1
                        az[ch][az_row + 2][az_col] = 1

            except KeyError:
                continue

    return az[:, ::-1, :].copy()


def az_to_pyquoridor(az_board: np.ndarray, board_size: int = 9,
                     fences_left_white: int = 10,
                     fences_left_black: int = 10) -> "PyqBoard":
    """Reconstruct a pyquoridor Board from an AZ numpy array.

    fences_left_white/black must be passed in — they can't be recovered from
    the array alone (count_fences_remaining infers them from placed walls).
    """
    if not _PYQUORIDOR_AVAILABLE:
        raise ImportError("pyquoridor is not installed")

    n = board_size
    dim = 2 * n - 1

    az = az_board[:, ::-1, :].copy()   # un-flip

    w_pos = np.argmax(az[0])
    w_row, w_col = divmod(w_pos, dim)
    w_row, w_col = w_row // 2, w_col // 2

    b_pos = np.argmax(az[1])
    b_row, b_col = divmod(b_pos, dim)
    b_row, b_col = b_row // 2, b_col // 2

    board = PyqBoard(white_init_row=w_row, white_init_col=w_col,
                     black_init_row=b_row, black_init_col=b_col)
    board.fences_left['white'] = fences_left_white
    board.fences_left['black'] = fences_left_black

    placed_h = set()
    placed_v = set()

    for ch, owner in [(2, 'white'), (3, 'black')]:
        layer = az[ch]
        for r in range(1, dim, 2):
            for c in range(0, dim - 2, 2):
                if layer[r][c] and layer[r][c + 2]:
                    anchor = (r // 2, c // 2)
                    if anchor not in placed_h:
                        placed_h.add(anchor)
                        try:
                            board.place_fence(row=anchor[0], col=anchor[1],
                                              orientation='h', player=owner)
                        except Exception:
                            pass

        for c in range(1, dim, 2):
            for r in range(0, dim - 2, 2):
                if layer[r][c] and layer[r + 2][c]:
                    anchor = (r // 2, c // 2)
                    if anchor not in placed_v:
                        placed_v.add(anchor)
                        try:
                            board.place_fence(row=anchor[0], col=anchor[1],
                                              orientation='v', player=owner)
                        except Exception:
                            pass

    return board


def az_action_to_pyquoridor_move(action: int, player: int, az_board: np.ndarray,
                                  board_size: int = 9):
    """Convert an AZ action index (canonical, player-1 POV) to a pyquoridor move dict.

    Returns {'type': 'pawn', 'row': r, 'col': c}
         or {'type': 'fence', 'row': r, 'col': c, 'orientation': 'h'/'v'}
    Raises ValueError if the action cannot be decoded.
    """
    import sys, os
    az_dir = os.path.join(os.path.dirname(__file__), '..')
    if az_dir not in sys.path:
        sys.path.insert(0, az_dir)

    from quoridor.QuoridorLogic import Board as AZBoard
    from quoridor.QuoridorGame import QuoridorGame

    game = QuoridorGame(board_size)
    n = board_size

    normalized = game.normalizeAction(action, player)

    dummy = AZBoard(n)
    dummy.pieces = az_board.copy()
    result = dummy.action_from_index(normalized, player if normalized < 8 else 1)

    if result is None:
        raise ValueError(f"action_from_index returned None for action {normalized}")

    move_type_int, (x, y) = result
    x, y = int(x), int(y)
    dim = 2 * n - 1

    if move_type_int < 8:
        pq_row = (n - 1) - (x // 2)
        pq_col = y // 2
        return {'type': 'pawn', 'row': pq_row, 'col': pq_col}

    elif move_type_int == 8:
        anchor_row = ((dim - 2) - x) // 2
        anchor_col = y // 2
        return {'type': 'fence', 'row': anchor_row, 'col': anchor_col, 'orientation': 'h'}

    elif move_type_int == 9:
        anchor_row = (n - 1) - (x // 2)
        anchor_col = (y - 1) // 2
        return {'type': 'fence', 'row': anchor_row, 'col': anchor_col, 'orientation': 'v'}

    raise ValueError(f"Unknown move_type_int {move_type_int}")


def count_fences_remaining(az_board: np.ndarray, board_size: int = 9):
    """Return (white_remaining, black_remaining) fences inferred from the AZ board."""
    white_placed = int(np.sum(az_board[2])) // 2
    black_placed = int(np.sum(az_board[3])) // 2
    return max(0, 10 - white_placed), max(0, 10 - black_placed)
