import sys
import os
import numpy as np
from typing import Optional

from .bot_interface import BotInterface, BotMove, FenceMove, PawnMove, PlayerColor
from .arena import _validate_move
from pyquoridor.board import Board
from pyquoridor.exceptions import InvalidFence, InvalidMove

# Add the AlphaZero repo to path
ALPHAZERO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'external', 'alphazero')
)
sys.path.insert(0, ALPHAZERO_PATH)

from quoridor.QuoridorGame import QuoridorGame
from quoridor.pytorch.NNet import NNetWrapper as nn
from MCTS import MCTS


class AlphaZeroBot(BotInterface):
    """AlphaZero bot that uses a trained neural network to play Quoridor.
    """

    def __init__(
        self,
        color: PlayerColor,
        model_folder: str,
        model_filename: str = 'best.pth.tar',
        board_size: int = 9,
        use_mcts: bool = True,
        fast_mcts: bool = True,
        clear_tree_each_move: bool = True,
        numMCTSSims: int = 50,
        cpuct: float = 1.0,
        temp: float = 0.0,
        eval_mcts_in_batch: int = 1,
        dim_mults: tuple = (1, 2, 4, 8),
        attn_depth: int = 1,
        num_channels: int = 128,
        infer_model_args: bool = True,
        use_cuda: Optional[bool] = None,
        use_amp: Optional[bool] = None,
        amp_dtype: Optional[str] = None,
        strict_legal_moves: Optional[bool] = None,
        trace_mcts: bool = False,
    ):
        """
        Useful Args:
            use_mcts: Whether to use MCTS for move selection (False would use raw policy)
            numMCTSSims: Number of MCTS simulations
            cpuct: MCTS exploration constant
            fast_mcts: Use optimizations for MCTS (caching valid moves)
            temp: Temperature for action selection (0=deterministic, 1=stochastic)
            clear_tree_each_move: Clear MCTS tree before each move
            eval_mcts_in_batch: Batch size for neural network evaluation (1=no batching, 8/16=batched)
            dim_mults: Network architecture dimension multipliers (default: (1,2,4,8))
            attn_depth: Attention block depth (default: 2)
            num_channels: Base number of channels (default: 128)
        """
        super().__init__(color)
        self.board_size = board_size
        self.game = QuoridorGame(board_size)
        self.use_mcts = use_mcts
        self.fast_mcts = fast_mcts
        self.temp = temp
        self.clear_tree_each_move = clear_tree_each_move
        self.trace_mcts = trace_mcts
        self.last_root_counts = None
        self._original_get_valid_moves = self.game.getValidMoves

        if strict_legal_moves is None:
            env_strict = os.environ.get('AZ_STRICT_LEGAL')
            if env_strict is not None:
                strict_legal_moves = env_strict.lower() in ('1', 'true', 'yes', 'y')
            else:
                strict_legal_moves = True
        self.strict_legal_moves = strict_legal_moves

        # Cache for valid moves when using fast_mcts
        self.valid_moves_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        model_path = os.path.join(model_folder, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize neural network with custom architecture args
        import torch
        class dotdict(dict):
            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError:
                    raise AttributeError(key)

        if use_cuda is None:
            env_use_cuda = os.environ.get('AZ_USE_CUDA')
            if env_use_cuda is not None:
                use_cuda = env_use_cuda.lower() in ('1', 'true', 'yes', 'y')
            else:
                use_cuda = torch.cuda.is_available()

        if use_amp is None:
            env_use_amp = os.environ.get('AZ_USE_AMP')
            if env_use_amp is not None:
                use_amp = env_use_amp.lower() in ('1', 'true', 'yes', 'y')
            else:
                use_amp = bool(use_cuda)

        if amp_dtype is None:
            amp_dtype = os.environ.get('AZ_AMP_DTYPE', 'bf16')
        amp_dtype = amp_dtype.lower()
        if amp_dtype not in ('fp16', 'bf16'):
            amp_dtype = 'fp16'

        if infer_model_args:
            inferred = self._infer_model_architecture(model_path)
            if inferred is not None:
                num_channels, attn_depth, dim_mults = inferred
                # print(f"[AlphaZeroBot] Inferred architecture: num_channels={num_channels}, "
                #       f"attn_depth={attn_depth}, dim_mults={dim_mults}")

        custom_args = dotdict({
            'lr': 0.0005,
            'dropout': 0.3,
            'epochs': 4,
            'batch_size': 256,
            'cuda': use_cuda,
            'num_channels': num_channels,
            'num_res_blocks': 6,
            'attn_depth': attn_depth,
            'num_heads': 8,
            'clip': 1.0,
            'weight_decay': 5e-4,
            'dim_mults': dim_mults,
            'use_amp': use_amp,
            'amp_dtype': amp_dtype,
        })

        try:
            self.nnet = nn(self.game, custom_args)
        except RuntimeError as exc:
            if use_cuda and "CUDA" in str(exc):
                print("WARNING: CUDA initialization failed; falling back to CPU for this bot.")
                custom_args['cuda'] = False
                self.nnet = nn(self.game, custom_args)
            else:
                raise

        # Load the trained model
        self.nnet.load_checkpoint(model_folder, model_filename)

        # Initialize MCTS if requested
        self.mcts = None
        self.turn_count = 0
        if use_mcts:
            # Create args dict for MCTS
            class dotdict(dict):
                def __getattr__(self, key):
                    return self[key]
            args = dotdict({
                'numMCTSSims': numMCTSSims,
                'cpuct': cpuct,
                'eval_mcts_in_batch': eval_mcts_in_batch
            })

            if fast_mcts:
                def patched_get_valid_moves(board_state, player):
                    return self._get_valid_moves(board_state, player)

                self.game.getValidMoves = patched_get_valid_moves

            self.mcts = MCTS(self.game, self.nnet, args)
            fast_str = " (fast mode)" if fast_mcts else ""
            # print(f"AlphaZero bot ({color}) loaded model with MCTS{fast_str} (sims={numMCTSSims}, cpuct={cpuct}) from {model_folder}/{model_filename}")
        else:
            # print(f"AlphaZero bot ({color}) loaded model (raw policy) from {model_folder}/{model_filename}")
            pass

    @staticmethod
    def _infer_model_architecture(model_path):
        """Infer num_channels, attn_depth, and dim_mults from a checkpoint."""
        import torch

        try:
            checkpoint = torch.load(model_path, map_location='cpu')
        except Exception:
            return None
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Detect architecture variant
        is_legacy = any(k.startswith(('initial_conv', 'attn_blocks', 'down_blocks'))
                        for k in state_dict)

        if is_legacy:
            num_channels = state_dict['initial_conv.weight'].shape[0]
        elif 'stem.0.weight' in state_dict:
            num_channels = state_dict['stem.0.weight'].shape[0]
        else:
            return None

        attn_indices = set()
        down_indices = set()
        for key in state_dict.keys():
            if key.startswith('attn_blocks.') or key.startswith('attn_layers.'):
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    attn_indices.add(int(parts[1]))
            if key.startswith('down_blocks.'):
                parts = key.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    down_indices.add(int(parts[1]))

        attn_depth = (max(attn_indices) + 1) if attn_indices else 0

        if not down_indices:
            # New architecture (stem/res_blocks) — no dim_mults
            return num_channels, attn_depth, (1, 2, 4, 8)

        dim_mults = []
        for idx in sorted(down_indices):
            prefix = f"down_blocks.{idx}.res_block1.conv1"
            weight_key = None
            for key in state_dict.keys():
                if key.startswith(prefix) and key.endswith('.weight'):
                    tensor = state_dict[key]
                    if tensor.ndim == 4:
                        weight_key = key
                        break
            if weight_key is None:
                continue
            out_ch = state_dict[weight_key].shape[0]
            dim_mult = int(round(out_ch / float(num_channels)))
            if dim_mult <= 0:
                continue
            dim_mults.append(dim_mult)

        if not dim_mults:
            return None
        return num_channels, attn_depth, tuple(dim_mults)

    def reset_mcts(self):
        """Reset MCTS tree to start fresh.

        Call this between games or when you want to clear the search tree.
        The MCTS tree persists across moves by default, which can lead to
        deterministic behavior even with temp > 0.
        """
        if self.mcts is not None:
            self.mcts.Qsa = {}
            self.mcts.Nsa = {}
            self.mcts.Ns = {}
            self.mcts.Ps = {}
            self.mcts.Es = {}
            self.mcts.Vs = {}
            self.mcts.sH = {}

    def select_move(self, board: Board) -> BotMove:
        """Select the best move using the AlphaZero neural network."""
        # Track turn count for fast_mcts heuristics
        self.turn_count += 1

        # Clear MCTS tree if requested
        if self.clear_tree_each_move:
            self.reset_mcts()

        alphazero_board = self._convert_board_to_alphazero(board)

        player = 1 if self.color == 'white' else -1
        canonical_board = self.game.getCanonicalForm(alphazero_board, player) # Board from player's perspective

        # Keep both orientations handy: canonical feeds the search, the original lets us decode moves
        from quoridor.QuoridorLogic import Board as AlphaZeroBoard
        canonical_board_wrapper = AlphaZeroBoard(self.board_size)
        canonical_board_wrapper.pieces = canonical_board.copy()
        original_board_wrapper = AlphaZeroBoard(self.board_size)
        original_board_wrapper.pieces = alphazero_board.copy()
        valid_moves = np.array(canonical_board_wrapper.get_legal_moves(1))
        legal_pawn_moves, legal_fence_moves = board.legal_moves(player=self.color)

        # Select action using MCTS or raw policy
        if self.mcts is not None:
            # Use MCTS for action selection
            # Note: MCTS has known issues with board state synchronization,
            # so we need to validate the action and potentially fall back to raw policy
            try:
                action_probs = np.array(self.mcts.getActionProb(canonical_board, temp=self.temp))
                if self.trace_mcts:
                    s = self.game.stringRepresentation(canonical_board)
                    action_size = self.game.getActionSize()
                    self.last_root_counts = [
                        self.mcts.Nsa.get((s, a), 0) for a in range(action_size)
                    ]
                return self._pick_move_from_distribution(
                    action_probs,
                    valid_moves,
                    board,
                    player,
                    original_board_wrapper,
                    legal_pawn_moves,
                    legal_fence_moves,
                    deterministic=False,
                )
            except Exception as exc:
                # Fall back to raw policy, but keep the error visible for debugging.
                debug = os.environ.get("AZ_MCTS_DEBUG", "").lower() in ("1", "true", "yes", "y")
                print(f"There was an error with MCTS for AlphaZero ({type(exc).__name__}: {exc}). Falling back to raw policy.")
                if debug:
                    import traceback
                    traceback.print_exc()
                self.reset_mcts()  # Clear potentially corrupted search state before continuing.
                if debug:
                    raise

        # Use raw policy from neural network (either as primary mode or fallback)
        policy, value = self.nnet.predict(canonical_board, valid_moves)

        policy = policy * valid_moves   # Mask invalid moves
        if policy.sum() == 0: raise RuntimeError("No valid moves available!")
        try:
            return self._pick_move_from_distribution(
                policy,
                valid_moves,
                board,
                player,
                original_board_wrapper,
                legal_pawn_moves,
                legal_fence_moves,
                deterministic=True,
            )
        except ValueError:
            raise RuntimeError("No legal moves available after filtering policy.")

    def _convert_board_to_alphazero(self, board: Board) -> np.ndarray:
        """Convert a pyquoridor Board into the 4-channel AlphaZero tensor."""
        n = self.board_size
        board_dim = 2 * n - 1

        # Initialize empty board
        alphazero_board = np.zeros((4, board_dim, board_dim), dtype='uint8')

        # Set player positions
        # White pawn
        white_row, white_col = board.white_pawn.square.location
        # Convert to AlphaZero coordinates (interleaved grid)
        white_az_row = white_row * 2
        white_az_col = white_col * 2
        alphazero_board[0][white_az_row][white_az_col] = 1

        # Black pawn
        black_row, black_col = board.black_pawn.square.location
        black_az_row = black_row * 2
        black_az_col = black_col * 2
        alphazero_board[1][black_az_row][black_az_col] = 1

        # Set walls
        # PyQuoridor now tracks which player placed which wall
        # Channel 2: Walls placed by white
        # Channel 3: Walls placed by black

        # Set walls using fence_center_grid to avoid double-counting
        # fence_center_grid is only True at the anchor position of each wall
        for row in range(n - 1):
            for col in range(n - 1):
                try:
                    if not board.fence_center_grid[(row, col)]:
                        continue

                    # Determine if this is a horizontal or vertical wall
                    is_horizontal = board.horizontal_fence_grid[(row, col)]
                    is_vertical = board.vertical_fence_grid[(row, col)]

                    if is_horizontal:
                        # Get the owner of this fence
                        owner = board.horizontal_fence_owner.get((row, col))
                        if owner is None:
                            owner_channel = [2, 3]
                        else:
                            owner_channel = [2] if owner == 'white' else [3]

                        # Mark the wall in AlphaZero format
                        # Horizontal wall: pieces[idx][x][y]=1 and pieces[idx][x][y+2]=1
                        az_row = row * 2 + 1
                        az_col = col * 2
                        for channel in owner_channel:
                            alphazero_board[channel][az_row][az_col] = 1
                            alphazero_board[channel][az_row][az_col + 2] = 1

                    elif is_vertical:
                        # Get the owner of this fence
                        owner = board.vertical_fence_owner.get((row, col))
                        if owner is None:
                            owner_channel = [2, 3]
                        else:
                            owner_channel = [2] if owner == 'white' else [3]

                        # Mark the wall in AlphaZero format
                        # Vertical wall: pieces[idx][x][y]=1 and pieces[idx][x+2][y]=1
                        az_col = col * 2 + 1
                        az_row = row * 2
                        for channel in owner_channel:
                            alphazero_board[channel][az_row][az_col] = 1
                            alphazero_board[channel][az_row + 2][az_col] = 1

                except KeyError:
                    pass

        # Flip vertically to match AlphaZero's top-origin coordinate system
        return alphazero_board[:, ::-1, :].copy()

    def _convert_action_to_move(
        self,
        action: int,
        board: Board,
        player: int,
        az_board
    ) -> BotMove:
        """Convert AlphaZero action index to BotMove.

        AlphaZero action encoding:
        - Actions 0-7: Pawn moves (up, down, left, right, diagonals, jumps)
        - Actions 8+: Wall placements

        Args:
            action: The action index from AlphaZero (already in canonical form, i.e., from player 1's perspective)
            board: The pyquoridor board (for reference)
            player: The player (1 or -1)
            az_board: The AlphaZero Board object with walls set correctly (used for pawn moves)
        """
        # The action index returned by MCTS is in canonical (current-player) orientation.
        # normalizeAction maps it back into the shared (white-at-bottom) orientation so that
        # we can reason about it on the original board.
        normalized_action = self.game.normalizeAction(action, player)

        # For wall placements (actions 8+), we can decode directly without board state
        # For pawn moves (actions 0-7), we need the board state to calculate destination
        if normalized_action >= 8:
            # Wall placement - decode directly
            from quoridor.QuoridorLogic import Board as AlphaZeroBoard
            dummy_board = AlphaZeroBoard(self.board_size)
            result = dummy_board.action_from_index(normalized_action, 1)
        else:
            # Pawn move - calculate destination from current pyquoridor board
            # The action was already validated in valid_moves, so it should be legal
            # We just need to figure out what the destination square is
            #
            # Since normalized_action is 0-7, it represents a direction:
            # 0=up, 1=down, 2=left, 3=right, 4-7=diagonals/jumps
            #
            # However, action_from_index calculates the ACTUAL destination considering
            # walls and opponent position.
            az_player = 1 if player == 1 else -1
            result = az_board.action_from_index(normalized_action, az_player)

            if result is None:
                # Silently raise error - will be caught by MCTS fallback logic
                raise ValueError(f"action_from_index returned None for pawn move action {normalized_action}")

        if result is None:
            print(f"ERROR: action_from_index returned None for action {normalized_action}")
            print(f"Original action: {action}, Player: {player}, Color: {self.color}")
            raise ValueError(f"action_from_index returned None for action {normalized_action}")

        move_type, (x, y) = result

        # Pawn moves (0-7)
        if move_type < 8:
            # Convert AlphaZero board coordinates to pyquoridor coordinates
            # AlphaZero uses (2*n-1) x (2*n-1) grid, pyquoridor uses n x n grid
            target_row = (self.board_size - 1) - int(x // 2)
            target_col = int(y // 2)
            return PawnMove(self.color, (target_row, target_col))

        # Wall placements
        elif move_type == 8:
            # Horizontal wall in AlphaZero format
            # x is the row (odd number like 1, 3, 5, ...)
            # y is the starting column (even number like 0, 2, 4, ...)
            # pyquoridor: anchor is top-left corner of the wall
            anchor_row = ((2 * self.board_size - 3) - int(x)) // 2
            anchor_col = int(y // 2)

            return FenceMove(self.color, (anchor_row, anchor_col), 'h')

        elif move_type == 9:
            # Vertical wall in AlphaZero format
            # x is the starting row (even number: 2, 4, 6, 8, 10, 12, 14, 16)
            # y is the column (odd number: 1, 3, 5, 7, 9, 11, 13, 15)
            # Map AZ coordinates to arena: 2→0, 4→1, 6→2, ..., 16→7
            anchor_row = (self.board_size - 1) - int(x // 2)
            anchor_col = int((y - 1) // 2)

            return FenceMove(self.color, (anchor_row, anchor_col), 'v')

        else:
            raise ValueError(f"Unknown move type: {move_type}")

    def _pick_move_from_distribution(
        self,
        probs,
        valid_moves,
        board: Board,
        player: int,
        az_board,
        legal_pawn_moves,
        legal_fence_moves,
        deterministic: bool = False,
    ) -> BotMove:
        """Sample a move from the provided probability distribution, ensuring legality."""
        weights = np.array(probs, dtype=np.float64, copy=True)
        mask = np.array(valid_moves, dtype=np.float64, copy=True)
        if not np.isfinite(weights).all():
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights *= mask

        if self.strict_legal_moves:
            py_mask = self._build_pyquoridor_mask(
                board,
                player,
                az_board,
                legal_pawn_moves,
                legal_fence_moves,
                len(weights),
            )
            if py_mask.sum() > 0:
                weights *= py_mask
                mask *= py_mask

        # Track failed actions for debug logging
        failed_actions = []

        if weights.sum() <= 0:
            valid_indices = np.flatnonzero(mask)
            if valid_indices.size == 0:
                # No valid actions from AlphaZero - fall back to pyquoridor legal moves
                return self._fallback_to_pyquoridor_legal_move(
                    board, az_board, player, probs, valid_moves,
                    legal_pawn_moves, legal_fence_moves, failed_actions
                )
            action = int(valid_indices[0] if deterministic or self.temp == 0 else np.random.choice(valid_indices))
            try:
                return self._convert_action_to_move(action, board, player, az_board)
            except (ValueError, TypeError) as e:
                failed_actions.append((action, probs[action], str(e)))
                return self._fallback_to_pyquoridor_legal_move(
                    board, az_board, player, probs, valid_moves,
                    legal_pawn_moves, legal_fence_moves, failed_actions
                )

        while weights.sum() > 0:
            if deterministic or self.temp == 0:
                action = int(np.argmax(weights))
            else:
                distribution = weights / weights.sum()
                action = int(np.random.choice(len(weights), p=distribution))

            try:
                move = self._convert_action_to_move(action, board, player, az_board)
            except (ValueError, TypeError) as e:
                failed_actions.append((action, probs[action], f"conversion: {e}"))
                weights[action] = 0
                continue

            if self._is_move_legal(move, legal_pawn_moves, legal_fence_moves):
                return move

            # Include more details about the failed fence move
            if hasattr(move, 'orientation'):
                failed_actions.append((action, probs[action], f"fence({move.anchor[0]},{move.anchor[1]},{move.orientation})"))
            else:
                failed_actions.append((action, probs[action], f"pawn({move.target})"))
            weights[action] = 0

        # All policy moves failed validation - fall back to pyquoridor legal moves
        return self._fallback_to_pyquoridor_legal_move(
            board, az_board, player, probs, valid_moves,
            legal_pawn_moves, legal_fence_moves, failed_actions
        )

    def _build_pyquoridor_mask(
        self,
        board: Board,
        player: int,
        az_board,
        legal_pawn_moves,
        legal_fence_moves,
        action_size: int,
    ) -> np.ndarray:
        """Return a mask of moves that are legal under pyquoridor rules."""
        mask = np.zeros(action_size, dtype=np.float64)
        for action in range(action_size):
            try:
                move = self._convert_action_to_move(action, board, player, az_board)
            except (ValueError, TypeError):
                continue
            if self._is_move_legal(move, legal_pawn_moves, legal_fence_moves):
                mask[action] = 1.0
        return mask

    def _fallback_to_pyquoridor_legal_move(
        self,
        board: Board,
        az_board,
        player: int,
        probs,
        valid_moves,
        legal_pawn_moves,
        legal_fence_moves,
        failed_actions: list
    ) -> BotMove:
        """Last resort: pick directly from pyquoridor's legal moves when policy fails."""
        # Log debug info to file
        self._log_fallback_debug(board, az_board, player, probs, valid_moves,
                                  legal_pawn_moves, legal_fence_moves, failed_actions)

        # Collect all legal pawn moves (prefer these - simpler and more reliable)
        pawn_moves = []
        pawn_targets = legal_pawn_moves.get(self.color, {})
        for sq in pawn_targets:
            pawn_moves.append(PawnMove(self.color, sq.location))

        # If we have pawn moves, pick the first one (deterministic fallback)
        if pawn_moves:
            return pawn_moves[0]

        # Collect all legal fence moves as fallback
        fence_moves = []
        horiz_grid, vert_grid = legal_fence_moves
        h_grid = horiz_grid.get(self.color)
        v_grid = vert_grid.get(self.color)

        if h_grid is not None:
            for (row, col), val in np.ndenumerate(h_grid):
                if val == 1:
                    fence_moves.append(FenceMove(self.color, (row, col), 'h'))

        if v_grid is not None:
            for (row, col), val in np.ndenumerate(v_grid):
                if val == 1:
                    fence_moves.append(FenceMove(self.color, (row, col), 'v'))

        if fence_moves:
            return fence_moves[0]

        raise ValueError("No legal moves available from pyquoridor (game should be over).")

    def _log_fallback_debug(
        self,
        board: Board,
        az_board,
        player: int,
        probs,
        valid_moves,
        legal_pawn_moves,
        legal_fence_moves,
        failed_actions: list
    ):
        """Log debug information when fallback is triggered."""
        import json
        from datetime import datetime

        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'alphazero_fallback_debug.jsonl')

        # Get top policy actions
        top_indices = np.argsort(probs)[-10:][::-1]
        top_policy = [(int(i), float(probs[i]), int(valid_moves[i])) for i in top_indices]

        # Get pyquoridor legal moves
        pawn_targets = [sq.location for sq in legal_pawn_moves.get(self.color, {})]
        horiz_grid, vert_grid = legal_fence_moves
        h_legal = []
        v_legal = []
        h_grid_obj = horiz_grid.get(self.color)
        v_grid_obj = vert_grid.get(self.color)
        if h_grid_obj is not None:
            try:
                h_legal = [(int(r), int(c)) for (r, c) in np.argwhere(h_grid_obj.grid > 0)]
            except Exception as e:
                print(f"DEBUG: h_grid iteration failed: {e}")
        if v_grid_obj is not None:
            try:
                v_legal = [(int(r), int(c)) for (r, c) in np.argwhere(v_grid_obj.grid > 0)]
            except Exception as e:
                print(f"DEBUG: v_grid iteration failed: {e}")

        # Get wall counts
        az_white_walls = int(np.sum(az_board.pieces[2])) if hasattr(az_board, 'pieces') else 'N/A'
        az_black_walls = int(np.sum(az_board.pieces[3])) if hasattr(az_board, 'pieces') else 'N/A'
        pyq_white_fences = board.fences_left.get('white', 'N/A') if hasattr(board, 'fences_left') else 'N/A'
        pyq_black_fences = board.fences_left.get('black', 'N/A') if hasattr(board, 'fences_left') else 'N/A'

        debug_entry = {
            'timestamp': datetime.now().isoformat(),
            'color': self.color,
            'player': player,
            'board_state': str(board),  # pyquoridor board string representation
            'az_board_shape': list(az_board.pieces.shape) if hasattr(az_board, 'pieces') else 'N/A',
            'az_white_wall_markers': az_white_walls,
            'az_black_wall_markers': az_black_walls,
            'pyq_white_fences_left': pyq_white_fences,
            'pyq_black_fences_left': pyq_black_fences,
            'top_policy_actions': top_policy,  # (action_idx, prob, is_valid)
            'failed_actions': failed_actions[:20],  # (action_idx, prob, reason)
            'num_az_valid': int(np.sum(valid_moves)),
            'pyquoridor_pawn_moves': pawn_targets,
            'pyquoridor_h_fences': h_legal[:10],  # Limit size
            'pyquoridor_v_fences': v_legal[:10],
        }

        with open(log_file, 'a') as f:
            f.write(json.dumps(debug_entry) + '\n')

        print(f"[AlphaZeroBot] Fallback triggered - {len(failed_actions)} actions failed. "
              f"Debug logged to {log_file}")

    def _is_move_legal(self, move: BotMove, legal_pawn_moves, legal_fence_moves) -> bool:
        """Validate the move against pyquoridor's legality checks."""
        try:
            _validate_move(self.color, move, legal_pawn_moves, legal_fence_moves)
            return True
        except (InvalidMove, InvalidFence):
            return False

    def _get_valid_moves(self, board_state, player):
        """Return the action mask the MCTS code expects."""
        if self.fast_mcts:
            return self._get_valid_moves_fast(board_state, player)
        if self._original_get_valid_moves is not None:
            return self._original_get_valid_moves(board_state, player)
        return self.game.getValidMoves(board_state, player)

    def _get_probable_walls(self, az_board, turn_count):
        """Get probable wall positions for fast MCTS.

        This implements the JS bot's heuristic: only check walls near
        existing walls and pawns, significantly reducing the search space.

        Returns a binary mask for wall actions (same format as get_legal_moves).
        """
        n = self.board_size
        probable = np.zeros(2 * (n - 1) ** 2, dtype='uint8')

        # Get pawn positions from the board
        p1_pos = np.argwhere(az_board[0] == 1)
        p2_pos = np.argwhere(az_board[1] == 1)

        if len(p1_pos) == 0 or len(p2_pos) == 0:
            return probable

        p1_row, p1_col = p1_pos[0][0] // 2, p1_pos[0][1] // 2
        p2_row, p2_col = p2_pos[0][0] // 2, p2_pos[0][1] // 2

        # Get existing walls
        walls = az_board[2] + az_board[3]

        # For each possible wall position, check if it's "probable"
        wall_idx = 0

        # Horizontal walls
        for row in range(n - 1):
            for col in range(n - 1):
                is_probable = False

                # Near pawns (within 2 squares)
                if abs(row - p1_row) <= 2 and abs(col - p1_col) <= 2:
                    is_probable = True
                if abs(row - p2_row) <= 2 and abs(col - p2_col) <= 2:
                    is_probable = True

                # Near existing walls (within 1 square)
                az_row = row * 2 + 1
                az_col = col * 2
                for dr in [-2, 0, 2]:
                    for dc in [-2, 0, 2]:
                        check_row = az_row + dr
                        check_col = az_col + dc
                        if 0 <= check_row < 2*n-1 and 0 <= check_col < 2*n-1:
                            if walls[check_row][check_col] > 0:
                                is_probable = True
                                break
                    if is_probable:
                        break

                # Edge walls (leftmost and rightmost columns) after turn 6
                if turn_count >= 6 and (col == 0 or col == n - 2):
                    is_probable = True

                probable[wall_idx] = 1 if is_probable else 0
                wall_idx += 1

        # Vertical walls
        for row in range(n - 1):
            for col in range(n - 1):
                is_probable = False

                # Near pawns (within 2 squares)
                if abs(row - p1_row) <= 2 and abs(col - p1_col) <= 2:
                    is_probable = True
                if abs(row - p2_row) <= 2 and abs(col - p2_col) <= 2:
                    is_probable = True

                # Near existing walls (within 1 square)
                az_row = row * 2
                az_col = col * 2 + 1
                for dr in [-2, 0, 2]:
                    for dc in [-2, 0, 2]:
                        check_row = az_row + dr
                        check_col = az_col + dc
                        if 0 <= check_row < 2*n-1 and 0 <= check_col < 2*n-1:
                            if walls[check_row][check_col] > 0:
                                is_probable = True
                                break
                    if is_probable:
                        break

                # Edge walls (top and bottom rows) after turn 6
                if turn_count >= 6 and (row == 0 or row == n - 2):
                    is_probable = True

                probable[wall_idx] = 1 if is_probable else 0
                wall_idx += 1

        return probable

    def _get_valid_moves_fast(self, az_board, player):
        """Get valid moves with fast_mcts optimizations.

        Uses caching to avoid recomputing valid moves for the same board state.
        NOTE: The 'probable walls' heuristic has been disabled because it was
        hurting play quality.
        """
        cache_key = (az_board.tobytes(), player)

        if cache_key in self.valid_moves_cache:
            self.cache_hits += 1
            return self.valid_moves_cache[cache_key]

        self.cache_misses += 1

        from quoridor.QuoridorLogic import Board as AlphaZeroBoard
        temp_board = AlphaZeroBoard(self.board_size)
        temp_board.pieces = az_board.copy()

        # Get & cache legal moves. With no heuristics
        valid_moves = temp_board.get_legal_moves(player)
        valid_moves = np.array(valid_moves) # The neural network expects numpy arrays, not lists
        self.valid_moves_cache[cache_key] = valid_moves

        # Limit cache size to prevent memory issues
        if len(self.valid_moves_cache) > 15000:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.valid_moves_cache.keys())[:1000]
            for key in keys_to_remove:
                del self.valid_moves_cache[key]

        return valid_moves
