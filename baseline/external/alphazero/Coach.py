from collections import deque
from Arena import Arena
from MCTS import MCTS
import numpy as np
from progress.bar import Bar
from quoridor.pytorch.NNet import AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
from utils import log_ram_usage, format_duration
import torch
import pickle


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args, iteration_callback=None):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.iteration_callback = iteration_callback
        self.save_every_n_iterations = getattr(args, "save_every_n_iterations", 10)
        self.mcts = self._make_mcts(self.nnet)
        self.trainExamplesHistory = []    # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False # can be overriden in loadTrainExamples()
        self.current_iteration = 0
        os.makedirs(self.args.checkpoint, exist_ok=True)

        # JS bot for heuristic prior training
        self._js_bot = None
        self._js_bot_failures = 0
        self._expert_examples = self._load_expert_examples()

    def _make_mcts(self, nnet, iteration=0):
        heuristic_prior_fn = self._js_heuristic_fn() if getattr(self.args, 'heuristic_alpha', 0.0) > 0.0 else None
        return MCTS(self.game, nnet, self.args, heuristic_prior_fn=heuristic_prior_fn, iteration=iteration)

    def _load_expert_examples(self):
        expert_path = getattr(self.args, 'expert_examples_data', None)
        if not expert_path:
            return []

        if not os.path.isfile(expert_path):
            print(f"[Coach] Expert example file not found: {expert_path}")
            return []

        try:
            with open(expert_path, "rb") as f:
                loaded = pickle.load(f)
        except Exception as exc:
            print(f"[Coach] Failed to load expert examples from {expert_path}: {exc}")
            return []

        if isinstance(loaded, dict) and 'examples' in loaded:
            loaded = loaded['examples']

        if not isinstance(loaded, list):
            loaded = list(loaded)

        print(f"[Coach] Loaded {len(loaded):,} expert examples for early-training top-up.")
        return loaded

    def _sample_expert_examples(self, count):
        if count <= 0 or not self._expert_examples:
            return []

        shuffled = list(self._expert_examples)
        shuffle(shuffled)
        if count >= len(shuffled):
            return shuffled
        return shuffled[:count]

    def _build_training_examples(self):
        train_examples = []
        for examples in self.trainExamplesHistory:
            train_examples.extend(examples)
        shuffle(train_examples)

        max_train_size = getattr(self.args, 'max_train_size', None)
        fill_with_expert_data = bool(getattr(self.args, 'fill_with_expert_data', False))

        expert_examples_used = 0
        if fill_with_expert_data and max_train_size and self._expert_examples:
            if len(train_examples) < max_train_size:
                expert_examples = self._sample_expert_examples(
                    max_train_size - len(train_examples)
                )
                expert_examples_used = len(expert_examples)
                train_examples.extend(expert_examples)
                shuffle(train_examples)

        available_examples = len(train_examples)
        if max_train_size and len(train_examples) > max_train_size:
            shuffle(train_examples)
            train_examples = train_examples[:max_train_size]

        return train_examples, expert_examples_used, available_examples

    def _get_js_bot(self):
        """Return a cached JSBot subprocess, creating it on first call."""
        if self._js_bot is not None:
            return self._js_bot
        try:
            import pathlib
            project_root = os.environ.get(
                "PROJECT_ROOT",
                str(pathlib.Path(__file__).resolve().parent.parent.parent)
            )
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            js_entry = str(pathlib.Path(project_root) / "external" / "js-mcts" / "bot.js")

            from arena.bot_js_mcts import JSBot

            rollouts = self.args.get('heuristic_rollouts', 200)
            # Always play as 'white' in canonical form (canonical board is always from player-1's POV)
            self._js_bot = JSBot('white', js_entry, rollouts=rollouts, greedy_prob=0.7)
        except Exception as exc:
            print(f"[Coach] Could not initialise JS bot: {exc}")
            self._js_bot = None
        return self._js_bot

    def _js_heuristic_fn(self):
        """Return a closure for MCTS heuristic_prior_fn: (canonicalBoard, valids) -> action index or None."""
        def heuristic(canonical_board, valids):
            try:
                import pathlib
                project_root = os.environ.get(
                    "PROJECT_ROOT",
                    str(pathlib.Path(__file__).resolve().parent.parent.parent)
                )
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                from quoridor.board_converter import az_to_pyquoridor, count_fences_remaining
                from arena.bot_interface import PawnMove, FenceMove
                from quoridor.QuoridorLogic import Board as AZBoard

                fw, fb = count_fences_remaining(canonical_board, self.game.n)
                pyq_board = az_to_pyquoridor(canonical_board, self.game.n,
                                             fences_left_white=fw,
                                             fences_left_black=fb)

                bot = self._get_js_bot()
                if bot is None:
                    return None

                move = bot.select_move(pyq_board)

                n = self.game.n
                az_b = AZBoard(n)
                az_b.pieces = canonical_board.copy()

                if isinstance(move, PawnMove):
                    pq_row, pq_col = move.target
                    az_row = (n - 1 - pq_row) * 2
                    az_col = pq_col * 2
                    for a in range(8):
                        res = az_b.action_from_index(a, 1)
                        if res is not None:
                            _, (rx, ry) = res
                            if int(rx) == az_row and int(ry) == az_col and valids[a]:
                                return a

                elif isinstance(move, FenceMove):
                    pq_row, pq_col = move.anchor
                    dim = 2 * n - 1
                    if move.orientation.lower() == 'h':
                        az_row = dim - 2 - pq_row * 2
                        az_col = pq_col * 2
                        action = az_b.index_of_action(8, az_row, az_col)
                    else:
                        az_row = (n - 1 - pq_row) * 2
                        az_col = pq_col * 2 + 1
                        action = az_b.index_of_action(9, az_row, az_col)
                    if action is not None and 0 <= action < len(valids) and valids[action]:
                        return action

            except Exception as exc:
                self._js_bot_failures += 1
                if self._js_bot_failures <= 10 or self._js_bot_failures % 100 == 0:
                    print(f"[Coach] heuristic_fn failed (#{self._js_bot_failures}): {exc}")
                try:
                    if self._js_bot is not None:
                        self._js_bot.__del__()
                except Exception:
                    pass
                self._js_bot = None

            return None

        return heuristic

    def executeEpisode(self):
        """Run one self-play episode. If heuristic_alpha > 0, blends JS-bot prior into MCTS root
        as P = (1-ε-α)·P_NN + ε·Dir + α·P_heuristic, with α decaying linearly to 0."""
        game_history = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        self._episode_illegal_action_count = 0
        max_game_length = self.args.get('max_game_length', 150)

        while episodeStep < max_game_length:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            valids = self.game.getValidMoves(canonicalBoard, 1)

            temp = int(episodeStep < self.args.tempThreshold)
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp, add_dirichlet_noise=True)
            if np.sum(pi) == 0:
                # adopt uniform over legal moves to avoid aborting game
                valid_sum = np.sum(valids)
                if valid_sum == 0:
                    print("No valid moves available; terminating episode with draw.")
                    break
                # print("Warning: MCTS produced zero policy; using uniform over valid moves.")
                pi = valids / valid_sum
            action = np.random.choice(len(pi), p=pi)
            if not valids[action]:
                self._episode_illegal_action_count += 1
                raise RuntimeError(
                    f"Self-play selected illegal action {action} at step {episodeStep}"
                )

            game_history.append((canonicalBoard, self.curPlayer, pi, valids, episodeStep))
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                if not game_history:
                    return []
                return [
                    (b, pi, r * player, valids, step, episodeStep)
                    for (b, player, pi, valids, step) in game_history
                ]

        if not game_history:
            print("Episode ended without moves.")
            return []
        # print("the game's not ended; treating as draw.")
        return [
            (b, pi, 0, valids, step, episodeStep)
            for (b, player, pi, valids, step) in game_history
        ]

    def _clear_mcts(self, mcts):
        """Clear all MCTS state between arena games for deterministic evaluation."""
        tree = mcts
        tree.Qsa = {}
        tree.Nsa = {}
        tree.Ns  = {}
        tree.Ps  = {}
        tree.Es  = {}
        tree.Vs  = {}
        if hasattr(tree, 'sH'):
            tree.sH = {}

    def _make_on_game_start(self, pmcts, nmcts, move_counter):
        """Return an on_game_start hook that resets MCTS and move counters."""
        def on_game_start():
            self._clear_mcts(pmcts)
            self._clear_mcts(nmcts)
            move_counter[0] = 0
        return on_game_start

    def _make_arena_player(self, mcts, move_counter, arena_temp, arena_temp_threshold):
        """Create a player function with temperature-based move selection."""
        def player(x):
            move_counter[0] += 1
            # Use temperature for first N moves, then deterministic.
            if move_counter[0] <= arena_temp_threshold * 2:  # *2 because both players increment
                temp = arena_temp
            else:
                temp = 0
            probs = mcts.getActionProb(x, temp=temp)
            if temp == 0:
                return np.argmax(probs)
            return np.random.choice(len(probs), p=probs)
        return player

    def _current_lr(self, iteration):
        base_lr = getattr(self.args, 'lr', 0.001)
        decay_step = getattr(self.args, 'lr_decay_step', None)
        decay_factor = getattr(self.args, 'lr_decay_factor', None)
        max_decays = getattr(self.args, 'lr_max_decays', None)

        if decay_step is None or decay_step <= 0 or decay_factor is None:
            return base_lr

        num_decays = max(0, (iteration - 1) // decay_step)
        if max_decays is not None:
            num_decays = min(num_decays, max_decays)

        current_lr = base_lr * (decay_factor ** num_decays)
        return max(current_lr, base_lr * 1e-3)

    def learn(self):
        """
        Main training loop. Each iteration:
          1. Self-play: collect numEps games using MCTS (with optional heuristic prior).
          2. Train: fit the network on the rolling example history.
          3. Evaluate: pit new vs old network in arena; keep whichever wins more. (Disabled)
        LR decays by lr_decay_factor every lr_decay_step iterations, floored at lr * 1e-3.
        The first drop happens after a full decay window completes.
        """

        start_iter = self.current_iteration + 1
        if start_iter > 1:
            print(f"\n{'='*60}")
            print(f"RESUMING TRAINING FROM ITERATION {start_iter}")
            print(f"{'='*60}\n")

        for i in range(start_iter, self.args.numIters+1):
            # bookkeeping
            print('-----------ITER ' + str(i) + '-----------')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            iteration_start = time.perf_counter()
            self_play_start = iteration_start
            games_completed = 0
            game_lengths = []
            positions_generated = 0
            illegal_action_count = 0
            self_play_seconds = 0.0
            training_seconds = 0.0
            checkpoint_path = None

            log_resources = getattr(self.args, 'print_summary', True) and (i == start_iter or i % 10 == 0)
            # examples of the iteration
            if not self.skipFirstSelfPlay or i>1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()
                total_moves = 0

                for eps in range(self.args.numEps):
                    self.mcts = self._make_mcts(self.nnet, iteration=i)   # reset search tree
                    episode_examples = self.executeEpisode()
                    illegal_action_count += getattr(
                        self,
                        '_episode_illegal_action_count',
                        0,
                    )
                    iterationTrainExamples.extend(episode_examples)
                    if not episode_examples:
                        continue

                    games_completed += 1
                    game_length = len(episode_examples)
                    game_lengths.append(game_length)
                    positions_generated += game_length

                    # bookkeeping + plot progress
                    total_moves += game_length
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                                                                                                               total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self_play_seconds = time.perf_counter() - self_play_start

                if getattr(self.args, 'print_summary', True):
                    mean_moves = total_moves / games_completed if games_completed else 0.0
                    print(f"[Self-Play] Games: {games_completed} | Total Time: {bar.elapsed_td} | Avg Time: {format_duration(eps_time.avg)} | Avg Moves: {mean_moves:.1f}")
                sys.stdout.flush()


                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)
                if iterationTrainExamples:
                    trainStats = [0, 0, 0]
                    for res in iterationTrainExamples:
                        value = res[2]
                        if value > 0:
                            trainStats[0] += 1
                        elif value < 0:
                            trainStats[1] += 1
                        else:
                            trainStats[2] += 1
                    print(trainStats)
                else:
                    print("No completed games this iteration.")
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                # print("len(trainExamplesHistory) =", len(self.trainExamplesHistory), " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            replay_buffer_size = sum(
                len(examples) for examples in self.trainExamplesHistory
            )
            trainExamples, expert_examples_used, available_examples = self._build_training_examples()

            if not trainExamples:
                print("Skipping training – no examples available yet.")
            if expert_examples_used > 0:
                self_play_examples = len(trainExamples) - expert_examples_used
                print(
                    f"[Training Data] Self-play: {self_play_examples:,} | "
                    f"Expert top-up: {expert_examples_used:,} | Total: {len(trainExamples):,}"
                )
            else:
                print(f"[Training Data] Self-play only: {len(trainExamples):,}")
            
            # Snapshot current network as the baseline for arena evaluation
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = self._make_mcts(self.pnet, iteration=i)

            current_lr = self._current_lr(i)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            training_start = time.perf_counter()
            train_metrics = self.nnet.train(
                trainExamples,
                print_summary=getattr(self.args, 'print_summary', True),
                lr_override=current_lr,
                available_examples=available_examples,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            training_seconds = time.perf_counter() - training_start

            if train_metrics["optimizer_steps"] <= 0:
                raise RuntimeError(
                    f"Iteration {i} completed without any optimizer step. "
                    f"available_examples={train_metrics['available_examples']}, "
                    f"examples_used={train_metrics['examples_used']}, "
                    f"batch_size={self.args.batch_size}"
                )
            if log_resources:
                log_ram_usage(label=f"Iter {i} post-train", include_peak=True)

            nmcts = self._make_mcts(self.nnet, iteration=i)

            print('PITTING AGAINST PREVIOUS VERSION')

            # Arena evaluation settings
            arena_temp = getattr(self.args, 'arena_temp', 0.3)
            arena_temp_threshold = getattr(self.args, 'arena_temp_threshold', 5)
            move_counter = [0]  # Use list to allow modification in nested function

            on_game_start = self._make_on_game_start(pmcts, nmcts, move_counter)
            arena = Arena(self._make_arena_player(pmcts, move_counter, arena_temp, arena_temp_threshold),
                          self._make_arena_player(nmcts, move_counter, arena_temp, arena_temp_threshold),
                          self.game,
                          on_game_start=on_game_start)
            pwins, nwins, draws = arena.playGames(
                self.args.arenaCompare,
                print_summary=getattr(self.args, 'print_summary', True),
                track_unique=True,
                max_iters=self.args.get('max_game_length', 150),
            )

            if self.args.arenaCompare == 0 or self.args.updateThreshold <= 0:
                # Skip evaluation entirely - always accept new model
                print('ACCEPTING NEW MODEL (evaluation disabled)')
                accepted_model = True
            elif pwins + nwins == 0:
                print('REJECTING NEW MODEL (all draws)')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                accepted_model = False
            elif float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
                accepted_model = False
            else:
                print('ACCEPTING NEW MODEL')
                accepted_model = True

            # Save numbered checkpoints at the configured interval.
            # Always persist the invocation boundary so a deliberate
            # --stop-after-iteration can be resumed even when it falls between
            # the normal checkpoint cadence.
            should_save = (
                i % self.save_every_n_iterations == 0
                or i == self.args.numIters
            )

            if should_save:
                checkpoint_name = f"checkpoint_{i}.pth.tar"
                temporary_checkpoint_name = checkpoint_name + ".tmp"
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint,
                    filename=temporary_checkpoint_name,
                )
                temporary_checkpoint_path = os.path.join(
                    self.args.checkpoint, temporary_checkpoint_name
                )
                with open(temporary_checkpoint_path, 'rb+') as checkpoint_file:
                    checkpoint_file.flush()
                    os.fsync(checkpoint_file.fileno())
                os.replace(
                    temporary_checkpoint_path,
                    os.path.join(self.args.checkpoint, checkpoint_name),
                )
                checkpoint_path = os.path.join(
                    self.args.checkpoint,
                    checkpoint_name,
                ).replace('\\', '/')

            # Commit the replay state only after the numbered model checkpoint.
            self.saveTrainExamples(i)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                peak_gpu_memory_mb = None
            iteration_seconds = time.perf_counter() - iteration_start
            mean_game_length = (
                sum(game_lengths) / len(game_lengths) if game_lengths else None
            )

            iteration_metrics = {
                "schema_version": 1,
                "iteration": i,
                "games_completed": games_completed,
                "positions_generated": positions_generated,
                "illegal_action_count": illegal_action_count,
                "mean_game_length": mean_game_length,
                "min_game_length": min(game_lengths) if game_lengths else None,
                "max_game_length": max(game_lengths) if game_lengths else None,
                "replay_buffer_size": replay_buffer_size,
                "available_examples": train_metrics["available_examples"],
                "examples_used": train_metrics["examples_used"],
                "samples_seen": train_metrics["samples_seen"],
                "training_batches": train_metrics["training_batches"],
                "optimizer_steps": train_metrics["optimizer_steps"],
                "effective_batch_size": train_metrics["effective_batch_size"],
                "micro_batch_size": train_metrics["micro_batch_size"],
                "gradient_accumulation_steps": train_metrics["gradient_accumulation_steps"],
                "micro_batches_processed": train_metrics["micro_batches_processed"],
                "policy_loss": train_metrics["policy_loss"],
                "value_loss": train_metrics["value_loss"],
                "total_loss": train_metrics["total_loss"],
                "mean_grad_norm": train_metrics["mean_grad_norm"],
                "max_grad_norm": train_metrics["max_grad_norm"],
                "self_play_seconds": self_play_seconds,
                "training_seconds": training_seconds,
                "iteration_seconds": iteration_seconds,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "checkpoint_path": checkpoint_path,
            }
            if self.iteration_callback is not None:
                self.iteration_callback(iteration_metrics)

            # Update the mutable pointer only after checkpoint, examples, and
            # metrics have committed the iteration boundary.
            if accepted_model:
                temporary_best = 'best.pth.tar.tmp'
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint,
                    filename=temporary_best,
                )
                temporary_best_path = os.path.join(
                    self.args.checkpoint, temporary_best
                )
                with open(temporary_best_path, 'rb+') as best_file:
                    best_file.flush()
                    os.fsync(best_file.fileno())
                os.replace(
                    temporary_best_path,
                    os.path.join(self.args.checkpoint, 'best.pth.tar'),
                )

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)

        data_to_save = {'iteration': iteration,
                        'examples': self.trainExamplesHistory}

        filename = os.path.join(folder, "latest.examples")
        temporary_filename = filename + ".tmp"
        with open(temporary_filename, "wb+") as f:
            Pickler(f).dump(data_to_save)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_filename, filename)
        # print(f"Saved examples: latest.examples (iter {iteration})")

    def loadTrainExamples(self):
        if hasattr(self.args, 'load_folder_examples_file'):
            folder = self.args.load_folder_examples_file[0]
            filename = self.args.load_folder_examples_file[1]
        else:
            folder = self.args.checkpoint
            filename = "latest.examples"
        examplesFile = os.path.join(folder, filename)

        if not os.path.isfile(examplesFile):
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print(f"Loading examples from: {examplesFile}")
            with open(examplesFile, "rb") as f:
                loaded_data = Unpickler(f).load()
            f.closed

            # Handle both old format and new format (list vs dict with iteration)
            if isinstance(loaded_data, dict) and 'iteration' in loaded_data:
                self.current_iteration = loaded_data['iteration']
                self.trainExamplesHistory = loaded_data['examples']
                print(f"Loaded {len(self.trainExamplesHistory)} iterations of examples from iteration {self.current_iteration}")
            else:
                self.trainExamplesHistory = loaded_data
                print(f"Loaded {len(self.trainExamplesHistory)} iterations of examples (old format, starting from iteration 0)")

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
