import math
import numpy as np
EPS = 1e-8

class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args, heuristic_prior_fn=None, iteration=0):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

        self.sH = {}        # state history, to detect cycle in search()
        self.eval_mcts_in_batch = getattr(args, 'eval_mcts_in_batch', 1)

        # Optional heuristic prior: callable(canonicalBoard, valids) -> action index or None.
        # Blended into the root prior for both players alongside Dirichlet noise:
        #   P = (1 - ε - α_iter) · P_NN  +  ε · Dir  +  α_iter · P_heuristic
        self.heuristic_prior_fn = heuristic_prior_fn
        self.iteration = iteration

    def getActionProb(self, canonicalBoard, temp=1, add_dirichlet_noise=False):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Args:
            canonicalBoard: current board state
            temp: temperature parameter for action selection
            add_dirichlet_noise: if True, add Dirichlet noise to root node for exploration (used in self-play)

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        # Modify the root prior for exploration during self-play.
        if add_dirichlet_noise:
            s = self.game.stringRepresentation(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)

            if s not in self.Ps:
                self.Ps[s], _ = self.nnet.predict(canonicalBoard, valids)
                self.Ps[s] = self.Ps[s] * valids
                sum_Ps = np.sum(self.Ps[s])
                if sum_Ps > 0:
                    self.Ps[s] /= sum_Ps
                else:
                    self.Ps[s] = valids / np.sum(valids)
                self.Vs[s] = valids
                self.Ns[s] = 0

            dirichlet_alpha   = getattr(self.args, 'dirichlet_alpha',        0.3)
            dirichlet_epsilon = getattr(self.args, 'dirichlet_epsilon',      0.25)
            h_alpha_start     = getattr(self.args, 'heuristic_alpha',        0.0)
            h_decay_iters     = getattr(self.args, 'heuristic_decay_iters', 150)

            # Linear decay
            heuristic_alpha = h_alpha_start * max(0.0, 1.0 - self.iteration / h_decay_iters)

            noise = np.random.dirichlet([dirichlet_alpha] * int(np.sum(valids)))
            noise_full = np.zeros(len(valids))
            noise_full[valids == 1] = noise

            # Heuristic prior applied to both players: one-hot on JS bot's suggested action.
            # P = (1 - ε - α)·P_NN + ε·Dir + α·P_heuristic
            # If the heuristic returns None (JS bot failure), fall back to α=0 so the policy stays normalised
            effective_heuristic_alpha = 0.0
            heuristic_prior = np.zeros(len(valids))
            if self.heuristic_prior_fn is not None and heuristic_alpha > 0.0:
                try:
                    h_action = self.heuristic_prior_fn(canonicalBoard, valids)
                    if h_action is not None and valids[h_action]:
                        heuristic_prior[h_action] = 1.0
                        effective_heuristic_alpha = heuristic_alpha
                except Exception:
                    pass   # silently fall back to α=0 for this step

            nn_weight = 1.0 - dirichlet_epsilon - effective_heuristic_alpha
            self.Ps[s] = (nn_weight * self.Ps[s]
                          + dirichlet_epsilon * noise_full
                          + effective_heuristic_alpha * heuristic_prior)

        if self.eval_mcts_in_batch == 1:
            # Standard MCTS - no batching
            for i in range(self.args.numMCTSSims):
                self.search(canonicalBoard)
        else:
            # Batch MCTS - collect leaves and evaluate in batches
            for batch_start in range(0, self.args.numMCTSSims, self.eval_mcts_in_batch):
                batch_end = min(batch_start + self.eval_mcts_in_batch, self.args.numMCTSSims)
                self._search_batch(canonicalBoard, batch_end - batch_start)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s,a)] if (s,a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

        if np.sum(counts) == 0:
            # No actions were explored (likely due to cycles or early termination)
            # Return uniform distribution over valid moves
            # print("Warning: MCTS failed to explore any actions, returning uniform over valid moves")
            valids = self.game.getValidMoves(canonicalBoard, 1)
            return valids / np.sum(valids)

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs


    def search(self, canonicalBoard, counter=0):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """
        if counter == 0:
            self.sH = {}
        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s]!=0:
            # terminal node
            return -self.Es[s]
        if s not in self.Ps:
            # leaf node
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s], v = self.nnet.predict(canonicalBoard, valids)
            self.Ps[s] = self.Ps[s]*valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s    # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s,a)] + self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s])/(1+self.Nsa[(s,a)])
                else:
                    u = self.args.cpuct*self.Ps[s][a]*math.sqrt(self.Ns[s] + EPS)     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        if s in self.sH or counter > 256: # cycle
            return 0
        self.sH[s] = 1

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s, counter+1)
        if v == 0:
            self.Ns[s] -= 1 if self.Ns[s] > 0 else 0
            return 0

        if (s,a) in self.Qsa:
            self.Qsa[(s,a)] = (self.Nsa[(s,a)]*self.Qsa[(s,a)] + v)/(self.Nsa[(s,a)]+1)
            self.Nsa[(s,a)] += 1

        else:
            self.Qsa[(s,a)] = v
            self.Nsa[(s,a)] = 1

        self.Ns[s] += 1
        return -v

    def _search_batch(self, canonicalBoard, num_sims):
        """
        Perform multiple MCTS simulations with batched leaf evaluation.

        This collects leaf nodes from multiple simulations, evaluates them
        in a single batch, then completes the backpropagation.

        Args:
            canonicalBoard: The root board state
            num_sims: Number of simulations to run in this batch
        """
        # Collect leaf nodes from simulations
        leaf_data = []  # List of (board, valids, path, string_repr)

        for _ in range(num_sims):
            leaf_info = self._search_to_leaf(canonicalBoard)
            if leaf_info is not None:
                leaf_data.append(leaf_info)

        if not leaf_data:
            return

        # Batch evaluate all leaves
        boards = [data[0] for data in leaf_data]
        valids_list = [data[1] for data in leaf_data]
        paths = [data[2] for data in leaf_data]
        strings = [data[3] for data in leaf_data]

        # Stack boards into batch and get predictions
        batch_boards = np.stack(boards, axis=0)
        policies, values = self.nnet.predict(batch_boards, batch_valids=valids_list)

        # Process each leaf and backpropagate
        for pi, v, valids, s, path in zip(policies, values, valids_list, strings, paths):
            # Mask invalid moves and normalize
            pi = pi * valids
            sum_pi = np.sum(pi)
            if sum_pi > 0:
                pi /= sum_pi
            else:
                # All moves masked - make all valid moves equally probable
                pi = valids / np.sum(valids)

            # Store policy and valid moves
            self.Ps[s] = pi
            self.Vs[s] = valids
            self.Ns[s] = 0

            # Backpropagate value
            self._backprop_batch(path, -float(np.squeeze(v)))

    def _search_to_leaf(self, canonicalBoard):
        """
        Search from root to a leaf node, tracking the path.
        Returns (board, valids, path, string_repr) if leaf found, None if terminal/cycle.

        This is similar to search() but doesn't evaluate leaves - just finds them.
        """
        path = []
        sH_local = {}
        board = canonicalBoard
        counter = 0

        while True:
            s = self.game.stringRepresentation(board)

            # Check terminal
            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(board, 1)
            if self.Es[s] != 0:
                # Terminal node - backprop immediately
                self._backprop_batch(path, -self.Es[s])
                return None

            # Check if leaf
            if s not in self.Ps:
                # Leaf node - return it for batch evaluation
                valids = self.game.getValidMoves(board, 1)
                return (board, valids, path, s)

            # Internal node - select action and continue
            valids = self.Vs[s]
            best_a = self._select_action_PUCT(s, valids)

            # Cycle detection
            if s in sH_local or counter > 256:
                # Don't pollute tree with value=0 - just stop this simulation
                return None
            sH_local[s] = 1
            counter += 1

            # Take action
            next_board, next_player = self.game.getNextState(board, 1, best_a)
            next_board = self.game.getCanonicalForm(next_board, next_player)
            path.append((s, best_a))
            board = next_board

    def _select_action_PUCT(self, s, valids):
        """Select best action using PUCT formula."""
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        return best_act

    def _backprop_batch(self, path, v):
        """Backpropagate value along path."""
        value = v
        for (s, a) in reversed(path):
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + value) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1
            else:
                self.Qsa[(s, a)] = value
                self.Nsa[(s, a)] = 1

            self.Ns[s] = self.Ns.get(s, 0) + 1
            value = -value
