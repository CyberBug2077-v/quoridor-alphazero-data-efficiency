import numpy as np
from progress.bar import Bar
from quoridor.pytorch.NNet import AverageMeter
import time
from utils import format_duration
#import os

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, display=None, on_game_start=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.
            on_game_start: optional callback function called before each game.
                          Used to reset MCTS trees for deterministic evaluation.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.on_game_start = on_game_start
        self._last_game_signature = None

    def playGame(self, verbose=False, max_iters=1000):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        if self.on_game_start:
            self.on_game_start()

        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        moves = []
        it = 0
        self._last_game_signature = None
        while self.game.getGameEnded(board, curPlayer)==0 and it<max_iters:
            it+=1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                #self.display(board)
                if players[curPlayer+1].__name__ != '<lambda>': # new
                    #os.system('clear')
                    self.display(self.game.getCanonicalForm(board, curPlayer), curPlayer)

            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))
            moves.append(action)
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)
            if valids[action]==0:
                print("invalid action", action)
                self._last_game_signature = tuple(moves)
                return -curPlayer
                #print ""
                #print(action)
                #return 0
                assert valids[action] >0
            #if verbose:
            #    print("Action index ", str(action))
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            if verbose and players[curPlayer+1].__name__ == '<lambda>': # new
                #os.system('clear')
                self.display(self.game.getCanonicalForm(board, -curPlayer), -curPlayer)

        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            if verbose and players[curPlayer+1].__name__ == '<lambda>':
                self.display(self.game.getCanonicalForm(board, -curPlayer), -curPlayer)
            else:
                self.display(self.game.getCanonicalForm(board, curPlayer), curPlayer)
                # self.display(self.game.getCanonicalForm(board, -curPlayer), -curPlayer)
            #self.display(board)

        self._last_game_signature = tuple(moves)
        return self.game.getGameEnded(board, 1), it

    def playGames(self, num, verbose=False, print_summary=True, track_unique=False, max_iters=1000):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        start_time = time.time()
        end = time.time()
        eps = 0
        maxeps = int(num)
        unique_games = set() if track_unique else None

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        total_moves = 0
        for _ in range(num):
            gameResult, moves = self.playGame(verbose=verbose, max_iters=max_iters)
            total_moves += moves
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
            if unique_games is not None and self._last_game_signature is not None:
                unique_games.add(self._last_game_signature)
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            gameResult, moves = self.playGame(verbose=verbose, max_iters=max_iters)
            total_moves += moves
            if gameResult==-1:
                oneWon+=1
            elif gameResult==1:
                twoWon+=1
            else:
                draws+=1
            if unique_games is not None and self._last_game_signature is not None:
                unique_games.add(self._last_game_signature)
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=num, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        bar.finish()

        if print_summary:
            total_time = time.time() - start_time
            avg_moves = total_moves / maxeps if maxeps > 0 else 0
            unique_suffix = ""
            if unique_games is not None:
                unique_count = len(unique_games)
                if unique_count < maxeps:
                    unique_suffix = f" | Unique games: {unique_count}/{maxeps}"
            print(f"[Arena] Games: {maxeps} | Total Time: {format_duration(total_time)} | Avg Time: {format_duration(eps_time.avg)}/game | Avg Moves: {avg_moves:.1f} | Results: New={twoWon} Prev={oneWon} Draws={draws}{unique_suffix}")

        return oneWon, twoWon, draws

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
