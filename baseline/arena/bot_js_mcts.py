import json
import os
import subprocess
from typing import Optional

from .bot_interface import BotInterface, BotMove, FenceMove, PawnMove, PlayerColor
from pyquoridor.board import Board
from .utils import board_to_json


class JSBot(BotInterface):
    def __init__(self, color: PlayerColor, js_entry: str, cwd: Optional[str] = None,
                 rollouts: int = 20000, trace_mcts: bool = False, greedy_prob: float = 0.7,
                 log_stderr: Optional[bool] = None):
        super().__init__(color)
        self.rollouts = rollouts
        self.greedy_prob = greedy_prob
        self.js_entry = os.path.abspath(js_entry)
        self.cwd = os.path.abspath(cwd) if cwd else os.path.dirname(self.js_entry)
        self.trace_mcts = trace_mcts
        self.last_root_trace = None
        if log_stderr is None:
            env_log = os.environ.get("JSBOT_LOG_STDERR")
            if env_log is None:
                log_stderr = False
            else:
                log_stderr = env_log.lower() in ("1", "true", "yes", "y")
        stderr_target = None if log_stderr else subprocess.DEVNULL

        try:
            self.proc = subprocess.Popen(
                ["node", self.js_entry],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_target,
                text=True,
                cwd=self.cwd,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Make sure that Node.js is installed, and the path to the bot is correctly set."
            ) from exc

    def select_move(self, board: Board) -> BotMove:
        payload = {
            "color": self.color,
            "state": board_to_json(board),
            "rollouts": self.rollouts,
            "trace": self.trace_mcts,
            "greedyProb": self.greedy_prob,
        }
        self._writeln(payload)
        resp = self._readln()
        if not isinstance(resp, dict):
            raise RuntimeError(f"Bad response from JS bot: {resp!r}")
        if "error" in resp:
            raise RuntimeError(f"JS bot error: {resp['error']}")
        self.last_root_trace = None
        if "move" in resp:
            self.last_root_trace = resp.get("trace")
            resp = resp["move"]
        if "type" not in resp:
            raise RuntimeError(f"Bad response from JS bot: {resp}")

        move_type = resp["type"]
        if move_type == "pawn":
            row, col = int(resp["row"]), int(resp["col"])
            return PawnMove(self.color, (row, col))
        if move_type == "fence":
            row, col = int(resp["row"]), int(resp["col"])
            orientation = resp["orientation"].lower()
            return FenceMove(self.color, (row, col), orientation)
        raise ValueError(f"Unknown move type from JS bot: {move_type}")

    # --- Helpers ---
    def _writeln(self, obj):
        line = json.dumps(obj) + "\n"
        if not self.proc.stdin:
            raise RuntimeError("JS bot process stdin unavailable")
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

    def _readln(self):
        if not self.proc.stdout:
            raise RuntimeError("JS bot process stdout unavailable")
        line = self.proc.stdout.readline()
        return json.loads(line) if line else {}

    def __del__(self):
        try:
            if getattr(self, "proc", None) and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass
