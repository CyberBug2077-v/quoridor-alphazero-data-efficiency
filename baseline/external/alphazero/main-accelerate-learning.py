import argparse
import os
import sys

from Coach import Coach
from quoridor.QuoridorGame import QuoridorGame as Game
from quoridor.pytorch.NNet import NNetWrapper as nn
from quoridor.pytorch import NNet
from utils import dotdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name",          type=str,   required=True)
    p.add_argument("--mcts_sims",             type=int,   default=200)
    p.add_argument("--train_history_iters",   type=int,   default=40,
                   help="Replay window: number of past self-play iterations kept in the pool")
    p.add_argument("--max_train_size",        type=int,   default=200_000,
                   help="Per-iteration cap on training examples (random subsample of the pool)")
    p.add_argument("--max_game_length",       type=int,   default=120,
                   help="Maximum moves per self-play game before it is scored as a draw")
    p.add_argument("--num_eps",               type=int,   default=75,
                   help="Self-play games per iteration")
    p.add_argument("--lr",                    type=float, default=2e-4,
                   help="Base learning rate")
    p.add_argument("--lr_decay_step",         type=int,   default=100,
                   help="Decay interval in iterations; <=0 disables decay")
    p.add_argument("--lr_decay_factor",       type=float, default=0.0,
                   help="Multiplicative LR decay factor applied every lr_decay_step iterations")
    p.add_argument("--lr_max_decays",         type=int,   default=None,
                   help="Maximum number of LR drops to apply (unset = unlimited)")
    p.add_argument("--load_pretrain_data",    type=str,   default=None)
    p.add_argument("--expert_examples_data",  type=str,   default=None,
                   help="Optional expert dataset used to top up early training when self-play is still sparse")
    p.add_argument("--fill_with_expert_data", type=int,   default=0,
                   help="If set, fill the training pool with expert examples up to max_train_size until self-play reaches that size")
    p.add_argument("--heuristic_alpha",       type=float, default=0.0,
                   help="Starting weight of JS bot prior blended into MCTS root (0=disabled)")
    p.add_argument("--heuristic_decay_iters", type=int,   default=150,
                   help="Linearly decay heuristic_alpha to 0 over this many iterations")
    p.add_argument("--heuristic_rollouts",    type=int,   default=200,
                   help="JS bot rollouts per move when used as heuristic prior")
    p.add_argument("--pretrain_epochs",       type=int,   default=10,
                   help="Epochs for supervised pretraining phase (ignored if no pretrain data)")
    p.add_argument("--dry_run",           action="store_true",
                   help="Run a tiny 1-iteration training for laptop validation")
    p.add_argument("--dry_run_minutes",   type=int, default=5,
                   help="Target duration for --dry_run; used for display only")
    p.add_argument("--use_amp",           type=int,   default=None)
    p.add_argument("--amp_dtype",         type=str,   default=None,
                   choices=["fp16", "bf16"])
    p.add_argument("--seed",              type=int,   default=None,
                   help="Random seed for python/numpy/torch (unset = nondeterministic)")

    # --- Architecture options, both OFF by default ---
    p.add_argument("--attn_depth",        type=int,   default=1,
                   help="Number of global self-attention layers after the residual tower")
    p.add_argument("--se_enabled",        type=int,   default=0,
                   help="Add a Squeeze-and-Excitation module to each residual block (0=off)")

    # --- GPU efficiency optimizations, OFF by default (no algorithmic change) ---
    p.add_argument("--fast_opts",         type=int,   default=0,
                   help="Enable GPU speedups: cudnn.benchmark + channels_last + torch.compile "
                        "(dynamic). No change to weights/architecture; checkpoints stay compatible.")
    return p.parse_args()


def build_train_args(cli, checkpoint_dir):
    lr_decay_step = cli.lr_decay_step if cli.lr_decay_step and cli.lr_decay_step > 0 else None
    lr_max_decays = cli.lr_max_decays if cli.lr_max_decays is not None and cli.lr_max_decays >= 0 else None
    return dotdict({
        "exp_name":               cli.exp_name,
        "load_pretrain_data":     cli.load_pretrain_data,
        "expert_examples_data":   cli.expert_examples_data,
        "fill_with_expert_data":  bool(cli.fill_with_expert_data),
        "heuristic_alpha":        cli.heuristic_alpha,
        "heuristic_decay_iters":  cli.heuristic_decay_iters,
        "heuristic_rollouts":     cli.heuristic_rollouts,

        "numIters":           1000,
        "numEps":             cli.num_eps,
        "max_game_length":    cli.max_game_length,
        "tempThreshold":      30,
        "numMCTSSims":        cli.mcts_sims,
        "eval_mcts_in_batch": 4,
        "maxlenOfQueue":      500_000,
        "max_train_size":     cli.max_train_size,

        "arenaCompare":       4,
        "updateThreshold":    -0.51,
        "arena_temp":         0.18,
        "arena_temp_threshold": 5,

        "cpuct":              1.25,
        "dirichlet_alpha":    0.15,
        "dirichlet_epsilon":  0.25,

        "lr":                 cli.lr,
        "lr_decay_step":      lr_decay_step,
        "lr_decay_factor":    cli.lr_decay_factor if lr_decay_step is not None else None,
        "lr_max_decays":      lr_max_decays,

        "seed":               cli.seed,

        "checkpoint":         checkpoint_dir,
        "save_every_n_iterations": 10,
        "load_model":         False,
        "load_examples":      False,
        "load_folder_file":   (checkpoint_dir, "best.pth.tar"),
        "load_folder_examples_file": (checkpoint_dir, "latest.examples"),
        "numItersForTrainExamplesHistory": cli.train_history_iters,
        "print_summary":      True,
    })
# TODO: Configure dropout

def build_nn_args(use_amp, amp_dtype, attn_depth=1, se_enabled=False, fast_opts=False):
    return {
        "batch_size":     2048,
        "epochs":         4,
        "num_channels":   128,
        "num_res_blocks": 6,
        "attn_depth":     attn_depth,
        "num_heads":      8,
        "se_enabled":     bool(se_enabled),
        "fast_opts":      bool(fast_opts),
        "clip":           1.0,
        "use_amp":        bool(use_amp),
        "amp_dtype":      amp_dtype,
    }


def describe_lr_schedule(cli):
    if cli.lr_decay_step is None or cli.lr_decay_step <= 0:
        return f"constant {cli.lr:g}"

    desc = f"{cli.lr:g} x {cli.lr_decay_factor:g} every {cli.lr_decay_step} iters"
    if cli.lr_max_decays is not None and cli.lr_max_decays >= 0:
        desc += f" (max {cli.lr_max_decays} drops)"
    return desc

def apply_dry_run_overrides(cli, train_args, nn_args):
    """
    Override settings for a quick dry run while keeping normal settings.
    Only reduces self-play games and training epochs.
    """
    train_args.update({
        "numIters":           4,
        "numEps":             4,
        "arenaCompare":       1,
    })

    nn_args.update({
        # "epochs":     1,
        "batch_size": 1024,
    })

    return train_args, nn_args


def dump_run_config(checkpoint_dir, cli, train_args, nn_args):
    """Write the fully-resolved run configuration to run_config.json for reproducibility.

    This records the *actual* values used (after dry-run overrides etc.), the raw CLI
    args, the network args, the git commit, and key environment, so a run can be
    reproduced exactly from this single file regardless of launcher/shell state.
    """
    import json
    import subprocess
    import datetime

    def _git(*cmd):
        try:
            return subprocess.check_output(["git", *cmd], cwd=os.path.dirname(os.path.abspath(__file__)),
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None

    # train_args holds tuples (load_folder_file etc.) that JSON can't serialise as-is.
    def _jsonable(v):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]
        if isinstance(v, dict):
            return {k: _jsonable(x) for k, x in v.items()}
        return str(v)

    config = {
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit":  _git("rev-parse", "HEAD"),
        "git_branch":  _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty":   bool(_git("status", "--porcelain")),
        "argv":        sys.argv,
        "cli_args":    _jsonable(vars(cli)),
        "train_args":  _jsonable(dict(train_args)),
        "nn_args":     _jsonable(dict(nn_args)),
        "env": {
            "SLURM_JOB_ID":   os.environ.get("SLURM_JOB_ID"),
            "hostname":       os.uname().nodename,
            "USE_AMP":        os.environ.get("USE_AMP"),
            "AMP_DTYPE":      os.environ.get("AMP_DTYPE"),
        },
    }
    try:
        import torch
        config["env"]["torch_version"] = torch.__version__
        config["env"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            config["env"]["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    print(f"Wrote run configuration -> {path}")


def pretrain(nnet, data_path, checkpoint_dir, epochs=10):
    import pickle
    import time
    print(f"\n{'='*55}")
    print(f"Supervised pre-training from: {data_path}")
    print(f"Epochs: {epochs}")
    print(f"{'='*55}\n")

    with open(data_path, "rb") as f:
        examples = pickle.load(f)
    print(f"Loaded {len(examples):,} examples.")

    pretrain_start = time.time()

    # Temporarily override epoch count for the supervised phase
    original_epochs = nnet.net_args.epochs
    nnet.net_args.epochs = epochs
    nnet.train(examples, print_summary=True, lr_override=5e-4)
    nnet.net_args.epochs = original_epochs

    pretrain_elapsed = time.time() - pretrain_start
    h, m = divmod(int(pretrain_elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"Pre-training total wall time: {h:02d}h {m:02d}m {s:02d}s")

    # Save as checkpoint_0 so this snapshot is separately identifiable
    os.makedirs(checkpoint_dir, exist_ok=True)
    nnet.save_checkpoint(folder=checkpoint_dir, filename="checkpoint_0.pth.tar")
    nnet.save_checkpoint(folder=checkpoint_dir, filename="best.pth.tar")
    print("Pre-trained weights saved as checkpoint_0.pth.tar + best.pth.tar\n")


def set_seed(seed):
    """Seed python/numpy/torch for reproducibility across runs."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    cli = parse_args()

    if cli.seed is not None:
        set_seed(cli.seed)

    use_amp   = cli.use_amp   if cli.use_amp   is not None else int(os.environ.get("USE_AMP",   "1"))
    amp_dtype = cli.amp_dtype if cli.amp_dtype is not None else os.environ.get("AMP_DTYPE", "bf16")
    if amp_dtype not in ("fp16", "bf16"):
        amp_dtype = "bf16"

    checkpoint_dir = f"./temp/{cli.exp_name}"

    print(f"{'='*55}")
    print(f"Experiment:     {cli.exp_name}")
    print(f"MCTS sims:      {cli.mcts_sims}")
    print(f"LR schedule:    {describe_lr_schedule(cli)}")
    print(f"Pretrain data:  {cli.load_pretrain_data or 'none'}")
    print(f"Expert fill:    {cli.expert_examples_data or 'none'}")
    print(f"Fill enabled:   {bool(cli.fill_with_expert_data)}")
    print(f"Heuristic α:    {cli.heuristic_alpha}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"AMP:            {use_amp} ({amp_dtype})")
    print(f"Seed:           {cli.seed if cli.seed is not None else 'none'}")
    print(f"Attn depth:     {cli.attn_depth}")
    print(f"SE enabled:     {bool(cli.se_enabled)}")
    print(f"Fast opts:      {bool(cli.fast_opts)} (cudnn.benchmark + channels_last + torch.compile)")
    print(f"Replay window:  {cli.train_history_iters} iters | train cap {cli.max_train_size:,} | max game len {cli.max_game_length}")
    if cli.dry_run:
        print(f"Dry run:        enabled (~{cli.dry_run_minutes} min target)")
    print(f"{'='*55}\n")

    nn_args = build_nn_args(use_amp, amp_dtype, attn_depth=cli.attn_depth, se_enabled=cli.se_enabled,
                            fast_opts=cli.fast_opts)
    args = build_train_args(cli, checkpoint_dir)

    if cli.dry_run:
        args, nn_args = apply_dry_run_overrides(cli, args, nn_args)

    NNet.args.update(nn_args)

    # Persist the fully-resolved configuration for reproducibility.
    dump_run_config(checkpoint_dir, cli, args, nn_args)

    g    = Game(9)
    nnet = nn(g)

    best_path     = os.path.join(checkpoint_dir, "best.pth.tar")
    examples_path = os.path.join(checkpoint_dir, "latest.examples")

    if os.path.isfile(best_path):
        print(f"Resuming from {best_path}")
        args.load_model = True
        nnet.load_checkpoint(checkpoint_dir, "best.pth.tar")
        if os.path.isfile(examples_path):
            print(f"Loading examples from {examples_path}")
            args.load_examples = True
        else:
            print("No examples file, starting fresh.")
    else:
        print("No existing checkpoint — starting from scratch.")

    if cli.load_pretrain_data:
        if not os.path.isfile(cli.load_pretrain_data):
            print(f"ERROR: pretrain file not found: {cli.load_pretrain_data}")
            sys.exit(1)
        pretrain(nnet, cli.load_pretrain_data, checkpoint_dir, epochs=cli.pretrain_epochs)

    c = Coach(g, nnet, args)
    if args.load_examples:
        c.loadTrainExamples()

    c.learn()
