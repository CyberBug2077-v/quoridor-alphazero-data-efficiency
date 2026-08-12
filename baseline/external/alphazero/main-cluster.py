from Coach import Coach
from quoridor.QuoridorGame import QuoridorGame as Game
from quoridor.pytorch.NNet import NNetWrapper as nn
from quoridor.pytorch import NNet
from utils import *
import os

TRAINING_NAME = os.environ.get('TRAINING_NAME', 'training_1')
CHECKPOINT_DIR = f'./temp/{TRAINING_NAME}'
USE_AMP = os.environ.get('USE_AMP', '1').lower() in ('1', 'true', 'yes', 'y')
AMP_DTYPE = os.environ.get('AMP_DTYPE', 'fp16').lower()
if AMP_DTYPE not in ('fp16', 'bf16'):
    AMP_DTYPE = 'fp16'

print(f"{'='*60}")
print(f"Training Name: {TRAINING_NAME}")
print(f"Checkpoint Directory: {CHECKPOINT_DIR}")
print(f"AMP Enabled: {USE_AMP} ({AMP_DTYPE})")
print(f"{'='*60}\n")

# Override neural network architecture
NNet.args.update({
    'batch_size': 896, # 64 for 7G of VRAM
    'epochs': 4,
    'num_channels': 96,
    'dim_mults': (1, 2, 4),
    'attn_depth': 6,
    'clip': 0.5,
    'use_amp': USE_AMP,
    'amp_dtype': AMP_DTYPE,
})

args = dotdict({
    'numIters': 1000,
    'numEps': 75,
    'max_game_length': 120,
    'tempThreshold': 30,
    'numMCTSSims': 200,
    'eval_mcts_in_batch': 4,
    'maxlenOfQueue': 500000,
    'max_train_size': 130000,
    'arenaCompare': 4,
    'updateThreshold': -0.51,
    'arena_temp': 0.18,
    'arena_temp_threshold': 5,
    'cpuct': 1.5,
    'dirichlet_alpha': 0.08,     # Noise concentration (larger = broader exploration)
    'dirichlet_epsilon': 0.25,   # Fraction of noise to add (0.25 = 25% exploration, 75% exploitation)

    # Learning rate schedule:
    #   iter 1-400:   lr
    #   iter 401-800: lr / 10
    #   iter 801+:    lr / 100
    'lr': 0.0002,

    'checkpoint': CHECKPOINT_DIR,
    'save_every_n_iterations': 10,   # Harcoded
    'load_model': False,             # Auto-detected below
    'load_examples': False,          # Auto-detected below
    'load_folder_file': (CHECKPOINT_DIR, 'best.pth.tar'),
    'load_folder_examples_file': (CHECKPOINT_DIR, 'latest.examples'),
    'numItersForTrainExamplesHistory': 100,

    'print_summary': True,
})

if __name__=="__main__":
    g = Game(9)
    nnet = nn(g)

    # Auto-detect resume
    best_model_path = os.path.join(args.checkpoint, 'best.pth.tar')
    latest_examples_path = os.path.join(args.checkpoint, 'latest.examples')

    if os.path.isfile(best_model_path):
        print(f"Found existing model at {best_model_path}, resuming training...")
        args.load_model = True
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

        if os.path.isfile(latest_examples_path):
            print(f"Found examples at {latest_examples_path}, loading...")
            args.load_examples = True
        else:
            print("No examples file found, starting fresh examples")
    else:
        print("No existing model found, starting training from scratch")

    c = Coach(g, nnet, args)
    if args.load_examples:
        c.loadTrainExamples()
    c.learn()
