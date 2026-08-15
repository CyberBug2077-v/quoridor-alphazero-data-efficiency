import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../../')
from utils import *
from progress.bar import Bar
from NeuralNet import NeuralNet

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torchvision import datasets, transforms
from torch.autograd import Variable

from .QuoridorNNet import QuoridorNNet as qnnet

args = dotdict({
    'lr': 0.0002,
    'dropout': 0.3,
    'epochs': 4,
    'batch_size': 512,
    'micro_batch_size': 512,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
    'clip': 1.0,
    'weight_decay': 1e-4,
    'lr_decay_gamma': 1.0,
    'use_amp': False,
    'amp_dtype': 'bf16',

    # ResNet architecture
    'num_res_blocks': 6,
    'attn_depth': 1,
    'num_heads': 8,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game, custom_args=None):
        # Use custom args if provided, otherwise use default args
        net_args = args if custom_args is None else custom_args
        self.net_args = net_args
        self.nnet = qnnet(game, net_args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.game = game
        self.use_amp = bool(net_args.get('use_amp', False) and net_args.cuda)
        amp_dtype = net_args.get('amp_dtype', 'fp16')
        if amp_dtype not in ('fp16', 'bf16'):
            amp_dtype = 'fp16'
        self.amp_dtype = torch.float16 if amp_dtype == 'fp16' else torch.bfloat16
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == torch.float16)
        self.fast_opts = bool(net_args.get('fast_opts', False)) and bool(net_args.cuda)
        self.channels_last = self.fast_opts

        if net_args.cuda:
            self.nnet.cuda()

        if self.fast_opts:
            torch.backends.cudnn.benchmark = True
            self.nnet = self.nnet.to(memory_format=torch.channels_last)

        self._fwd = self.nnet
        self._maybe_compile()

    def _maybe_compile(self):
        """Wrap the forward callable with torch.compile when fast_opts is on."""
        if not self.fast_opts:
            self._fwd = self.nnet
            return
        try:
            self._fwd = torch.compile(self.nnet, dynamic=True)
            print("[NNet] fast_opts: torch.compile enabled (dynamic=True), "
                  "cudnn.benchmark=True, channels_last=True")
        except Exception as exc:
            self._fwd = self.nnet
            print(f"[NNet] fast_opts: torch.compile unavailable ({exc}); "
                  "running eager with cudnn.benchmark + channels_last")

    def train(
        self,
        examples,
        withValids=True,
        print_summary=True,
        lr_override=None,
        available_examples=None,
    ):
        """
        examples: list of examples, each example is of form (board, pi, v)
        lr_override: optional learning rate to use instead of net_args.lr
        available_examples: size of the training pool before max_train_size
        """
        net_args = self.net_args
        examples_used = len(examples)
        if available_examples is None:
            available_examples = examples_used

        effective_batch_size = int(net_args.batch_size)
        micro_batch_size = int(net_args.get('micro_batch_size', effective_batch_size))
        if micro_batch_size > effective_batch_size:
            raise ValueError("micro_batch_size must be <= batch_size")
        if effective_batch_size % micro_batch_size != 0:
            raise ValueError("batch_size must be divisible by micro_batch_size")
        accumulation_steps = effective_batch_size // micro_batch_size

        optimizer_steps = 0
        micro_batches_processed = 0
        samples_seen = 0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        total_loss_sum = 0.0
        grad_norm_sum = 0.0
        grad_norm_max = 0.0

        def training_metrics():
            if optimizer_steps > 0:
                mean_policy_loss = policy_loss_sum / optimizer_steps
                mean_value_loss = value_loss_sum / optimizer_steps
                mean_total_loss = total_loss_sum / optimizer_steps
                mean_grad_norm = grad_norm_sum / optimizer_steps
            else:
                mean_policy_loss = None
                mean_value_loss = None
                mean_total_loss = None
                mean_grad_norm = None

            return {
                "available_examples": available_examples,
                "examples_used": examples_used,
                "samples_seen": samples_seen,
                "training_batches": optimizer_steps,
                "optimizer_steps": optimizer_steps,
                "effective_batch_size": effective_batch_size,
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "micro_batches_processed": micro_batches_processed,
                "policy_loss": mean_policy_loss,
                "value_loss": mean_value_loss,
                "total_loss": mean_total_loss,
                "mean_grad_norm": mean_grad_norm,
                "max_grad_norm": grad_norm_max if optimizer_steps > 0 else None,
            }

        if examples_used == 0:
            print("no training examples")
            return training_metrics()

        # wandb.init(project="quoridor alphazero", config=config_dict, mode="disabled" if IS_CI else "run")
        current_lr = lr_override if lr_override is not None else net_args.lr
        optimizer = optim.AdamW(self.nnet.parameters(), lr=current_lr, weight_decay=net_args.weight_decay)

        def count_optimizer_step(_optimizer, _args, _kwargs):
            nonlocal optimizer_steps
            optimizer_steps += 1

        optimizer.register_step_post_hook(count_optimizer_step)
        lr_gamma = net_args.get('lr_decay_gamma', 1.0)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
        start_time = time.time()
        if len(examples[0]) < 4:
            withValids = False

        print(
            f"TRAINING FOR {net_args.epochs} EPOCHS  |  {len(examples):,} examples  "
            f"|  effective batch {effective_batch_size}  |  micro batch {micro_batch_size}  "
            f"|  accumulation {accumulation_steps}"
        )
        for epoch in range(net_args.epochs):
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            epoch_start = time.time()
            end = time.time()

            effective_batches = int(len(examples) / effective_batch_size)
            bar = Bar('Training Net', max=effective_batches)
            batch_idx = 0

            while batch_idx < effective_batches:
                sample_ids = np.random.randint(
                    len(examples), size=effective_batch_size
                )
                data_time.update(time.time() - end)
                optimizer.zero_grad(set_to_none=True)
                step_policy_loss = 0.0
                step_value_loss = 0.0
                step_total_loss = 0.0
                valid_effective_batch = True
                amp_device = "cuda" if net_args.cuda else "cpu"

                for micro_index in range(accumulation_steps):
                    start = micro_index * micro_batch_size
                    stop = start + micro_batch_size
                    micro_ids = sample_ids[start:stop]
                    if withValids:
                        res = list(zip(*[examples[i] for i in micro_ids]))
                        boards, pis, vs, valids = res[0], res[1], res[2], res[3]
                    else:
                        boards, pis, vs = list(zip(*[examples[i] for i in micro_ids]))
                        valids = None
                    boards = torch.FloatTensor(np.array(boards).astype(np.uint8))
                    target_pis = torch.FloatTensor(np.array(pis))
                    target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                    if withValids:
                        valids = torch.FloatTensor(np.array(valids).astype(np.uint8))

                    if net_args.cuda:
                        boards = boards.contiguous().cuda()
                        target_pis = target_pis.contiguous().cuda()
                        target_vs = target_vs.contiguous().cuda()
                        if withValids:
                            valids = valids.contiguous().cuda()
                    boards = Variable(boards)
                    target_pis = Variable(target_pis)
                    target_vs = Variable(target_vs)
                    if withValids:
                        valids = Variable(valids)

                    with torch.amp.autocast(
                        device_type=amp_device,
                        enabled=self.use_amp,
                        dtype=self.amp_dtype,
                    ):
                        out_pi, out_v = self._fwd(boards, withValids)

                        if withValids:
                            out_pi = out_pi * valids
                            out_pi[valids == 0.0] = float('-inf')
                            out_pi = F.softmax(out_pi, dim=1)

                        l_pi = self.loss_pi(target_pis, out_pi)
                        l_v = self.loss_v(target_vs, out_v)
                        total_loss = l_pi + l_v

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(
                            f"WARNING: NaN/Inf detected in loss at batch {batch_idx}, "
                            f"micro-batch {micro_index}. Skipping effective batch."
                        )
                        print(
                            f"  Loss_pi: {l_pi.item():.4f}, "
                            f"Loss_v: {l_v.item():.4f}"
                        )
                        valid_effective_batch = False
                        break

                    backward_loss = total_loss / accumulation_steps
                    if self.use_amp and self.amp_dtype == torch.float16:
                        self.scaler.scale(backward_loss).backward()
                    else:
                        backward_loss.backward()
                    micro_batches_processed += 1
                    step_policy_loss += l_pi.item() / accumulation_steps
                    step_value_loss += l_v.item() / accumulation_steps
                    step_total_loss += total_loss.item() / accumulation_steps

                if not valid_effective_batch:
                    optimizer.zero_grad(set_to_none=True)
                    continue

                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.nnet.parameters(), net_args.clip
                )
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"WARNING: NaN/Inf detected in gradients at batch {batch_idx}. Skipping update.")
                    optimizer.zero_grad(set_to_none=True)
                    if self.use_amp and self.amp_dtype == torch.float16:
                        self.scaler.update()
                    continue

                steps_before_update = optimizer_steps
                if self.use_amp and self.amp_dtype == torch.float16:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                # GradScaler may skip optimizer.step() after detecting overflow.
                if optimizer_steps == steps_before_update:
                    continue

                batch_samples = effective_batch_size
                policy_loss_value = step_policy_loss
                value_loss_value = step_value_loss
                total_loss_value = step_total_loss
                grad_norm_value = float(grad_norm)
                samples_seen += batch_samples
                policy_loss_sum += policy_loss_value
                value_loss_sum += value_loss_value
                total_loss_sum += total_loss_value
                grad_norm_sum += grad_norm_value
                grad_norm_max = max(grad_norm_max, grad_norm_value)

                # record loss
                pi_losses.update(policy_loss_value, batch_samples)
                v_losses.update(value_loss_value, batch_samples)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                            batch=batch_idx,
                            size=effective_batches,
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            lpi=pi_losses.avg,
                            lv=v_losses.avg,
                            )
                bar.next()
            bar.finish()
            epoch_time = time.time() - epoch_start
            print(f"  [Epoch {epoch+1}/{net_args.epochs}]  Loss_pi: {pi_losses.avg:.4f}  Loss_v: {v_losses.avg:.3f}  Time: {format_duration(epoch_time)}")
            if lr_gamma != 1.0:
                scheduler.step()

        metrics = training_metrics()
        if print_summary:
            total_time = time.time() - start_time
            if optimizer_steps > 0:
                print(
                    f"[Training] Epochs: {net_args.epochs} | Total Time: {format_duration(total_time)} "
                    f"| LR: {current_lr:g} | Avg Loss_pi: {metrics['policy_loss']:.4f} "
                    f"| Avg Loss_v: {metrics['value_loss']:.3f} | Optimizer steps: {optimizer_steps}"
                )
            else:
                print(
                    f"[Training] Epochs: {net_args.epochs} | Total Time: {format_duration(total_time)} "
                    f"| LR: {current_lr:g} | Optimizer steps: 0"
                )
        return metrics


    def predict(self, board, valids=None, batch_valids=None):
        """
        board: np array with board or batch of boards
        valids: valid moves for single board
        batch_valids: list/array of valid moves, one per batch element
        """
        is_batch = board.ndim == 4
        board_tensor = torch.FloatTensor(board.astype(np.uint8))
        net_args = self.net_args
        if net_args.cuda:
            board_tensor = board_tensor.contiguous().cuda()

        if not is_batch:
            board_tensor = board_tensor.view(4, self.board_x, self.board_y)

        with torch.no_grad():
            self.nnet.eval()
            amp_device = "cuda" if net_args.cuda else "cpu"
            with torch.amp.autocast(device_type=amp_device, enabled=self.use_amp, dtype=self.amp_dtype):
                pi, v = self._fwd(board_tensor, (valids is not None) or (batch_valids is not None))
            if batch_valids is not None:
                valids_tensor = torch.FloatTensor(np.array(batch_valids).astype(np.uint8)).to(pi.device)
                pi[valids_tensor == 0.0] = float('-inf')
                pi = F.softmax(pi, dim=1)
            elif valids is not None:
                valids_tensor = torch.FloatTensor(valids.astype(np.uint8)).to(pi.device).unsqueeze(0)
                pi[valids_tensor == 0.0] = float('-inf')
                pi = F.softmax(pi, dim=1)

        # numpy() doesn't support bfloat16; cast outputs to float32
        pi = pi.float()
        v = v.float()

        pi_np = pi.data.cpu().numpy()
        v_np = v.data.cpu().numpy()
        if is_batch:
            return pi_np, v_np
        return pi_np[0], v_np[0]

    def loss_pi(self, targets, outputs):
        return (-targets * torch.log(outputs + 1.0e-8)).sum(dim=-1).mean()

    def loss_v(self, targets, outputs):
        return ((targets-outputs.view(-1))**2).mean()
    
    def loss_invalid(self, logits, valids):
        return ((1-valids) * logits.log()).mean()

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            # print("Checkpoint Directory exists! ")
            pass
        torch.save({
            'state_dict' : self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # Detect architecture from checkpoint keys and swap model if needed.
        # Old architecture uses 'initial_conv' / 'attn_blocks' / 'down_blocks';
        # new ResNet architecture uses 'stem' / 'res_blocks'.
        is_legacy = any(k.startswith(('initial_conv', 'attn_blocks', 'down_blocks'))
                        for k in state_dict)
        current_is_legacy = any(k.startswith(('initial_conv', 'attn_blocks', 'down_blocks'))
                                for k in self.nnet.state_dict())

        if is_legacy and not current_is_legacy:
            from .QuoridorNNet_legacy import QuoridorNNet as LegacyNet
            print("[NNet] Legacy architecture detected — loading with legacy model.")
            self.nnet = LegacyNet(self.game, self.net_args)
        elif not is_legacy and current_is_legacy:
            self.nnet = qnnet(self.game, self.net_args)

        self.nnet = self.nnet.to('cpu')
        self.nnet.load_state_dict(state_dict)
        if self.net_args.cuda:
            self.nnet.cuda()

        if self.fast_opts and self.net_args.cuda:
            self.nnet = self.nnet.to(memory_format=torch.channels_last)
        self._maybe_compile()

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
