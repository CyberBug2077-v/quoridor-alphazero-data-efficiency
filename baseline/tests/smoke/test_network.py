from __future__ import annotations

import math

import numpy as np
import torch

from quoridor.QuoridorGame import QuoridorGame
from quoridor.pytorch.NNet import NNetWrapper


def test_forward_cuda(
    game: QuoridorGame,
    cuda_network: NNetWrapper,
    cuda_device: torch.device,
    initial_state: np.ndarray,
) -> None:
    batch_size = 2
    inputs = torch.from_numpy(
        np.stack([initial_state] * batch_size)
    ).to(device=cuda_device, dtype=torch.float32)
    valids = torch.from_numpy(
        np.stack([game.getValidMoves(initial_state, 1)] * batch_size)
    ).to(device=cuda_device, dtype=torch.bool)

    cuda_network.nnet.eval()
    with torch.no_grad():
        policy_logits, values = cuda_network.nnet(inputs, logits=True)
    probabilities_np, predicted_values_np = cuda_network.predict(
        np.stack([initial_state] * batch_size),
        batch_valids=valids.cpu().numpy(),
    )
    probabilities = torch.from_numpy(probabilities_np).to(cuda_device)
    predicted_values = torch.from_numpy(predicted_values_np).to(cuda_device)

    assert next(cuda_network.nnet.parameters()).device == cuda_device
    assert inputs.device == cuda_device
    assert policy_logits.shape == (batch_size, 136)
    assert values.shape == (batch_size, 1)
    assert probabilities.shape == (batch_size, 136)
    assert predicted_values.shape == (batch_size, 1)
    assert torch.isfinite(policy_logits).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(probabilities).all()
    assert torch.isfinite(predicted_values).all()
    assert torch.all(values >= -1.0)
    assert torch.all(values <= 1.0)
    assert torch.count_nonzero(probabilities[~valids]) == 0
    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones(batch_size, device=cuda_device),
        atol=1e-6,
        rtol=1e-6,
    )


def test_single_optimizer_step(
    cuda_network: NNetWrapper,
    training_examples: list[tuple],
) -> None:
    batch_size = cuda_network.net_args.batch_size
    one_batch = training_examples[:batch_size]
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in cuda_network.nnet.named_parameters()
    }

    metrics = cuda_network.train(
        one_batch,
        print_summary=False,
        available_examples=len(one_batch),
    )

    for metric_name in ("policy_loss", "value_loss", "total_loss"):
        assert math.isfinite(metrics[metric_name])
    assert math.isfinite(metrics["mean_grad_norm"])
    assert metrics["mean_grad_norm"] > 0
    assert metrics["optimizer_steps"] == 1
    assert metrics["training_batches"] == 1
    assert any(
        not torch.equal(parameters_before[name], parameter.detach())
        for name, parameter in cuda_network.nnet.named_parameters()
    )
