"""Training loop for the transition-field networks."""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..settings import log, settings
from .losses import NCC, RegistrationLoss, smoothness_loss
from .model import TransitionFieldNet

__all__ = ["train_transition_network", "save_checkpoint", "seed_everything"]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: torch.nn.Module, path: Union[str, os.PathLike], **extra) -> None:
    """Save ``{"config": model.config, "state_dict": ..., **extra}``.

    The file can be read back with :func:`axis.registration.load_model`.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": getattr(model, "config", {}), "state_dict": model.state_dict(), **extra}
    torch.save(payload, path)


def _to_5d(t: torch.Tensor, device: str) -> torch.Tensor:
    return t.to(device).float().unsqueeze(dim=2)


def train_transition_network(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    epochs: int = 1000,
    lr: float = 1e-4,
    loss_fn: Optional[Callable] = None,
    device: Optional[str] = None,
    num_workers: int = 0,
    max_steps: Optional[int] = None,
    out_dir: Optional[Union[str, os.PathLike]] = None,
    save_every: Optional[int] = None,
    log_every: int = 20,
    seed: Optional[int] = 0,
    callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
) -> Dict[str, List[float]]:
    """Train a transition-field network on all pairs of ``dataset``.

    Mirrors the original training procedure: Adam, batch size 1, shuffled
    ordered pairs, objective of Eq. (20).

    Parameters
    ----------
    model
        :class:`TransitionFieldNet` or :class:`ImageTransitionNet`.
    dataset
        Yields dicts with ``fixed_image``, ``moving_image`` and, for the
        multimodal model, ``fixed_gene``/``moving_gene`` (``[C, H, W]`` tensors).
    epochs
        Passes over all pairs.
    lr
        Adam learning rate.
    loss_fn
        Defaults to :class:`RegistrationLoss` for the multimodal model and to
        morphology NCC + smoothness for the image-only model.
    max_steps
        Optional hard cap on the number of optimisation steps (useful for
        quick runs and tests).
    out_dir, save_every
        When given, checkpoints ``model_{epoch}.pth`` are written every
        ``save_every`` epochs and ``model_final.pth`` at the end.
    seed
        RNG seed for reproducible shuffling / initialisation of the loop.

    Returns
    -------
    History of the loss terms (one entry per step).
    """
    device = settings.resolve_device(device)
    if seed is not None:
        seed_everything(seed)
    multimodal = isinstance(model, TransitionFieldNet)
    if loss_fn is None:
        if multimodal:
            loss_fn = RegistrationLoss(n_mol_channels=model.gene_channels)
        else:
            morph = NCC(win=21)

            def loss_fn(fixed_image, outputs):
                warped, flow, affined, _ = outputs
                m = morph(fixed_image, warped) + morph(fixed_image, affined)
                s = smoothness_loss(flow[:, 1:3])
                return {"loss": m + s, "morph": m, "smooth": s}

    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers, generator=generator)

    history: Dict[str, List[float]] = {}
    step = 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            fixed_image = _to_5d(batch["fixed_image"], device)
            moving_image = _to_5d(batch["moving_image"], device)
            if multimodal:
                fixed_gene = _to_5d(batch["fixed_gene"], device)
                moving_gene = _to_5d(batch["moving_gene"], device)
                outputs = model(fixed_image, fixed_gene, moving_image, moving_gene)
                terms = loss_fn(fixed_image, fixed_gene, outputs)
            else:
                outputs = model(fixed_image, moving_image)
                terms = loss_fn(fixed_image, outputs)

            optimizer.zero_grad()
            terms["loss"].backward()
            optimizer.step()
            step += 1

            for key, value in terms.items():
                history.setdefault(key, []).append(float(value.detach()))
            if callback is not None:
                callback(epoch, step, {k: v[-1] for k, v in history.items()})
            if log_every and step % log_every == 0:
                msg = " ".join(f"{k}={v[-1]:.4f}" for k, v in history.items())
                log(f"[axis] epoch {epoch}/{epochs} step {step} {msg} ({time.time() - start:.0f}s)")
            if max_steps is not None and step >= max_steps:
                break
        if out_dir is not None and save_every and epoch % save_every == 0:
            save_checkpoint(model, Path(out_dir) / f"model_{epoch}.pth", epoch=epoch)
        if max_steps is not None and step >= max_steps:
            break

    if out_dir is not None:
        save_checkpoint(model, Path(out_dir) / "model_final.pth", epoch=epoch)
    model.eval()
    return history
