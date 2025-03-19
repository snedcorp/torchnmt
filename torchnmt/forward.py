from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

import torch.nn as nn
from torch import Tensor

from torchnmt.data import Batch, RNNBatch, TransformerBatch


B = TypeVar("B", bound=Batch)


class Forward(ABC, Generic[B]):
    """
    An abstract base class for defining the forward pass of a model, with loss computation.

    Attributes:
        model (nn.Module): The PyTorch model to be used in the forward pass.
        device (str): The device (e.g., "cpu" or "cuda") where the model is located.
        loss_fn (nn.CrossEntropyLoss): The loss function used to compute the loss.
    """

    def __init__(self, model: nn.Module, device: str, loss_fn: nn.CrossEntropyLoss):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn

    @abstractmethod
    def __call__(self, batch: B, *args, **kwargs) -> tuple[Tensor, int]:
        raise NotImplementedError


class RNNForward(Forward[RNNBatch]):
    """
    Implements the forward pass for an RNN-based sequence-to-sequence model with teacher forcing.

    Args:
        batch (RNNBatch): The input batch containing:
            - src (Tensor): Padded batch of source sequences.
                - Shape: (B, max_src_len)
            - src_len (Tensor): Source sequence lengths.
                - Shape: (B)
            - tgt (Tensor): Padded batch of target sequences.
                - Shape: (max_tgt_len, B)
        teacher_forcing_rate (float | None, default=0.0):
            The probability of using teacher forcing during this forward pass (providing the true target as the next
            input).

    Returns:
        tuple[Tensor, int]:
            - Tensor: The computed loss for the batch.
            - int: The count of valid (non-padding) tokens used in loss computation.
    """

    def __call__(self, batch: RNNBatch, *args, **kwargs) -> tuple[Tensor, int]:
        src, src_len, tgt, _ = batch
        src = src.to(self.device)
        tgt = tgt.to(self.device)

        teacher_forcing_rate = kwargs.get("teacher_forcing_rate", 0.0)

        logits = self.model(
            src, src_len, tgt, teacher_forcing_rate=teacher_forcing_rate
        )  # (max_tgt_len, B, tgt_vocab_size)
        tgt_vocab_size = logits.shape[-1]
        logits = logits[1:].view(-1, tgt_vocab_size)  # -> (max_tgt_len-1 * B, tgt_vocab_size)
        tgt = tgt[1:].view(-1)  # -> (max_tgt_len-1 * B)

        loss: Tensor = self.loss_fn(logits, tgt)
        cnt = cast(int, (tgt != self.loss_fn.ignore_index).sum().item())

        return loss, cnt


class TransformerForward(Forward[TransformerBatch]):
    """
    Implements the forward pass for a Transformer-based sequence-to-sequence model.

    Args:
        batch (TransformerBatch): The input batch containing:
            - src (Tensor): Padded batch of source sequences.
                - Shape: (B, max_src_len)
            - tgt (Tensor): Padded batch of target sequences.
                - Shape: (B, max_tgt_len)

    Returns:
        tuple[Tensor, int]:
            - Tensor: The computed loss for the batch.
            - int: The count of valid (non-padding) tokens used in loss computation.
    """

    def __call__(self, batch: TransformerBatch, *args, **kwargs) -> tuple[Tensor, int]:
        src, tgt, _ = batch
        src = src.to(self.device)
        tgt = tgt.to(self.device)

        logits = self.model(src, tgt[:, :-1])  # (B, max_tgt_len-1, tgt_vocab_size)

        tgt_vocab_size = logits.shape[-1]
        logits = logits.view(-1, tgt_vocab_size)  # (B * max_tgt_len-1, tgt_vocab_size)
        tgt = tgt[:, 1:].contiguous().view(-1)  # -> (B * max_tgt_len-1)

        loss: Tensor = self.loss_fn(logits, tgt)
        cnt = cast(int, (tgt != self.loss_fn.ignore_index).sum().item())

        return loss, cnt
