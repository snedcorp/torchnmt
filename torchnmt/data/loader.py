from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeAlias, TypeVar, Union, cast

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .models import Example


RawBatch = list[tuple[Tensor, Tensor, Example]]

RNNBatch: TypeAlias = tuple[Tensor, Tensor, Tensor, Sequence[Example]]
TransformerBatch: TypeAlias = tuple[Tensor, Tensor, Sequence[Example]]
Batch: TypeAlias = Union[RNNBatch, TransformerBatch]

B = TypeVar("B", bound=Batch)


class Collator(ABC, Generic[B]):
    """
    Abstract base class for collating raw batches into a specific batch format, to be used by a
    PyTorch DataLoader.

    Attributes:
        src_pad_ix (int): Padding token index for the source sequences.
        tgt_pad_ix (int): Padding token index for the target sequences.
    """

    def __init__(self, src_pad_ix: int, tgt_pad_ix: int):
        self.src_pad_ix = src_pad_ix
        self.tgt_pad_ix = tgt_pad_ix

    @abstractmethod
    def __call__(self, batch: RawBatch) -> B:
        raise NotImplementedError


class RNNCollator(Collator[RNNBatch]):
    """
    A collator for RNN models that processes raw batches into the correct RNNBatch format.
    """

    def __call__(self, batch: RawBatch) -> RNNBatch:
        """
        Collates a raw batch of examples into an RNNBatch, to be used by a PyTorch DataLoader.

        Args:
            batch (RawBatch): A list of tuples, where each tuple contains:
                - src (Tensor): The source sequence tensor.
                - tgt (Tensor): The target sequence tensor.
                - example (Example): The original example data.

        Returns:
            tuple[Tensor, Tensor, Tensor, Sequence[Example]]:
                src (Tensor): Padded source sequences (batch-first).
                    - Shape: (B, max_src_len)
                src_len (Tensor): Lengths of the source sequences.
                    - Shape: (B,)
                tgt (Tensor): Padded target sequences.
                    - Shape: (max_tgt_len, B)
                examples (Sequence[Example]): The original example data.
        """

        srcs, tgts, examples = zip(*batch, strict=True)
        src_len = torch.tensor([len(src) for src in srcs], dtype=torch.int)  # (B,)

        src = pad_sequence(list(srcs), batch_first=True, padding_value=self.src_pad_ix)  # (B, max_src_len)
        tgt = pad_sequence(list(tgts), padding_value=self.tgt_pad_ix)  # (max_tgt_len, B)
        return src, src_len, tgt, cast(Sequence[Example], examples)


class TransformerCollator(Collator[TransformerBatch]):
    """
    A collator for Transformer models that processes raw batches into the correct TransformerBatch format.
    """

    def __call__(self, batch: RawBatch) -> TransformerBatch:
        """
        Collates a raw batch of examples into a TransformerBatch, to be used by a PyTorch DataLoader.

        Args:
            batch (RawBatch): A list of tuples, where each tuple contains:
                - src (Tensor): The source sequence tensor.
                - tgt (Tensor): The target sequence tensor.
                - example (Example): The original example data.

        Returns:
            tuple[Tensor, Tensor, Sequence[Example]]:
                src (Tensor): Padded source sequences (batch-first).
                    - Shape: (B, max_src_len)
                tgt (Tensor): Padded target sequences (batch-first).
                    - Shape: (B, max_tgt_len)
                examples (Sequence[Example]): The original example data.
        """

        srcs, tgts, examples = zip(*batch, strict=True)

        src = pad_sequence(list(srcs), batch_first=True, padding_value=self.src_pad_ix)
        tgt = pad_sequence(list(tgts), batch_first=True, padding_value=self.tgt_pad_ix)

        return src, tgt, cast(Sequence[Example], examples)
