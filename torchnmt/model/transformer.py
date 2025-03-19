from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import TransformerParams


class PadMasker(nn.Module):
    """
    Creates masks to ignore padded positions in sequences during attention computation.

    Compatible for use within MultiHeadAttention modules.

    Attributes:
        pad_ix (int): The index used to identify padding tokens in the input sequences.
    """

    def __init__(self, pad_ix: int):
        super().__init__()
        self.pad_ix = pad_ix

    @classmethod
    def from_params(cls, params: TransformerParams):
        return cls(params.pad_ix) if not params.naive else NaivePadMasker(params.pad_ix)

    def forward(self, input: Tensor) -> Tensor:
        """
        Computes the padding mask for the input sequences.

        Args:
            input (Tensor): Input tensor.
                - Shape: (B, max_seq_len)

        Returns:
            Tensor: A boolean mask tensor, expanded to be compatible with multi-head attention.
                - Shape: (B, 1, 1, max_seq_len)
                    - Dim 1: head dimension
                    - Dim 2: query dimension
        """

        mask = input != self.pad_ix  # (B, max_seq_len)
        mask = mask.view(input.shape[0], 1, 1, input.shape[1])  # (B, 1, 1, max_seq_len)
        return mask


class NaivePadMasker(PadMasker):
    """
    Creates masks to ignore padded positions in sequences during attention computation.

    Compatible for use within NaiveMultiHeadAttention modules.

    Attributes:
        pad_ix (int): The index used to identify padding tokens in the input sequences.
    """

    def __init__(self, pad_ix: int):
        super().__init__(pad_ix)

    def forward(self, input: Tensor) -> Tensor:
        """
        Computes the padding mask for the input sequences.

        Args:
            input (Tensor): Input tensor.
                - Shape: (B, max_seq_len)

        Returns:
            Tensor: A boolean mask tensor, expanded to be compatible with naive multi-head attention.
                - Shape: (B, 1, max_seq_len)
                    - Dim 1: query dimension
        """

        mask = input != self.pad_ix  # (B, max_seq_len)
        mask = mask.view(input.shape[0], 1, input.shape[1])  # (B, 1, max_seq_len)
        return mask


class TargetMasker(nn.Module):
    """
    Creates masks for the causal self-attention mechanism within TransformerDecoder modules by combining padding masks and
    subsequent masks.

    Compatible for use within MultiHeadAttention modules.

    Attributes:
        pad_masker (nn.Module): A module for creating padding masks. Uses `PadMasker` by default,
                                or `NaivePadMasker` if `naive=True`.
    """

    def __init__(self, pad_ix: int, naive: bool = False):
        super().__init__()
        self.pad_masker = PadMasker(pad_ix) if not naive else NaivePadMasker(pad_ix)

    @classmethod
    def from_params(cls, params: TransformerParams):
        return cls(params.pad_ix) if not params.naive else NaiveTargetMasker(params.pad_ix)

    @classmethod
    def _get_subsequent_mask(cls, input: Tensor) -> Tensor:
        """
        Creates a subsequent mask for enforcing causal masking.

        Args:
            input (Tensor): Input tensor.
                - Shape: (B, max_seq_len)

        Returns:
            Tensor: A boolean mask tensor.
                - Shape: (B, max_seq_len, max_seq_len).
        """

        mask = torch.tril(
            torch.ones(input.shape[1], input.shape[1], device=input.device)
        ).bool()  # (max_seq_len, max_seq_len)
        mask = mask.unsqueeze(0).expand(input.shape[0], -1, -1)  # (B, max_seq_len, max_seq_len)
        return mask

    def forward(self, input: Tensor) -> Tensor:
        """
        Computes the combined target mask, including both padding and subsequent masks.

        Args:
            input (Tensor): Target input tensor.
                - Shape: (B, max_tgt_len)

        Returns:
            Tensor: A boolean mask tensor, expanded to be compatible with multi-head attention.
                - Shape: (B, 1, max_tgt_len, max_tgt_len)
                    - Dim 1: head dim
        """

        tgt_pad_mask = self.pad_masker(input)  # (B, 1, 1, max_tgt_len)
        tgt_pad_mask = tgt_pad_mask.expand(-1, -1, input.shape[1], -1)  # (B, 1, max_tgt_len, max_tgt_len)

        tgt_sub_mask = self._get_subsequent_mask(input)  # (B, max_tgt_len, max_tgt_len)
        tgt_sub_mask = tgt_sub_mask.unsqueeze(1)  # (B, 1, max_tgt_len, max_tgt_len)

        return tgt_pad_mask & tgt_sub_mask  # (B, 1, max_tgt_len, max_tgt_len)


class NaiveTargetMasker(TargetMasker):
    """
    Creates masks for the causal self-attention mechanism within TransformerDecoder modules by combining padding masks and
    subsequent masks.

    Compatible for use within NaiveMultiHeadAttention modules.

    Attributes:
        pad_masker (nn.Module): A `NaivePadMasker` module for creating padding masks.
    """

    def __init__(self, pad_ix: int):
        super().__init__(pad_ix, naive=True)

    def forward(self, input: Tensor) -> Tensor:
        """
        Computes the combined target mask, including both padding and subsequent masks.

        Args:
            input (Tensor): Target input tensor.
                - Shape: (B, max_tgt_len)

        Returns:
            Tensor: A boolean mask tensor, compatible with naive multi-head attention.
                - Shape: (B, max_tgt_len, max_tgt_len)
        """

        tgt_pad_mask = self.pad_masker(input)  # (B, 1, max_tgt_len)
        tgt_pad_mask = tgt_pad_mask.expand(-1, input.shape[1], -1)  # (B, max_tgt_len, max_tgt_len)

        tgt_sub_mask = self._get_subsequent_mask(input)  # (B, max_tgt_len, max_tgt_len)

        return tgt_pad_mask & tgt_sub_mask  # (B, max_tgt_len, max_tgt_len)


def get_sine_encoding_matrix(max_seq_len: int, d: int, n: int = 10000) -> Tensor:
    """
    Computes sinusoidal positional encodings up to the given maximum sequence length.

    Args:
        max_seq_len (int): The maximum sequence length (number of positions).
        d (int): The dimensionality of the encoding.
        n (int, optional): The scaling factor for positional frequencies. Defaults to 10000.

    Returns:
        Tensor: A tensor containing the positional encodings.
            - Shape: (max_seq_len, d)
    """

    pos = torch.ones((max_seq_len, d // 2)) * torch.arange(max_seq_len).unsqueeze(1)
    dim = torch.arange(pos.shape[1]).repeat((pos.shape[0], 1))
    x = pos / (n ** (2 * dim / d))
    sin = torch.sin(x)
    cos = torch.cos(x)
    return torch.cat((sin.unsqueeze(2), cos.unsqueeze(2)), dim=2).view(max_seq_len, -1)  # (max_seq_len, d_model)


class SinePositionalEncoding(nn.Module):
    """
    Generates sinusoidal positional encodings.

    Attributes:
        encoding (Tensor): A precomputed positional encoding matrix, stored as a buffer.
            - Shape: (max_seq_len, d_model)

    Args:
        max_seq_len (int): The maximum sequence length for which positional encodings are generated.
        d_model (int): The dimensionality of the encoding.
    """

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.register_buffer("encoding", get_sine_encoding_matrix(max_seq_len, d_model), persistent=False)

    def forward(self, pos: Tensor) -> Tensor:
        """
        Retrieves the positional encoding for specific positions.

        Args:
            pos (Tensor): The positions for which to retrieve the encodings.
                - Shape: (B, max_seq_len)

        Returns:
            Tensor: The positional encodings.
                - Shape: (B, max_seq_len, d_model)
        """

        return self.get_buffer("encoding")[pos]


class Embedding(nn.Module):
    """
    Computes token and positional embeddings.

    Attributes:
        tok_emb (nn.Embedding): The embedding layer for token representations.
        pos_emb (nn.Module): The positional encoding layer, which can be either:
            - SinePositionalEncoding: For sinusoidal positional encodings.
            - nn.Embedding: For learned positional encodings.
        dropout (nn.Dropout): A dropout layer applied to the combined embeddings.
        scale_factor (Tensor): A scaling factor for token embeddings (only needed w/ sine).
    """

    def __init__(self, params: TransformerParams):
        super().__init__()
        self.tok_emb = nn.Embedding(params.vocab_size, params.d_model)
        if params.sine_pos:
            self.pos_emb: nn.Module = SinePositionalEncoding(params.max_seq_len, params.d_model)
            scale_factor = torch.sqrt(torch.tensor([params.d_model]))
        else:
            self.pos_emb = nn.Embedding(params.max_seq_len, params.d_model)
            scale_factor = torch.tensor([1.0])
        self.dropout = nn.Dropout(params.dropout)
        self.register_buffer("scale_factor", scale_factor, persistent=False)

    def forward(self, input: Tensor) -> Tensor:  # (B, max_seq_len)
        """
        Computes the combined token and positional embeddings.

        Args:
            input (Tensor): Input tensor of token indices.
                - Shape: (B, max_seq_len)

        Returns:
            Tensor: The combined embeddings.
                - Shape: (B, max_seq_len, d_model)
        """

        input_emb = self.tok_emb(input) * self.get_buffer("scale_factor")  # (B, max_seq_len, d_model)

        input_pos = (
            torch.arange(0, input.shape[1], device=input.device).unsqueeze(0).expand(input.shape[0], -1)
        )  # (B, max_seq_len)

        input_pos_emb = self.pos_emb(input_pos)  # (B, max_seq_len, d_model)

        emb = self.dropout(input_emb + input_pos_emb)  # (B, max_seq_len, d_model)
        return emb


class AttentionHead(nn.Module):
    """
    Implements a single attention head for scaled dot-product attention.

    Attributes:
        W_q (nn.Linear): The linear transformation for the queries input.
        W_k (nn.Linear): The linear transformation for the keys input.
        W_v (nn.Linear): The linear transformation for the values input.
        dropout (nn.Dropout): A dropout layer applied to the attention weights.
        scale_factor (Tensor): A scaling factor (`sqrt(d_head)`) used to normalize the attention logits.
    """

    def __init__(self, d_model: int, d_head: int, dropout: float):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("scale_factor", torch.sqrt(torch.tensor([d_head])), persistent=False)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Performs scaled dot-product attention for the given queries, keys, and values tensors.

        Steps:
            - Queries, keys, and values tensors are each transformed / projected down to the head's dimensionality.
            - Transformed queries are multiplied by the transposition of the transformed keys tensor, to obtain the raw
            attention scores.
            - Raw scores are:
                - Scaled down by the square root of the head's dimensionality, and
                - Populated with high magnitude negative scalars at positions indicated by the mask tensor, to ensure
                that those positions will have no effect on the upcoming softmax calculation.
            - Scaled, masked scores are softmaxed, to obtain the actual attention scores.
            - Attention scores are multiplied by the transformed values tensor, to obtain the module's output.

        The `V` tensor, of dimensions (B, value_len, d_head), represents some data that the head has learned from the
        values tensor. By batch multiplying the attention weights, of dimensions (B, query_len, key_len), by it, we obtain
        an output tensor of shape (B, query_len, d_head).
            - Each i'th value in a query's result vector within the output tensor (dim 2, size=d_head) is now a weighted
            sum of the corresponding i-th values in the values vectors within the `V` tensor (dim 2, size=d_head) that
            are present for each timestep (dim 1, size=value_len).
            - In this way, the model can "choose", for each value in the query's result vector, based on the attention
            distribution computed using the query, to magnify the impact that certain timesteps have on the resulting
            value, and minimize the impact of other timesteps.

        Args:
            queries (Tensor): Queries tensor, used to compute the attention scores.
                - Shape: (B, query_len, d_model)
            keys (Tensor): Keys tensor, used to compute the attention scores.
                - Shape: (B, key_len, d_model)
            values (Tensor): Values tensor, used to compute the output tensor.
                - Shape: (B, value_len, d_model)
            mask (Tensor): A mask for the keys tensor, in which non-relevant positions are set to false / zero to exclude them
            from the attention computation.
                - Shape:
                    - (B, 1, key_len) if utilized as bidirectional self-attention or cross-attention
                    - (B, query_len, key_len) if utilized as causal self-attention (query_len == key_len)

        Returns:
            tuple[Tensor, Tensor]:
                Tensor: The output tensor.
                    - Shape: (B, query_len, d_head)
                Tensor: The attention weights.
                    - Shape: (B, query_len, key_len)
        """

        Q = self.W_q(queries)  # (B, query_len, d_head)
        K = self.W_k(keys)  # (B, key_len, d_head)
        V = self.W_v(values)  # (B, value_len, d_head)

        scores = Q @ K.permute(0, 2, 1)  # (B, query_len, d_head) @ (B, d_head, key_len) = (B, query_len, key_len)
        scores = scores / self.get_buffer("scale_factor")
        scores = scores.masked_fill(mask == 0, -1e10)  # mask is broadcast along query dim if necessary

        attention = F.softmax(scores, dim=2)

        out = self.dropout(attention) @ V  # (B, query_len, key_len) @ (B, value_len, d_head) = (B, query_len, d_head)

        return out, attention


class NaiveMultiHeadAttention(nn.Module):
    """
    Implements a "naive" multi-head attention mechanism for transformer models.

    By naive, I mean that the results for each attention head are calculated sequentially, within separate AttentionHead
    instances, and then concatenated together after. Compared to the standard implementation that calculates attention
    across all heads simultaneously, this is easier to follow, but much slower.

    Attributes:
        heads (nn.ModuleList): A list of `AttentionHead` modules, one for each desired attention head.
        proj (nn.Linear): A linear layer that projects the concatenated outputs of all attention heads
                          to the model's dimensionality.
        dropout (nn.Dropout): A dropout layer applied to the module's output.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        d_head = d_model // n_heads
        self.heads = nn.ModuleList([AttentionHead(d_model, d_head, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor, scores_by_layer: list[Tensor]
    ) -> Tensor:
        """
        Computes the multi-head attention output, and optionally stores the attention weights within the given list.

        Args:
            queries (Tensor): Queries tensor.
                - Shape: (B, query_len, d_model)
            keys (Tensor): Keys tensor.
                - Shape: (B, key_len, d_model)
            values (Tensor): Values tensor.
                - Shape (B, value_len, d_model)
            mask (Tensor): A mask for the keys tensor, in which non-relevant positions are set to false / zero to exclude them
            from the attention computation.
                - Shape:
                    - (B, 1, key_len) if utilized as bidirectional self-attention or cross-attention
                    - (B, query_len, key_len) if utilized as causal self-attention (query_len == key_len)
            scores_by_layer (list[Tensor]): A list to which the combined attention scores from each head can be appended (optional).
                - Shape: (B, n_heads, query_len, key_len)

        Returns:
            Tensor: The output tensor.
                - Shape: (B, query_len, d_model)
        """

        res = [
            head(queries, keys, values, mask) for head in self.heads
        ]  # [ ((B, query_len, d_head), (B, query_len, key_len)) * n_heads ]

        heads_out, attention = zip(*res, strict=True)
        heads_out = torch.cat(heads_out, dim=2)  # type: ignore # (B, query_len, d_model)

        if scores_by_layer is not None:
            scores_by_layer.append(torch.stack(attention, dim=1))  # (B, n_heads, query_len, key_len)

        out = self.dropout(self.proj(heads_out))  # (B, query_len, d_model)
        return out


class MultiHeadAttention(nn.Module):
    """
    An implementation of multi-head attention for transformer models.

    Can be utilized for all attention variants: bidirectional self-attention in the encoder, and causal self-attention and
    cross-attention in the decoder. Note that it is not necessary to specify the attention variant as an argument here because the
    desired result will occur naturally, as a result of passing in the correct queries, keys, values, and mask tensors for
    that variant.

    Attributes:
        d_model (int): The model's dimensionality.
        n_heads (int): Desired number of attention heads.
        d_head (int): Dimensionality of each attention head.
        W_q (nn.Linear): The linear transformation for the queries input.
        W_k (nn.Linear): The linear transformation for the keys input.
        W_v (nn.Linear): The linear transformation for the values input.
        attn_dropout (nn.Dropout): A dropout layer applied to the attention weights.
        proj (nn.Linear): A linear layer that projects the concatenated outputs of all attention heads
                          to the model's dimensionality.
        dropout (nn.Dropout): A dropout layer applied to the module's output.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.register_buffer("scale_factor", torch.sqrt(torch.tensor([self.d_head])), persistent=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor, scores_by_layer: list[Tensor] | None
    ) -> Tensor:
        """
        Computes the multi-head attention output, and optionally stores the attention weights within the given list.

        Args:
            queries (Tensor): Queries tensor.
                - Shape: (B, query_len, d_model)
            keys (Tensor): Keys tensor.
                - Shape: (B, key_len, d_model)
            values (Tensor): Values tensor.
                - Shape (B, value_len, d_model)
            mask (Tensor): A mask for the keys tensor, in which non-relevant positions are set to false / zero to exclude them
            from the attention computation.
                - Shape:
                    - (B, 1, 1, key_len) if utilized as bidirectional self-attention or cross-attention
                    - (B, 1, query_len, key_len) if utilized as causal self-attention (query_len == key_len)
            scores_by_layer (list[Tensor] | None): A list to which the combined attention scores from each head can be appended (optional).
                - Shape: (B, n_heads, query_len, key_len)

        Returns:
            Tensor: The output tensor.
                - Shape: (B, query_len, d_model)
        """

        B = queries.shape[0]

        Q = self.W_q(queries)  # (B, query_len, d_model)
        K = self.W_k(keys)  # (B, key_len, d_model)
        V = self.W_v(values)  # (B, value_len, d_model)

        Q = Q.view(B, -1, self.n_heads, self.d_head)  # (B, query_len, n_heads, d_head)
        Q = Q.permute(0, 2, 1, 3)  # (B, n_heads, query_len, d_head)

        K = K.view(B, -1, self.n_heads, self.d_head)  # (B, key_len, n_heads, d_head)
        K = K.permute(0, 2, 3, 1)  # (B, n_heads, d_head, key_len)

        V = V.view(B, -1, self.n_heads, self.d_head)  # (B, value_len, n_heads, d_head)
        V = V.permute(0, 2, 1, 3)  # (B, n_heads, value_len, d_head)

        scores = (
            Q @ K
        )  # (B, n_heads, query_len, d_head) @ (B, n_heads, d_head, key_len) = (B, n_heads, query_len, key_len)
        scores = scores / self.get_buffer("scale_factor")
        scores = scores.masked_fill(
            mask == 0, -1e10
        )  # mask is broadcast along query dim, if necessary, and then head dim

        attention = F.softmax(scores, dim=-1)
        if scores_by_layer is not None:
            scores_by_layer.append(attention)

        heads_out = (
            self.attn_dropout(attention) @ V
        )  # (B, n_heads, query_len, key_len) @ (B, n_heads, value_len, d_head) = (B, n_heads, query_len, d_head)

        heads_out = heads_out.permute(0, 2, 1, 3).contiguous()  # (B, query_len, n_heads, d_head)
        heads_out = heads_out.view(B, -1, self.d_model)  # (B, query_len, d_model)

        out = self.dropout(self.proj(heads_out))  # (B, query_len, d_model)

        return out


class FeedForward(nn.Module):
    """
    Implements a position-wise feedforward network.

    Attributes:
        ff_1 (nn.Linear): The first linear layer that projects input vectors from `d_model` up to `d_ff`.
        ff_2 (nn.Linear): The second linear layer that projects back down from `d_ff` to `d_model`.
        dropout (nn.Dropout): A dropout layer applied after the activation function for regularization.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.ff_1 = nn.Linear(d_model, d_ff)
        self.ff_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor) -> Tensor:  # (B, max_seq_len, d_model)
        """
        Computes the forward pass of the feedforward network, in which input vectors are projected up to a higher
        dimensionality, and then compressed back down to the original dimensionality.

        Args:
            input (Tensor): Input tensor.
                - Shape: (B, max_seq_len, d_model)

        Returns:
            Tensor: Output tensor.
                - Shape: (B, max_seq_len, d_model)
        """

        out_1 = self.dropout(F.relu(self.ff_1(input)))  # (B, max_seq_len, d_ff)
        out_2 = self.ff_2(out_1)  # (B, max_seq_len, d_model)
        return out_2


class ResidualSubLayer(nn.Module, ABC):
    """
    Abstract base class for implementing a Transformer sub-layer with a residual connection and layer normalization.

    Attributes:
        ln (nn.LayerNorm): A layer normalization module applied before or after the sub-layer computation.
        dropout (nn.Dropout): A dropout layer applied to the output of the sub-layer for regularization.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @abstractmethod
    def forward(self, input: Tensor, sub_layer: Union[nn.Module, Callable[[Tensor], Tensor]]) -> Tensor:
        raise NotImplementedError


class PostNormResidualSubLayer(ResidualSubLayer):
    """
    Implements a residual sub-layer where layer normalization is applied after the residual connection.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__(d_model, dropout)

    def forward(self, input: Tensor, sub_layer: Union[nn.Module, Callable[[Tensor], Tensor]]) -> Tensor:
        """
        Computes the output of the residual sub-layer with Post-LN.

        Args:
            input (Tensor): Input tensor.
                - Shape: (B, seq_len, d_model)
            sub_layer (Union[nn.Module, Callable[[Tensor], Tensor]]): The sub-layer to which the input should be passed.

        Returns:
            Tensor: Output tensor.
                - Shape: (B, seq_len, d_model)
        """

        return self.ln(input + self.dropout(sub_layer(input)))


class PreNormResidualSubLayer(ResidualSubLayer):
    """
    Implements a residual sub-layer where layer normalization is applied within the residual connection.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__(d_model, dropout)

    def forward(self, input: Tensor, sub_layer: Union[nn.Module, Callable[[Tensor], Tensor]]) -> Tensor:
        """
        Computes the output of the residual sub-layer with Pre-LN.

        Args:
            input (Tensor): Input tensor.
                - Shape: (B, seq_len, d_model)
            sub_layer (Union[nn.Module, Callable[[Tensor], Tensor]]): The sub-layer to which the normalized input should
            be passed.

        Returns:
            Tensor: Output tensor.
                - Shape: (B, seq_len, d_model)
        """

        return input + self.dropout(sub_layer(self.ln(input)))


class TransformerEncoderLayer(nn.Module):
    """
    Implements a single transformer encoder layer, containing two sub-layers: bidirectional self-attention and a
    feedforward network.

    Attributes:
        attention (nn.Module): The multi-head attention module, utilized for bidirectional self-attention.
        sub_layers (nn.ModuleList): A list of two sub-layers with residual connections:
            - The first sub-layer wraps the self-attention module.
            - The second sub-layer wraps the feedforward network.
            The sub-layer type is determined by the `pre_norm` parameter.
    """

    def __init__(self, params: TransformerParams):
        super().__init__()
        att_cls = NaiveMultiHeadAttention if params.naive else MultiHeadAttention
        self.attention = att_cls(params.d_model, params.n_heads, params.dropout)
        self.ff = FeedForward(params.d_model, params.d_ff, params.dropout)
        sublayer_cls = PreNormResidualSubLayer if params.pre_norm else PostNormResidualSubLayer
        self.sub_layers = nn.ModuleList([sublayer_cls(params.d_model, params.dropout) for _ in range(2)])

    def forward(self, input: Tensor, mask: Tensor, self_scores: list[Tensor] | None) -> Tensor:
        """
        Computes the forward pass by feeding input through a bidirectional self-attention sub-layer, then a feedforward sub-layer.

        Args:
            input (Tensor): Input tensor.
                - Shape: (B, max_src_len, d_model)
            mask (Tensor): Bidirectional self-attention mask.
                - Shape: (B, 1, 1, max_src_len)
            self_scores (list[Tensor] | None): A list of self-attention scores from each layer.

        Returns:
            Tensor: Output tensor.
                - Shape: (B, max_src_len, d_model)
        """

        out = self.sub_layers[0](
            input, lambda x: self.attention(x, x, x, mask, self_scores)
        )  # (B, max_src_len, d_model)

        out = self.sub_layers[1](out, self.ff)  # (B, max_src_len, d_model)

        return out


class TransformerEncoder(nn.Module):
    """
    Implements a stack of TransformerEncoderLayer modules, which ultimately produces outputs that a decoder
    can utilize for cross-attention.

    Attributes:
        params (TransformerParams): Configuration object containing model configuration parameters.
        emb (Embedding): The embedding layer for token and positional embeddings.
        layers (nn.ModuleList): A list of `TransformerEncoderLayer` modules, each implementing self-attention
                                and feedforward operations with residual connections.
        ln (nn.LayerNorm, optional): A layer normalization module applied after all encoder layers if pre-normalization is enabled.
    """

    def __init__(self, params: TransformerParams):
        super().__init__()
        self.params = params
        self.emb = Embedding(params)
        self.layers = nn.ModuleList([TransformerEncoderLayer(params) for _ in range(params.n_layers)])
        if params.pre_norm:
            self.ln = nn.LayerNorm(params.d_model)

    def forward(self, src: Tensor, mask: Tensor) -> tuple[Tensor, list[Tensor] | None]:
        """
        Computes token + positional embeddings and then processes them through a stack of encoder layers
        in which the output of one layer becomes the input to the next, progressively refining the input
        representations.

        Args:
            src (Tensor): Padded batch of source sequences.
                - Shape: (B, max_src_len)
            mask (Tensor): Bidirectional self-attention mask.
                - Shape: (B, 1, 1, max_src_len)

        Returns:
            tuple[Tensor, list[Tensor] | None]:
                Tensor: Final encoder outputs.
                    - Shape: (B, max_src_len, d_model)
                list[Tensor] | None: List of attention scores from each layer (eval mode) or `None` (train mode).
                    - Shape: (B, n_heads, max_src_len, max_src_len)
        """

        src_emb = self.emb(src)  # (B, max_src_len, d_model)

        self_scores: list[Tensor] | None = None if self.training else []

        enc_out = src_emb
        for layer in self.layers:
            enc_out = layer(enc_out, mask, self_scores)

        if self.params.pre_norm:
            enc_out = self.ln(enc_out)

        return (
            enc_out,
            self_scores,
        )  # (B, max_src_len, d_model), [(B, n_heads, max_src_len, max_src_len) * n_layers] | None


class TransformerDecoderLayer(nn.Module):
    """
    Implements a single transformer decoder layer, containing three sub-layers: causal self-attention, cross-attention,
    and a feedforward network.

    Attributes:
        self_attention (nn.Module): The multi-head attention module utilized for causal self-attention.
        cross_attention (nn.Module): The multi-head attention module utilized for cross-attention with the encoder outputs.
        ff (FeedForward): The position-wise feedforward network module.
        sub_layers (nn.ModuleList): A list of three sub-layers with residual connections:
            - The first sub-layer wraps the self-attention module.
            - The second sub-layer wraps the cross-attention module.
            - The third sub-layer wraps the feedforward network.
            The sub-layer type is determined by the `pre_norm` parameter.
    """

    def __init__(self, params: TransformerParams):
        super().__init__()
        att_cls = NaiveMultiHeadAttention if params.naive else MultiHeadAttention
        self.self_attention = att_cls(params.d_model, params.n_heads, params.dropout)
        self.cross_attention = att_cls(params.d_model, params.n_heads, params.dropout)
        self.ff = FeedForward(params.d_model, params.d_ff, params.dropout)
        sublayer_cls = PreNormResidualSubLayer if params.pre_norm else PostNormResidualSubLayer
        self.sub_layers = nn.ModuleList([sublayer_cls(params.d_model, params.dropout) for _ in range(3)])

    def forward(
        self,
        input: Tensor,
        enc_out: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        self_scores: list[Tensor] | None,
        cross_scores: list[Tensor] | None,
    ) -> Tensor:
        """
        Computes the forward pass by feeding input through a causal self-attention sub-layer, then a cross-attention sub-layer,
        and finally a feedforward sub-layer.

        Args:
            input (Tensor): Input tensor.
                - Shape: (B, max_tgt_len, d_model)
            enc_out (Tensor): Encoder outputs tensor.
                - Shape: (B, max_src_len, d_model)
            src_mask (Tensor): Cross-attention mask for the source sequences / encoder outputs.
                - Shape: (B, 1, 1, max_src_len)
            tgt_mask (Tensor): Causal self-attention mask for the target sequences.
                - Shape: (B, 1, max_tgt_len, max_tgt_len)
            self_scores (list[Tensor] | None): A list of self-attention scores from each layer.
            cross_scores (list[Tensor] | None): A list of cross-attention scores from each layer.

        Returns:
            Tensor: Output tensor.
                - Shape: (B, max_src_len, d_model)
        """

        out = self.sub_layers[0](
            input, lambda x: self.self_attention(x, x, x, tgt_mask, self_scores)
        )  # (B, max_tgt_len, d_model)

        out = self.sub_layers[1](
            out, lambda x: self.cross_attention(x, enc_out, enc_out, src_mask, cross_scores)
        )  # (B, max_tgt_len, d_model)

        out = self.sub_layers[2](out, self.ff)  # (B, max_tgt_len, d_model)

        return out


class TransformerDecoder(nn.Module):
    """
    Implements a stack of TransformerDecoderLayer modules, followed by a fully-connected layer, which produces logits over
    the target vocabulary.

    Attributes:
        params (TransformerParams): Configuration object containing model configuration parameters.
        emb (Embedding): The embedding layer for token and positional embeddings.
        layers (nn.ModuleList): A list of `TransformerDecoderLayer` modules, each implementing self-attention,
                                cross-attention, and feedforward operations with residual connections.
        ln (nn.LayerNorm, optional): A layer normalization module applied after all encoder layers if pre-normalization is enabled.
        fc (nn.Linear): A fully-connected layer producing logits over the target vocabulary.
    """

    def __init__(self, params: TransformerParams):
        super().__init__()
        self.params = params
        self.emb = Embedding(params)
        self.layers = nn.ModuleList([TransformerDecoderLayer(params) for _ in range(params.n_layers)])
        if params.pre_norm:
            self.ln = nn.LayerNorm(params.d_model)
        self.fc = nn.Linear(params.d_model, params.vocab_size)

    def forward(
        self, tgt: Tensor, enc_out: Tensor, src_mask: Tensor, tgt_mask: Tensor
    ) -> tuple[
        Tensor, list[Tensor] | None, list[Tensor] | None
    ]:  # (B, max_tgt_len), (B, max_src_len, d_model), (B, 1, max_src_len), (B, max_tgt_len, max_tgt_len)
        """
        Computes token + positional embeddings and then processes them through a stack of decoder layers
        in which the output of one layer becomes the input to the next, before finally feeding the stack's output
        into a fully connected layer to produce logits over the target vocabulary.

        Args:
            tgt (Tensor): Padded batch of target sequences.
                - Shape: (B, max_tgt_len)
            enc_out (Tensor): Encoder ouputs tensor.
                - Shape: (B, max_src_len, d_model)
            src_mask (Tensor): Cross-attention mask for the source sequences / encoder outputs.
                - Shape: (B, 1, 1, max_src_len)
            tgt_mask (Tensor): Causal self-attention mask for the target sequences.
                - Shape: (B, 1, max_tgt_len, max_tgt_len)

        Returns:
            tuple[Tensor, list[Tensor] | None, list[Tensor] | None]:
                Tensor: Logits over the target vocabulary.
                    - Shape: (B, max_tgt_len, tgt_vocab_size)
                list[Tensor] | None: List of causal self-attention scores from each layer (eval mode) or `None` (train mode).
                    - Shape: (B, n_heads, max_tgt_len, max_tgt_len)
                list[Tensor] | None: List of cross-attention scores from each layer (eval mode) or `None` (train mode).
                    - Shape: (B, n_heads, max_tgt_len, max_src_len)
        """

        tgt_emb = self.emb(tgt)  # (B, max_tgt_len, d_model)

        self_scores: list[Tensor] | None = None
        cross_scores: list[Tensor] | None = None
        if not self.training:
            self_scores = []
            cross_scores = []

        dec_out = tgt_emb
        for layer in self.layers:
            dec_out = layer(
                dec_out, enc_out, src_mask, tgt_mask, self_scores, cross_scores
            )  # (B, max_tgt_len, d_model)

        if self.params.pre_norm:
            dec_out = self.ln(dec_out)

        dec_out = self.fc(
            dec_out
        )  # (B, max_tgt_len, d_model) @ (d_model, tgt_vocab_size) = (B, max_tgt_len, tgt_vocab_size)

        return dec_out, self_scores, cross_scores


class TransformerSeq2Seq(nn.Module):
    """
    A Transformer-based sequence-to-sequence model used for neural machine translation, based on "Attention is All You Need"
    (Vaswani et al. 2017).

    This class utilizes a TransformerEncoder to process source sequences, and TransformerDecoder to then generate the corresponding
    target sequences.

    Attributes:
        encoder (TransformerEncoder): The transformer encoder module that processes source sequences.
        decoder (TransformerDecoder): The transformer decoder module that generates target sequences.
        src_masker (nn.Module): A module for generating source masks to ignore padding tokens in the source sequences.
        tgt_masker (nn.Module): A module for generating target masks, combining padding masks and subsequent masks
                                to enforce causal masking in the target sequences.
    """

    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_masker = PadMasker.from_params(encoder.params)
        self.tgt_masker = TargetMasker.from_params(decoder.params)

    def forward(self, src: Tensor, tgt: Tensor):  # (B, max_src_len), (B, max_tgt_len)
        """
        Processes a source sequence and generates a target sequence using the provided encoder and decoder.

        Args:
            src (Tensor): Batched source sequences tensor.
                - Shape: (B, max_src_len)
            tgt (Tensor): Batched target sequences tensor.
                - Shape: (B, max_tgt_len)

        Returns:
            Tensor: Logits over the target vocabulary, for each timestep and example.
                - Shape: (B, max_tgt_len, tgt_vocab_size)
        """

        src_mask = self.src_masker(src)  # (B, 1, 1, max_src_len)
        tgt_mask = self.tgt_masker(tgt)  # (B, 1, max_tgt_len, max_tgt_len)

        enc_out, _ = self.encoder(src, src_mask)  # (B, max_src_len, d_model)

        dec_out, *_ = self.decoder(tgt, enc_out, src_mask, tgt_mask)  # (B, max_tgt_len, tgt_vocab_size)
        return dec_out
