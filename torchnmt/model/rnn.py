import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Generic, Optional, Protocol, TypedDict, TypeVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .config import MergeMethod, RNNDecoderParams, RNNEncoderParams, ScoreMethod


class GRUCell(nn.Module):
    """
    A gated recurrent unit (GRU) cell implementation.

    The GRU cell processes one time step of input data, updating the hidden state based on the input and the previous
    hidden state.

    Attributes:
        W_r (nn.Linear): The linear layer for computing the reset gate.
        W_h (nn.Linear): The linear layer for computing the candidate hidden state.
        W_z (nn.Linear): The linear layer for computing the update gate.

    Args:
        n_emb (int): Dimensionality of the input embeddings.
        n_hidden (int): Dimensionality of the GRU hidden state.
    """

    def __init__(self, n_emb: int, n_hidden: int):
        super().__init__()
        self.W_r = nn.Linear(n_emb + n_hidden, n_hidden)
        self.W_h = nn.Linear(n_emb + n_hidden, n_hidden)
        self.W_z = nn.Linear(n_emb + n_hidden, n_hidden)

    def forward(self, X_t: Tensor, h_prev: Tensor) -> Tensor:
        """
        Performs a forward pass through the GRU cell for a single time step.

        Args:
            X_t (Tensor): The input tensor at the current time step.
                - Shape: (B, n_emb)
            h_prev (Tensor): The hidden state from the previous time step.
                - Shape: (B, n_hidden)

        Returns:
            Tensor: The updated hidden state for the current time step.
                - Shape: (B, n_hidden)
        """

        X_h = torch.cat((X_t, h_prev), 1)  # (B, n_emb), (B, n_hidden) -> (B, n_emb + n_hidden)

        r_t = F.sigmoid(self.W_r(X_h))  # (B, n_emb + n_hidden) @ (n_emb + n_hidden, n_hidden) = (B, n_hidden)

        h_r = r_t * h_prev
        X_hr = torch.cat((X_t, h_r), 1)  # (B, n_emb), (B, n_hidden) -> (B, n_emb + n_hidden)
        cand_t = F.tanh(self.W_h(X_hr))  # (B, n_emb + n_hidden) @ (n_emb + n_hidden, n_hidden) = (B, n_hidden)

        z_t = F.sigmoid(self.W_z(X_h))  # (B, n_emb + n_hidden) @ (n_emb + n_hidden, n_hidden) = (B, n_hidden)
        h_t = z_t * h_prev + (1 - z_t) * cand_t
        return h_t


class MultiLayerRNNModule(Protocol):
    """
    Protocol that defines the required attributes for an RNN module that operates across multiple layers
    and optionally supports dropout during training.

    Attributes:
        training (bool): Indicates whether the module is in training mode.
        dropout (float): Dropout probability applied between layers during training.
        n_layers (int): Number of layers in the RNN.
        n_hidden (int): Dimensionality of the hidden states for each layer.
    """

    training: bool
    dropout: float
    n_layers: int
    n_hidden: int


class DropoutMixin:
    """
    A mixin class for handling dropout operations in multi-layer RNN modules (used by both encoder and decoder GRUs).
    """

    def init_masks(self: MultiLayerRNNModule, B: int, device: torch.device) -> Tensor | None:
        """
        Initializes dropout masks for intermediate RNN layers.

        Args:
            B (int): Batch size for the input data.
            device (torch.device): The device to create the dropout masks on.

        Returns:
            Tensor | None: Dropout masks for each intermediate layer (if in training mode and dropout is enabled).
                - Shape: (n-layers - 1, B, H)
        """

        if self.training and self.dropout > 0 and self.n_layers > 1:
            return (torch.rand((self.n_layers - 1, B, self.n_hidden), device=device) > self.dropout).float()
        return None

    def apply_dropout(self: MultiLayerRNNModule, state: Tensor, mask: Tensor) -> Tensor:
        """
        Applies dropout to the given tensor using the provided dropout mask.

        Args:
            state (Tensor): The hidden state tensor to which dropout is applied.
                - Shape: (sum(input.batch_sizes), H)
            mask (Tensor): The dropout mask of the same shape as `state`.
                - Shape: (sum(input.batch_sizes, H)

        Returns:
            Tensor: The tensor after applying dropout, scaled by `(1 - dropout)`.
                - Shape: (sum(input.batch_sizes, H)
        """

        return (state * mask) / (1 - self.dropout)


class EncoderGRU(nn.Module, DropoutMixin):
    """
    EncoderGRU is a bidirectional GRU-based encoder that processes packed sequences and generates context-aware
    representations for input data. It supports multiple layers with optional dropout and configurable output merging
    methods.

    Attributes:
        n_emb (int): Dimensionality of the input embeddings.
        n_hidden (int): Dimensionality of the GRU hidden states.
        n_layers (int): Number of GRU layers in the encoder.
        forward_cells (nn.ModuleList): List of GRU cells for the forward direction - one per layer.
        backward_cells (nn.ModuleList): List of GRU cells for the backward direction - one per layer.
        dropout (float): Dropout probability applied between layers.
        merge_method (MergeMethod): Method for merging forward and backward outputs.
    """

    def __init__(self, n_emb: int, n_hidden: int, n_layers: int, dropout: float, merge_method: MergeMethod):
        super().__init__()
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.forward_cells = nn.ModuleList(
            [GRUCell(n_emb if i == 0 else n_hidden * 2, n_hidden) for i in range(n_layers)]
        )
        self.backward_cells = nn.ModuleList(
            [GRUCell(n_emb if i == 0 else n_hidden * 2, n_hidden) for i in range(n_layers)]
        )
        self.dropout = dropout
        self.merge_method = merge_method

    def forward(self, input: PackedSequence, ix_mapping: list[int]) -> tuple[PackedSequence, Tensor, Tensor]:
        """
        Performs a forward pass through a multi-layer, bidirectional GRU-based encoder for the given packed input sequence.

        This method encodes the input sequence bidirectionally, applies dropout between layers if in training mode,
        and merges the forward and backward states according to the specified merging method.

        Args:
            input (PackedSequence): The input packed sequence to encode. It contains the input data and batch sizes.
            ix_mapping (list[int]): A list of indices for reversing the sequence to create the backward input and combine
            the corresponding hidden states for each direction.

        Returns:
            tuple[PackedSequence, Tensor, Tensor]:
                PackedSequence: The encoded output sequence as a packed sequence, where each timestep is a combination
                of forward and backward outputs based on the `merge_method`.
                    - Shape:
                        - (sum(X.batch_sizes), H) if MergeMethod.SUM
                        - (sum(X.batch_sizes, H*2) if MergeMethod.CONCAT
                Tensor: The hidden states of the forward RNN layers of shape `(n_layers, B, n_hidden)`.
                    - Shape: (n_layers, B, H)
                Tensor: The hidden states of the backward RNN layers of shape `(n_layers, B, n_hidden)`.
                    - Shape: (n_layers, B, H)

        """

        direction = self.get_direction(input)
        forward_input = input.data
        backward_input = input.data[ix_mapping]
        B = cast(int, input.batch_sizes[0].item())
        forward_h_n = torch.zeros((self.n_layers, B, self.n_hidden), device=input.data.device)
        backward_h_n = torch.zeros((self.n_layers, B, self.n_hidden), device=input.data.device)

        masks = self.init_masks(sum(input.batch_sizes), input.data.device)  # (n_layers, sum(input.batch_sizes), H)

        for layer in range(self.n_layers):
            forward_states, forward_h_n[layer] = direction(forward_input, self.forward_cells[layer])
            backward_states, backward_h_n[layer] = direction(backward_input, self.backward_cells[layer])

            if layer < self.n_layers - 1:
                if masks is not None:
                    forward_states = self.apply_dropout(forward_states, masks[layer])
                    backward_states = self.apply_dropout(backward_states, masks[layer, ix_mapping])
                forward_input = torch.cat((forward_states, backward_states[ix_mapping]), dim=1)
                backward_input = torch.cat((backward_states, forward_states[ix_mapping]), dim=1)

        if self.merge_method is MergeMethod.SUM:
            out = forward_states + backward_states[ix_mapping]
        elif self.merge_method is MergeMethod.CONCAT:
            out = torch.cat((forward_states, backward_states[ix_mapping]), dim=1)

        out_packed = PackedSequence(
            data=out,
            batch_sizes=input.batch_sizes,
            sorted_indices=input.sorted_indices,
            unsorted_indices=input.unsorted_indices,
        )

        # PackedSequence((sum(X.batch_sizes), H or H*2)), (n_layers, B, H), (n_layers, B, H)
        return out_packed, forward_h_n, backward_h_n

    def get_direction(self, input: PackedSequence) -> Callable[[Tensor, GRUCell], tuple[Tensor, Tensor]]:
        """
        Closure over the given PackedSequence.

        Args:
            input (PackedSequence): Packed input sequence to the first layer.

        Returns:
            Callable[[Tensor, GRUCell], tuple[Tensor, Tensor]]: Function that performs a forward pass for a layer and
            direction of the GRU.
        """

        def direction(X: Tensor, cell: GRUCell) -> tuple[Tensor, Tensor]:
            """
            Performs a forward pass through the given packed input tensor using the given GRUCell, which
            corresponds to a specific direction and layer within the overall GRU.

            Args:
                X (Tensor): Packed input tensor for the given cell.
                    - Shape: (sum(input.batch_sizes), n_emb) if layer == 0
                    - Shape: (sum(input.batch_sizes), H * 2) if layer >= 1
                cell (GRUCell): GRUCell corresponding to a specific layer and direction within the EncoderGRU

            Returns:
                states_cat (Tensor): All hidden states.
                    - Shape: (sum(input.batch_sizes), H)
                h_n (Tensor): Last hidden state for each example.
                    - Shape: (B, H)
            """

            if input.sorted_indices is None:
                raise ValueError("Sorted indices must be populated on input PackedSequence")
            B = cast(int, input.batch_sizes[0].item())
            T = len(input.batch_sizes)
            states = [torch.zeros((B, self.n_hidden), device=X.device)]
            h_n = torch.zeros((B, self.n_hidden), device=X.device)

            start = 0
            num_left = B
            for i, size in enumerate(input.batch_sizes):
                end = start + size
                X_t = X[start:end]
                state = cell(X_t, states[-1][:size])
                states.append(state)

                size_diff = size - input.batch_sizes[i + 1] if i < T - 1 else size
                if size_diff > 0:
                    h_n[input.sorted_indices[num_left - size_diff : num_left]] = state[-size_diff:]
                start = end
                if size_diff > 0:
                    num_left -= size_diff

            states_cat = torch.cat(states[1:], dim=0)  # (sum(batch_sizes), H)
            return states_cat, h_n

        return direction


class RNNEncoder(nn.Module):
    """
    RNNEncoder encodes a batch of input sequences using a multi-layer, bidirectional GRU.

    First uses an embedding layer to transform input tokens into dense vectors, then processes them with
    the GRU, before optionally applying a fully connected layer to each GRU layer's final hidden
    states.

    Args:
        params (RNNEncoderParams): The parameters for the encoder, including embedding size, hidden state size, number
                                   of layers, dropout rate, vocabulary size, padding index, and merge method.

    Attributes:
        params (RNNEncoderParams): Stores the encoder's configuration parameters.
        emb (nn.Embedding): Embedding layer to convert input tokens into embedding vectors.
        rnn (EncoderGRU): Multi-layer bidirectional GRU for sequence encoding.
        dropout (nn.Dropout): Dropout layer applied to embeddings during training.
        fc_hidden (nn.Linear, optional): An optional linear layer for operating on the final hidden states from
        each GRU layer. Used if `use_context` is True.
    """

    def __init__(self, params: RNNEncoderParams):
        super().__init__()
        self.params = params
        self.emb = nn.Embedding(params.vocab_size, params.n_emb, padding_idx=params.pad_ix)
        self.rnn = EncoderGRU(params.n_emb, params.n_hidden, params.n_layers, params.dropout, params.merge_method)
        self.dropout = nn.Dropout(params.dropout)
        if self.params.use_context:
            self.fc_hidden = nn.Linear(params.n_hidden * 2, params.n_hidden)

    def forward(self, input: Tensor, input_len: Tensor) -> tuple[Tensor, Tensor | None]:
        """
        Processes the batch of source sequences and returns the encoded outputs, as well as optional context
        derived from the final hidden states of each GRU layer.

        Args:
            input (Tensor): Padded batch of source sequences.
                - Shape: (B, max_src_len)
            input_len (Tensor): Lengths of each sequence in the batch.
                - Shape: (B,)

        Returns:
            tuple[Tensor, Optional[Tensor]]:
                out (Tensor): Padded tensor containing the bidirectional outputs at each time step
                from the last GRU layer.
                    - Shape:
                        - (B, max_src_len, H) if merge_method is SUM
                        - (B, max_src_len, H*2) if merge_method is CONCAT
                context (Tensor | None): Optional context tensor resulting from applying a linear
                layer to the final hidden states of each GRU layer.
                    - Shape: (n_layers, B, H)
        """

        input_emb = self.dropout(self.emb(input))  # (B, max_src_len, n_emb)
        # PackedSequence(sum(input_packed.batch_sizes), n_emb)
        input_packed = pack_padded_sequence(input_emb, input_len, batch_first=True, enforce_sorted=False)
        ix_mapping = RNNEncoder.get_ix_mapping(input_packed, len(input_len))

        # PackedSequence((sum(X.batch_sizes), H), (n_layers, B, H), (n_layers, B, H)
        out_packed, forward_h_n, backward_h_n = self.rnn(input_packed, ix_mapping)

        out = pad_packed_sequence(out_packed, batch_first=True)[
            0
        ]  # (B, max_src_len, H) if sum else (B, max_src_len, H*2)

        context = None
        if self.params.use_context:
            h_n_cat = torch.cat((forward_h_n, backward_h_n), dim=2)  # (n_layers, B, H*2)
            context = torch.tanh(self.fc_hidden(h_n_cat))  # (n_layers, B, H)

        return out, context

    @staticmethod
    def get_ix_mapping(packed: PackedSequence, batch_size: int) -> list[int]:
        """
        Helper method that traverses the given packed tensor and constructs a list that
        maps each index within the packed tensor to its corresponding index in the other
        direction.

        This is used by the bidirectional EncoderGRU to both:
            - Enable the processing of the given sequence in reverse order, and
            - Take the hidden states from each time step in the forward direction and combine
            them with their corresponding hidden states from the backward direction (and vice versa)

        Note that by "corresponding", I mean that the forward layer's hidden state after
        processing a certain input `T<n>` should be combined with the backward layer's hidden state
        after processing `T<n>`.

        Example of sequence [`T0`, `T1`, `T2`]
            - The backward layer will see [`T2`, `T1`, `TO`]
            - Returned `ix_mapping` is [2, 1, 0]
            - If the list of computed hidden states in the forward direction is denoted by [`F0`, `F1`, `F2`], and
            the list of computed hidden states in the backward direction is denoted by [`F2`, `F1, `F0`], then it should
            be clear to see how the `ix_mapping` works to combine them.
                - forward[0] and backward[2] each correspond to the hidden state after processing `T0`, so ix_mapping[0] = 2
                - forward[1] and backward[1] each correspond to the hidden state after processing `T1`, so ix_mapping[1] = 1
                - forward[2] and backward[0] each correspond to the hidden state after processing `T2`, so ix_mapping[2] = 0

        Args:
            packed (PackedSequence): Packed batch of source sequences.
                - Shape: (sum(packed.batch_sizes), n_emb)
            batch_size (int): Batch size.

        Returns:
            list[int]: Mapping of each index within the packed tensor to its corresponding index in the other direction.
        """

        per_example_ix: list[list[int]] = [[] for _ in range(batch_size)]
        start = 0
        for size in packed.batch_sizes:
            end = start + size.item()
            for i, _ in enumerate(packed.data[start:end]):
                per_example_ix[i].append(start + i)
            start = end

        ix_mapping = [-1] * packed.data.shape[0]
        for example_ix in per_example_ix:
            for i, ix in enumerate(example_ix):
                ix_mapping[ix] = example_ix[len(example_ix) - 1 - i]
        return ix_mapping


class DecoderKwargs(TypedDict):
    """
    A base class representing the contents of the additional keyword arguments required for a
    decoder's forward pass.

    Attributes:
        prev_hidden (Tensor): A Tensor representing the hidden states from the previous time step.
            - In the case of the decoder's first time step, then represents the encoder's final hidden states.
            - Shape: (n_layers, B, H)
    """

    prev_hidden: Tensor


T = TypeVar("T", bound=DecoderKwargs)


class RNNDecoder(nn.Module, ABC, Generic[T]):
    """
    Abstract base class for an RNN-based decoder.

    This class provides a generic framework for implementing RNN decoders, including embedding and dropout layers.
    Subclasses must implement the `init_kwargs` and `decode` methods to define decoder-specific initialization
    and decoding logic.

    Attributes:
        params (RNNDecoderParams): Stores the decoder's configuration parameters.
        emb (nn.Embedding): Embedding layer to convert input tokens into embedding vectors.
        dropout (nn.Dropout): Dropout layer applied to embeddings during training.

    Type Parameters:
        T (bound=DecoderKwargs): A type variable bound to `DecoderKwargs`, representing the structure of additional
                                 arguments passed to the decoder for its forward pass.
    """

    def __init__(self, params: RNNDecoderParams):
        super().__init__()
        self.params = params
        self.emb = nn.Embedding(params.vocab_size, params.n_emb, padding_idx=params.pad_ix)
        self.dropout = nn.Dropout(params.dropout)

    @abstractmethod
    def init_kwargs(self, src: Tensor, enc_outputs: Tensor, enc_hidden: Tensor) -> T:
        """
        Initializes the decoder-specific keyword arguments dictionary, to be passed into the
        decoder by RNNSeq2Seq.

        Args:
            src (Tensor): The source input tensor.
                - Shape: (B, max_src_len)
            enc_outputs (Tensor): The encoder's hidden states at each timestep.
                - Shape: (B, max_src_len, H or H*2)
            enc_hidden (Tensor): The encoder's final hidden states at each layer.
                - Shape: (n_layers, B, H)
        Returns:
            DecoderKwargs variant: Dictionary containing all arguments expected by the decoder's forward pass.
        """

        raise NotImplementedError

    @abstractmethod
    def decode(self, input: Tensor, kwargs: T) -> tuple[Tensor, T]:
        """
        Wrapper around the decoder's forward pass - after decoding, updates the `kwargs` dict to ready its contents for
        the next timestep's forward pass.

        Args:
            input (Tensor): The target input tensor (for a single timestep).
                - Shape: (B,)
            kwargs (DecoderKwargs): Dictionary containing the keyword arguments expected by the decoder's forward pass.

        Returns:
            tuple[Tensor, DecoderKwargs]:
                Tensor: Logits over the target vocabulary.
                    - Shape: (B, tgt_vocab_size)
                DecoderKwargs: Updated arguments for the next timestep.
        """

        raise NotImplementedError


class DecoderGRU(nn.Module, DropoutMixin):
    """
    A multi-layer GRU that processes a sequence of inputs and generates hidden states
    and outputs for each timestep. Dropout is applied between layers during training if enabled.

    Attributes:
        n_emb (int): Dimensionality of the input embeddings.
        n_hidden (int): Dimensionality of the GRU hidden states.
        n_layers (int): Number of GRU layers.
        cells (nn.ModuleList): A list of GRU cells, one for each layer.
        dropout (float): Dropout probability for inter-layer dropout.
    """

    def __init__(self, n_emb: int, n_hidden: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.cells = nn.ModuleList([GRUCell(n_emb if i == 0 else n_hidden, n_hidden) for i in range(n_layers)])
        self.dropout = dropout

    def _init_states(self, h_0: Tensor | None, B: int, device: torch.device) -> list[list[Tensor]]:
        """
        Initializes the hidden states for all GRU layers.

        If `h_0` is provided, it is used as the initial hidden states; otherwise, they are initialized to zeroes.

        Args:
            h_0 (Tensor | None): Initial hidden states, if they exist.
                - Shape: (n_layers, B, H)
            B (int): Batch size.
            device (torch.device): The device to allocate the hidden states.

        Returns:
            list[list[Tensor]]: A nested list of hidden states. Each sublist corresponds to a GRU layer and
                                contains the initial hidden state tensor for that layer.
        """

        return [
            [torch.zeros((B, self.n_hidden), device=device) if h_0 is None else h_0[i]] for i in range(self.n_layers)
        ]

    def forward(self, X: Tensor, h_0: Tensor | None) -> tuple[Tensor, Tensor]:
        """
        Performs a forward pass through a multi-layer GRU for the given input sequence, with optional inter-layer
        dropout.

        Note that although technically this method can handle arbitrarily many time steps, it will only ever
        get called with a single time step, because with teacher forcing, we have to decide at each time step
        what to feed in next.

        Args:
            X (Tensor): Input tensor.
                - Shape: (T, B, n_emb)
                    - (where T is the number of timesteps)
            h_0 (Tensor | None): Initial hidden states, if they exist.
                - Shape: (n_layers, B, H)

        Returns:
            tuple[Tensor, Tensor]:
                Tensor: Output tensor containing the hidden states for each timestep in the last layer.
                    - Shape: (T, B, H)
                Tensor: Final hidden states for each layer.
                    - Shape: (n_layers, B, H)
        """

        T, B, _ = X.shape

        states = self._init_states(h_0, B, X.device)
        masks = self.init_masks(B, X.device)
        for i in range(T):
            X_t = X[i]
            for layer in range(self.n_layers):
                state = self.cells[layer](X_t, states[layer][-1])
                states[layer].append(state)
                X_t = (
                    self.apply_dropout(state, masks[layer])
                    if masks is not None and layer < self.n_layers - 1
                    else state
                )

        output = torch.stack(states[-1][1:], 0)  # (T, B, H)
        h_n = torch.stack([layer[-1] for layer in states], 0)  # (n_layers, B, H)

        return output, h_n


class SutskeverRNNDecoder(RNNDecoder[DecoderKwargs]):
    """
    A simple RNN decoder based on "Sequence to Sequence Learning with Neural Networks" (Sutskever et al. 2014).

    Contains a multi-layer GRU to turn input embeddings into hidden states, and a fully-connected layer to transform
    those hidden states into logits.

    Note that for this decoder, it is mandatory that the encoder's final hidden states are passed as the initial hidden
    states for decoding. This is necessary because, unlike the other variants, this decoder has no attentional component.

    Attributes:
        rnn (DecoderGRU): A multi-layer GRU module that processes input embeddings.
        fc (nn.Linear): A fully connected layer for generating logits over the target vocabulary.
    """

    def __init__(self, params: RNNDecoderParams):
        super().__init__(params)
        self.rnn = DecoderGRU(self.params.n_emb, self.params.n_hidden, self.params.n_layers, self.params.dropout)
        self.fc = nn.Linear(self.params.n_hidden, self.params.vocab_size)

    def init_kwargs(self, src: Tensor, enc_outputs: Tensor, enc_hidden: Tensor) -> DecoderKwargs:
        kwargs: DecoderKwargs = dict(prev_hidden=enc_hidden)
        return kwargs

    def decode(self, input: Tensor, kwargs: DecoderKwargs) -> tuple[Tensor, DecoderKwargs]:
        output, hidden, _ = self(input, **kwargs)  # (B, tgt_vocab_size), (n_layers, B, H)
        kwargs["prev_hidden"] = hidden  # update prev hidden state for next timestep
        return output, kwargs

    def forward(self, input: Tensor, prev_hidden: Tensor) -> tuple[Tensor, Tensor, None]:
        """
        Processes a single input timestep through the decoder to generate vocabulary logits.

        This method takes an input token and the previous hidden states, and passes through an
        embedding layer, multi-layer GRU, and fully connected layer to produce logits over the target vocabulary.

        Args:
            input (Tensor): Input tensor (for a single timestep)
                - Shape: (B,)
            prev_hidden (Tensor): Tensor containing the hidden states from the previous timestep.
                - Shape: (n_layers, B, H)

        Returns:
            tuple[Tensor, Tensor, None]:
                Tensor: The output logits over the target vocabulary.
                    - Shape: (B, tgt_vocab_size)
                Tensor: The updated hidden states of the decoder RNN.
                    - Shape: (n_layers, B, H)
                None: Placeholder for attentional outputs, included for interface compatibility.
        """

        input = input.unsqueeze(0)  # (1, B)
        input_emb = self.dropout(self.emb(input))  # (1, B, n_emb)
        out, h_n = self.rnn(input_emb, prev_hidden)  # (1, B, H), (n_layers, B, H)

        fc_out = self.fc(out)  # (1, B, H) @ (H, vocab_size) = (1, B, vocab_size)
        fc_out = fc_out.squeeze(0)  # (B, vocab_size)

        return fc_out, h_n, None


class DotScorer(nn.Module):
    """
    A dot-product attention scorer.
    """

    def forward(self, queries: Tensor, keys: Tensor) -> Tensor:
        """
        Computes raw attention scores using the dot product between a query vector and a set of key vectors.

        Args:
            queries (Tensor): Query tensor.
                - Shape: (B, num_queries, H)
                    - num_queries is always 1, since the only query is the previous hidden state.
            keys (Tensor): Keys tensor (encoder outputs).
                - Shape: (B, max_src_len, H)

        Returns:
            Tensor: Raw attention scores, representing the learned similarity between the query and each key.
                - Shape: (B, num_queries, max_src_len)
        """

        return queries @ keys.permute(0, 2, 1)  # (B, 1, H) @ (B, H, max_src_len) = (B, 1, max_src_len)


class GeneralScorer(nn.Module):
    """
    A general attention scorer (from Luong et al.)

    Attributes:
        W (nn.Linear): Linear layer to transform queries into desired dimensionality.
    """

    def __init__(self, dec_n_hidden: int, enc_n_hidden: int):
        super().__init__()
        self.W = nn.Linear(enc_n_hidden, dec_n_hidden, bias=False)

    def forward(self, queries: Tensor, keys: Tensor) -> Tensor:
        """
        Computes raw attention scores using the dot product between a transformed query vector and a set of key vectors.

        Args:
            queries (Tensor): Query tensor.
                - Shape: (B, num_queries, H)
                    - num_queries is always 1, since the only query is the previous hidden state.
            keys (Tensor): Keys tensor (encoder outputs).
                - Shape: (B, max_src_len, H)

        Returns:
            Tensor: Raw attention scores, representing the learned similarity between the query and each key.
                - Shape: (B, num_queries, max_src_len)
        """

        keys = self.W(keys)  # (B, max_src_len, encH) @ (encH, decH) = (B, max_src_len, decH)
        return queries @ keys.permute(0, 2, 1)  # (B, 1, decH) @ (B, decH, max_src_len) = (B, 1, max_src_len)


class AdditiveScorer(nn.Module):
    """
    An additive attention scorer - i.e. Bahdanau attention (equivalent to Luong's 'concat' scoring).

    Attributes:
        - W_q (nn.Linear): Linear layer to transform queries into desired dimensionality.
        - W_k (nn.Linear): Linear layer to transform keys into desired dimensionality.
        - w_v (nn.Linear): Linear layer to compress raw "energy" vectors into a single score per key.
    """

    def __init__(self, dec_n_hidden: int, enc_n_hidden: int, n_hidden: int):
        super().__init__()
        self.W_q = nn.Linear(dec_n_hidden, n_hidden, bias=False)
        self.W_k = nn.Linear(enc_n_hidden, n_hidden, bias=False)
        self.w_v = nn.Linear(n_hidden, 1, bias=False)

    def forward(self, queries: Tensor, keys: Tensor) -> Tensor:
        """
        Computes raw attention scores using additive attention.

        Steps:
            - Queries and keys tensors each pass through a linear layer so their dimensions match.
            - Resulting tensors are added together and fed through a tanh activation to compute raw "energy" scores.
            - Those scores are then passed through another linear layer to obtain a single score per key.

        Args:
            queries: Query tensor.
                - Shape: (B, num_queries, H), where num_queries is always 1
            keys: Keys tensor (encoder outputs).
                - Shape: (B, max_src_len, H*2)
                    - Technically could just be H, doesn't ultimately matter, but if within a Bahdanau configuration,
                    will be H*2 b/c encoder outputs are concatenation of forward and backward hidden states.

        Returns:
            Tensor: Raw attention scores, representing the learned similarity between the query and each key.
                - Shape: (B, num_queries, max_src_len)
        """

        queries = self.W_q(queries)  # (B, 1, decH) @ (decH, H) = (B, 1, H)
        keys = self.W_k(keys)  # (B, max_src_len, encH) @ (encH, H) = (B, max_src_len, H)

        queries = queries.unsqueeze(2)  # (B, 1, 1, H)
        keys = keys.unsqueeze(1)  # (B, 1, max_src_len, H)

        energy = torch.tanh(
            queries + keys
        )  # (B, 1, 1, H) + (B, 1, max_src_len, H) = (B, 1, max_src_len, H) (broadcasting duplicates query across max_src_len)

        return self.w_v(energy).squeeze(
            -1
        )  # (B, 1, max_src_len, H) @ (H, 1) = (B, 1, max_src_len, 1) -> (B, 1, max_src_len)


class ScaledDotScorer(nn.Module):
    """
    A scaled dot-product attention scorer.
    """

    def __init__(self, n_hidden: int):
        super().__init__()
        self.register_buffer("scale_factor", torch.sqrt(torch.tensor([n_hidden])), persistent=False)

    def forward(self, queries: Tensor, keys: Tensor) -> Tensor:
        """
        Computes raw attention scores using the dot product between a query vector and a set of key vectors,
        scaled by the dimensionality of those vectors.

        Args:
            queries (Tensor): Query tensor.
                - Shape: (B, num_queries, H)
                    - num_queries is always 1, since the only query is the previous hidden state.
            keys (Tensor): Keys tensor (encoder outputs).
                - Shape: (B, max_src_len, H)

        Returns:
            Tensor: Raw attention scores, representing the learned similarity between the query and each key.
                - Shape: (B, num_queries, max_src_len)
        """

        raw_scores = queries @ keys.permute(0, 2, 1)  # (B, 1, H) @ (B, H, max_src_len) = (B, 1, max_src_len)
        return raw_scores / self.get_buffer("scale_factor")


class Attention(nn.Module):
    """
    Implements an attention mechanism for sequence-to-sequence models.

    This module computes attention scores between query and key vectors using a specified scoring method,
    applies a mask to the scores, and generates a context vector by weighting the value vectors with the
    computed attention distribution.

    Args:
        method (ScoreMethod): The scoring method to use for computing attention scores. Options include:
            - `ScoreMethod.DOT`: Dot-product attention.
            - `ScoreMethod.GENERAL`: General attention with a learned weight matrix.
            - `ScoreMethod.ADDITIVE`: Additive (or Bahdanau) attention.
            - `ScoreMethod.SCALED_DOT`: Scaled dot-product attention.
        n_hidden (int): Dimensionality of the decoder's hidden states.
        enc_n_hidden (int): Dimensionality of the encoder's hidden states.

    Attributes:
        scorer (nn.Module): The scoring module used to compute the raw attention scores.
    """

    def __init__(self, method: ScoreMethod, n_hidden: int, enc_n_hidden: int):
        super().__init__()
        match method:
            case ScoreMethod.DOT:
                self.scorer: nn.Module = DotScorer()
            case ScoreMethod.GENERAL:
                self.scorer = GeneralScorer(n_hidden, enc_n_hidden)
            case ScoreMethod.ADDITIVE:
                self.scorer = AdditiveScorer(n_hidden, enc_n_hidden, n_hidden)
            case ScoreMethod.SCALED_DOT:
                self.scorer = ScaledDotScorer(n_hidden)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor):
        """
        Computes the attention distribution over the queries and keys tensors, and then uses it to produce context
        vectors by performing a weighted sum on the values tensor.

        Details:
            - Attention distribution has shape (B, 1, max_src_len), values tensor has shape (B, max_src_len, H*2).
            - By batch multiplying the attention distribution by the values tensor, obtain a tensor of shape (B, 1, H*2).
            - Each i-th value in an example's context vector is now a weighted sum of the corresponding i-th values
            in the encoder output vectors, present for each timestep.
            - In this way, the model can "choose", for each value in the resulting context vector, based on the
            attention distribution, to magnify the impact that certain timesteps have on the resulting value, and
            minimize the impact of other timesteps.

        Note that the keys and values tensors will always be the same - they are both the encoder outputs.

        Args:
            queries (Tensor): Query tensor, used to compute the attention scores.
                - Shape: (B, num_queries, H), where num_queries is always 1
            keys (Tensor): Keys tensor, used to compute the attention scores.
                - Shape: (B, max_src_len, H)
                    - (H*2 if MergeMethod.CONCAT)
            values (Tensor): Values tensor, used to compute the context tensor.
                - Shape: (B, max_src_len, H)
                    - (H*2 if MergeMethod.CONCAT)
            mask (Tensor): A mask for the source sequence, in which padded positions are set to zero to exclude them
            from the attention computation.
                - Shape: (B, max_src_len)

        Returns:
            tuple[Tensor, Tensor]:
                Tensor: The context vectors, each computed as a weighted sum of the values vectors, using the attention
                distribution as the weights.
                    - Shape: (B, num_queries, H)
                Tensor: The attention weights.
                    - Shape: (B, num_queries, max_src_len)
        """

        scores = self.scorer(queries, keys)  # (B, 1, max_src_len)
        scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e10)
        attention = F.softmax(scores, dim=2)

        context = torch.bmm(attention, values)  # (B, 1, max_src_len) @ (B, max_src_len, H) = (B, 1, H)
        return context, attention  # (B, 1, H), (B, 1, max_src_len)


class BahdanauKwargs(DecoderKwargs):
    """
    A class representing the contents of the additional keyword arguments required for the BahdanauDecoder's
    forward pass.

    Attributes:
        enc_outputs (Tensor): The encoder outputs (hidden states at each timestep from the last encoder GRU layer)
            - Shape: (B, max_src_len, H or H*2)
        mask (Tensor): A mask for the source sequence, in which padded positions are set to zero to exclude them from
        the attention computation.
            - Shape: (B, max_src_len)
    """

    enc_outputs: Tensor
    mask: Tensor


class BahdanauRNNDecoder(RNNDecoder[BahdanauKwargs]):
    """
    A more advanced RNN decoder based on "Neural Machine Translation by Jointly Learning to Align and
    Translate" (Bahdanau et al. 2014).

    This module, unlike the SutskeverRNNDecoder, includes an attentional component - the encoder outputs are weighted
    by their learned relevance to the previous hidden state to compute "context" vectors, which are then passed to
    both the rnn module and the fully-connected layer.

    Attributes:
        attention (Attention): A module used to compute context vectors based on the encoder outputs and the decoder's
        previous hidden state.
        rnn (DecoderGRU): A GRU-based RNN module that processes the concatenation of input embeddings and context vectors.
        fc (nn.Linear): A fully connected layer mapping the RNN outputs, context vectors, and input embeddings to logits
        over the target vocabulary.
    """

    def __init__(self, params: RNNDecoderParams):
        super().__init__(params)
        if params.att_score_method is None:
            raise ValueError("Attention scoring method must be defined for Bahdanau decoder")
        self.attention = Attention(
            method=params.att_score_method, n_hidden=params.n_hidden, enc_n_hidden=params.n_hidden * 2
        )
        self.rnn = DecoderGRU(
            params.n_emb + (params.n_hidden * 2), params.n_hidden, self.params.n_layers, self.params.dropout
        )
        self.fc = nn.Linear(params.n_hidden + (params.n_hidden * 2) + params.n_emb, params.vocab_size)

    def init_kwargs(self, src: Tensor, enc_outputs: Tensor, enc_hidden: Tensor | None) -> BahdanauKwargs:
        if enc_hidden is None:
            enc_hidden = torch.zeros((self.params.n_layers, src.shape[0], self.params.n_hidden), device=src.device)
        mask = src != self.params.pad_ix  # generate mask to exclude pad tokens from attention
        kwargs: BahdanauKwargs = dict(prev_hidden=enc_hidden, enc_outputs=enc_outputs, mask=mask)
        return kwargs

    def decode(self, input: Tensor, kwargs: BahdanauKwargs) -> tuple[Tensor, BahdanauKwargs]:
        output, hidden, _ = self(input, **kwargs)  # (B, tgt_vocab_size), (n_layers, B, H), ignore attention
        kwargs["prev_hidden"] = hidden
        return output, kwargs

    def forward(
        self, input: Tensor, prev_hidden: Tensor, enc_outputs: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Processes a single input timestep through the decoder to generate vocabulary logits.

        This method computes input embeddings and context vectors, concatenates them, and passes them as input to the
        GRU module to obtain the hidden state for the current timestep. The hidden state is then concatenated with the
        embeddings and context vectors and fed through a linear layer, in order to produce output logits over the target
        vocabulary.

        Args:
            input (Tensor): Input tensor (for a single timestep)
                - Shape: (B,)
            prev_hidden (Tensor): Tensor containing the hidden states from the previous timestep.
                - Shape: (n_layers, B, H)
            enc_outputs (Tensor): Encoder outputs (concatenated hidden states from final encoder layer) tensor.
                - Shape: (B, max_src_len, H*2)
            mask (Tensor): A mask for the source sequence, in which padded positions are set to zero to exclude them
            from the attention computation.
                - Shape: (B, max_src_len)

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                Tensor: The output logits over the target vocabulary.
                    - Shape: (B, tgt_vocab_size)
                Tensor: The updated hidden states of the decoder RNN.
                    - Shape: (n_layers, B, H)
                Tensor: The attention distribution / weights used to construct the context vectors.
                    - Shape: (B, max_src_len)
        """

        input = input.unsqueeze(0)  # -> (1, B)
        input_emb = self.dropout(self.emb(input))  # -> (1, B, n_emb)

        query = prev_hidden[-1].unsqueeze(1)  # (B, 1, H) (just one query)
        context, att = self.attention(query, enc_outputs, enc_outputs, mask)  # (B, 1, H*2), (B, 1, max_src_len)
        context = context.permute(1, 0, 2)  # (1, B, H*2)

        rnn_input = torch.cat((input_emb, context), dim=2)  # (1, B, n_emb + H*2)

        rnn_out, h_n = self.rnn(rnn_input, prev_hidden)  # -> (1, B, H), (n_layers, B, H)

        context = context.squeeze(0)  # (B, H*2)
        input_emb = input_emb.squeeze(0)  # (B, n_emb)

        fc_input = torch.cat((rnn_out[0], context, input_emb), dim=1)  # (B, H + (H*2) + n_emb)

        # (B, H + (H*2) + n_emb) @ (H + (H*2) + n_emb, tgt_vocab_size) = (B, tgt_vocab_size)
        fc_out = self.fc(fc_input)

        return fc_out, h_n, att.squeeze(1)  # (B, tgt_vocab_size), (n_layers, B, H), (B, max_src_len)


class LuongKwargs(BahdanauKwargs):
    """
    A class representing the contents of the additional keyword arguments required for the LuongDecoder's
    forward pass.

    Attributes:
        prev_mod_hidden (Tensor): The computed "attentional hidden state" from the previous time step in the decoder.
            - Shape: (B, H)
    """

    prev_mod_hidden: Tensor | None


class LuongRNNDecoder(RNNDecoder[LuongKwargs]):
    """
    A more refined attentional RNN decoder based on "Effective Approaches to Attention-based Neural Machine Translation"
    (Luong et al. 2015).

    This module is similar to the BahdanauRNNDecoder in that it utilizes an attention module, but differs greatly in its
    implementation. Instead of calculating attention based off the decoder's previous hidden state, this decoder computes
    the new hidden state first, before using it to compute the context vectors. The context vectors are then combined
    with the hidden state and transformed into the "attentional hidden state", which is then used to obtain the output
    logits over the target vocabulary.

    Optionally, "input feeding" can be enabled, which ensures that, in addition to input embeddings, the attentional
    hidden state from the previous time step is also fed as input into the rnn module for the current time step.

    Additionally, use_context in the config can be toggled on or off, to control whether the initial hidden states for
    the GRU are constructed from the final hidden states of the encoder.

    Attributes:
        rnn (DecoderGRU): A GRU-based RNN module that processes input embeddings and, optionally, the attentional hidden
        state from the previous time step.
        attention (Attention): A module used to compute context vectors based on the encoder outputs and the computed
        hidden state for the current time step.
        W_c (nn.Linear): A linear layer mapping the computed hidden state and context vectors to the attentional hidden
        state.
        fc (nn.Linear): A linear layer mapping the computed attentional hidden state to logits over the target
        vocabulary.
    """

    def __init__(self, params: RNNDecoderParams):
        super().__init__(params)
        if params.att_score_method is None:
            raise ValueError("Attention scoring method must be defined for Luong decoder")
        rnn_input_size = params.n_hidden + params.n_emb if self.params.input_feeding else params.n_emb
        self.rnn = DecoderGRU(rnn_input_size, params.n_hidden, params.n_layers, dropout=params.dropout)
        self.attention = Attention(
            method=params.att_score_method, n_hidden=params.n_hidden, enc_n_hidden=params.n_hidden
        )
        self.W_c = nn.Linear(params.n_hidden * 2, params.n_hidden)
        self.fc = nn.Linear(params.n_hidden, params.vocab_size)

    def init_kwargs(self, src: Tensor, enc_outputs: Tensor, enc_hidden: Optional[Tensor]) -> LuongKwargs:
        if enc_hidden is None:
            enc_hidden = torch.zeros((self.params.n_layers, src.shape[0], self.params.n_hidden), device=src.device)

        mod_hidden = None
        if self.params.input_feeding:
            mod_hidden = torch.zeros((src.shape[0], self.params.n_hidden), device=src.device)

        mask = src != self.params.pad_ix  # generate mask to exclude pad tokens from attention
        kwargs: LuongKwargs = dict(
            prev_hidden=enc_hidden, enc_outputs=enc_outputs, mask=mask, prev_mod_hidden=mod_hidden
        )
        return kwargs

    def decode(self, input: Tensor, kwargs: LuongKwargs) -> tuple[torch.Tensor, LuongKwargs]:
        output, (hidden, mod_hidden), _ = self(input, **kwargs)  # (B, tgt_vocab_size), (n_layers, B, H), (B, H)
        kwargs["prev_hidden"] = hidden
        if self.params.input_feeding:
            kwargs["prev_mod_hidden"] = mod_hidden
        return output, kwargs

    def forward(
        self, input: Tensor, prev_hidden: Tensor, enc_outputs: Tensor, mask: Tensor, prev_mod_hidden: Tensor | None
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        """
        Processes a single input timestep through the decoder to generate vocabulary logits.

        This method passes input embeddings to the GRU module to compute the hidden state for the current timestep, and
        then uses that hidden state to calculate context vectors based off the encoder outputs. The hidden state and
        context vectors are then concatenated and fed through a linear layer and tanh activation to compute the
        "attentional hidden state", which is finally fed through another linear layer to produce output logits over the
        target vocabulary.

        If input feeding is enabled, then the attentional hidden state from the previous time step will also be passed
        to the GRU module as input.

        Args:
            input (Tensor): Input tensor (for a single timestep)
                - Shape: (B,)
            prev_hidden (Tensor): Tensor containing the hidden states from the previous timestep.
                - Shape: (n_layers, B, H)
            enc_outputs (Tensor): Encoder outputs (hidden states from final encoder layer) tensor.
                - Shape: (B, max_src_len, H)
            mask (Tensor): A mask for the source sequence, in which padded positions are set to zero to exclude them
            from the attention computation.
                - Shape: (B, max_src_len)
            prev_mod_hidden (Tensor | None): The attentional hidden state computed at the previous timestep - only
            present if input feeding is enabled.
                - Shape: (B, H)

        Returns:
            tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
                Tensor: The output logits over the target vocabulary.
                    - Shape: (B, tgt_vocab_size)
                tuple[Tensor, Tensor]:
                    Tensor: The updated hidden states of the decoder RNN.
                        - Shape: (n_layers, B, H)
                    Tensor: The attentional hidden state for the current time step.
                        - Shape: (B, H)
                Tensor: The attention distribution / weights used to construct the context vectors.
                    - Shape: (B, max_src_len)
        """

        input = input.unsqueeze(0)  # (1, B)
        input_emb = self.dropout(self.emb(input))  # (1, B, n_emb)

        if prev_mod_hidden is not None:
            rnn_input = torch.cat((prev_mod_hidden.unsqueeze(0), input_emb), dim=2)  # (1, B, H + n_emb)
        else:
            rnn_input = input_emb

        rnn_out, h_n = self.rnn(rnn_input, prev_hidden)  # (1, B, H), (n_layers, B, H)
        hidden = rnn_out.squeeze(0)  # (B, H)

        query = hidden.unsqueeze(1)  # (B, 1, H) (just one query)
        context, att = self.attention(query, enc_outputs, enc_outputs, mask)  # (B, 1, H), (B, 1, max_src_len)

        concat = torch.cat((context.squeeze(1), hidden), dim=1)  # (B, H * 2)
        mod_hidden = torch.tanh(self.W_c(concat))  # (B, H * 2) @ (H * 2, H) = (B, H)

        fc_out = self.fc(mod_hidden)  # (B, H) @ (H, tgt_vocab_size) = (B, tgt_vocab_size)

        return (
            fc_out,
            (h_n, mod_hidden),
            att.squeeze(1),
        )  # (B, tgt_vocab_size), ((n_layers, B, H), (B, H)), (B, max_src_len)


class RNNSeq2Seq(nn.Module):
    """
    An RNN-based sequence-to-sequence model, used for neural machine translation.

    This class combines an encoder and a decoder, both implemented as RNN modules, to process
    a source sequence and then generate a target sequence. The decoder supports teacher forcing, for
    improved training dynamics.

    Attributes:
        encoder (RNNEncoder): The RNN-based encoder module that processes source sequences.
        decoder (RNNDecoder): The RNN-based decoder module that generates target sequences.
    """

    def __init__(self, encoder: RNNEncoder, decoder: RNNDecoder):
        super().__init__()
        if encoder.params.n_hidden != decoder.params.n_hidden or encoder.params.n_layers != decoder.params.n_layers:
            raise ValueError("Invalid configuration for RNNSeq2Seq: mismatched encoder and decoder")

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: Tensor, src_len: Tensor, tgt: Tensor, teacher_forcing_rate: float = 0.0) -> Tensor:
        """
        Processes a source sequence and generates a target sequence using the provided encoder and decoder.

        Args:
            src (Tensor): Batched source sequences tensor.
                - Shape: (B, max_src_len)
            src_len (Tensor): Lengths of each source sequence in the batch.
                - Shape: (B,)
            tgt (Tensor): Batched target sequences tensor.
                - Shape: (max_tgt_len, B)
            teacher_forcing_rate (float, optional): Probability of using teacher forcing during training. Defaults to 0.0.

        Returns:
            Tensor: Logits over the target vocabulary, for each timestep and example.
                - Shape: (max_tgt_len, B, tgt_vocab_size)
        """

        enc_outputs, enc_hidden = self.encoder(src, src_len)  # (B, max_src_len, H or H*2), (n_layers, B, H)

        max_tgt_len = tgt.shape[0]
        B = tgt.shape[1]
        # (max_tgt_len, B, tgt_vocab_size)
        outputs = torch.zeros((max_tgt_len, B, self.decoder.params.vocab_size), device=src.device)

        kwargs = self.decoder.init_kwargs(src, enc_outputs, enc_hidden)

        input = tgt[0]  # (B)
        for i in range(1, max_tgt_len):
            output, kwargs = self.decoder.decode(input, kwargs)  # (B, tgt_vocab_size)
            outputs[i] = output
            teacher_force = teacher_forcing_rate > 0.0 and random.random() < teacher_forcing_rate
            input = tgt[i] if teacher_force else torch.argmax(output, dim=1)  # (B)

        return outputs  # (max_tgt_len, B, tgt_vocab_size)
