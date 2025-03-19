from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Self, Union, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from torchnmt.config import BeamConfig
from torchnmt.data import NMTDataset
from torchnmt.model import RNNDecoder

from .beam import BeamSearchResult


@dataclass
class RNNBeamSearchNode:
    """
    A node in the beam search process for an RNN-based model, representing a token at a certain timestep
    within a predicted sequence (i.e., a single beam).

    These nodes are added to a priority queue, and are ordered in ascending order according to the `val` attribute - why?
        - At each iteration of the beam search process, for each source example, we want to dequeue the most likely
        (highest probability) target sequences generated so far by the decoder.
        - A node's likelihood is determined not just by the probability given by the decoder at that time step, but by
        multiplying it by the probabilities of all the nodes that came before it in the sequence.
        - This conditional probability is stored in the `logp` attribute - why?
            - Multiplying a chain of small probabilities together can lead to precision issues, so instead, store the
            sum of their log probabilities:
                - log(P0 * P1 * P2) = log(P0) + log(P1) + log(P2)
            - For each new node, to accurately set its conditional probability, must set the `logp` attribute to be
            equal to the previous node's `logp`, plus the log probability of the newly predicted token.
        - When comparing two nodes' `logp` attributes, they will always be negative, but the closer to zero the better:
            - Node A -> log(0.6) + log(0.4) + log(0.3) = -1.14
            - Node B -> log(0.6) + log(0.4) + log(0.1) = -1.62
        - However, comparing the conditional log probabilities straight up will penalize longer sequences, so have to
        perform length normalization and divide the `logp` by the length of the sequence so far to obtain the `val`
        attribute.
        - Additionally, b/c the priority queue is implemented by a min-heap, also have to multiply the normalized logp
        by -1, so that the higher probability nodes now have lower `val`s, and are thus dequeued first.

    Attributes:
        hidden (Tensor): The hidden state of the RNN at the current step.
            - Shape: (n_layers, H)
        prev (RNNBeamSearchNode | None): A reference to the previous node in the sequence. `None` if this is the start node.
        ix (Tensor): The index tensor representing the token at the current step.
            - Shape: (1,)
        logp (float): The log probability of the current token, given the context of the previous tokens in the sequence.
        att (Tensor | None): The attention weights at the current step. `None` if attention is not used.
            - Shape: (max_src_len,)
        length (int): The length of the sequence ending at this node.
        val (float): The evaluation metric used for sorting nodes during beam search.
    """

    hidden: Tensor
    prev: Self | None
    ix: Tensor
    logp: float
    att: Tensor | None
    length: int = field(init=False)
    val: float = field(init=False)

    def __post_init__(self):
        if self.prev is None:
            self.length = 1
            self.val = 0
        else:
            self.length = self.prev.length + 1
            self.val = -(self.logp / (self.length - 1))

    def __lt__(self, nxt: Self) -> bool:
        return self.val < nxt.val

    def next_input_data(self) -> tuple[Tensor, ...]:
        """
        Returns the data required to construct the input for the next time step, based off this node.

        Returns:
            tuple[Tensor, Tensor]:
                Tensor: Index of current token.
                    - Shape: (1,)
                Tensor: Hidden state.
                    - Shape: (n_layers, H)
        """

        return self.ix, self.hidden


@dataclass
class LuongInputFeedingNode(RNNBeamSearchNode):
    mod_hidden: Tensor

    def next_input_data(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.ix, self.hidden, self.mod_hidden


class RNNBeamSearcher:
    """
    Batched beam search implementation compatible with RNN models.

    Attributes:
        - decoder (RNNDecoder): The RNN-based decoder module that generates the target sequences.
        - dataset (NMTDataset): NMTDataset instance providing the start and end token indices.
        - device (str): The device the decoder module is located on.
        - config (BeamConfig): Beam search configuration object.
    """

    def __init__(self, decoder: RNNDecoder, dataset: NMTDataset, device: str, config: BeamConfig):
        self.decoder = decoder
        self.dataset = dataset
        self.device = device
        self.config = config

    def _create_start_node(self, input: Tensor, decoder_kwargs: dict, batch_idx: int) -> RNNBeamSearchNode:
        """
        Creates a new start node for a given example within a batch.

        Args:
            input (Tensor): Input tensor for batched decoding.
                - Shape: (1, B)
            decoder_kwargs (dict): Keyword arguments required by the decoder's forward pass.
            batch_idx (int): Index of example within the batch.

        Returns:
            RNNBeamSearchNode: The created start node.
        """

        node_kwargs = dict(
            hidden=decoder_kwargs["prev_hidden"][:, batch_idx], prev=None, ix=input[:, batch_idx], logp=0, att=None
        )
        if self.decoder.params.input_feeding:
            node_kwargs["mod_hidden"] = decoder_kwargs["prev_mod_hidden"][batch_idx]
            return LuongInputFeedingNode(**node_kwargs)
        return RNNBeamSearchNode(**node_kwargs)

    def _create_node(
        self,
        ix: Tensor,
        logp: float,
        prev_node: RNNBeamSearchNode,
        out_hidden: Tensor,
        batch_att: Union[tuple[Tensor, Tensor], Tensor],  # ??
        batch_idx: int,
    ):
        """
        Creates a new node to be enqueued for the beam search process.

        Args:
            ix (Tensor): Token index tensor.
                - Shape: (1)
            logp (float): The log probability of the token, given the context of the sequence preceding it.
            prev_node (RNNBeamSearchNode): The previous node in the sequence.
            out_hidden (Tensor): Newly computed hidden states for the batch.
                - Shape: (n_layers, B, H)
            batch_att (Tensor): Attention weights for the batch.
                - Shape: (B, max_src_len)
            batch_idx (int): Index of the example within the batch

        Returns:
            RNNBeamSearchNode: The created node.
        """

        new_logp = logp + prev_node.logp  # add new logp to existing, to maintain conditional probability
        att = None if batch_att is None else batch_att[batch_idx]
        if isinstance(out_hidden, tuple):
            hidden = out_hidden[0][:, batch_idx]
            if self.decoder.params.input_feeding:
                mod_hidden = out_hidden[1][batch_idx]
                return LuongInputFeedingNode(
                    ix=ix, logp=new_logp, prev=prev_node, hidden=hidden, att=att, mod_hidden=mod_hidden
                )
        else:
            hidden = out_hidden[:, batch_idx]
        return RNNBeamSearchNode(ix=ix, logp=new_logp, prev=prev_node, hidden=hidden, att=att)

    def _assemble_batch(self, B: int, prev_nodes: list[RNNBeamSearchNode], kwargs: dict) -> tuple[Tensor, dict]:
        """
        Assembles a batch from the given list of nodes, to be passed to the decoder as input.

        Args:
            B (int): Batch size.
            prev_nodes (list[RNNBeamSearchNode]): List of nodes, each representing the end of the most likely
            target sequence generated so far for their respective example.
            kwargs (dict): Arguments to pass to the decoder.

        Returns:
            tuple[Tensor, dict]:
                Tensor: Input tensor, containing the token indices representing the output from the previous timestep,
                from the perspective of the decoder.
                    - Shape: (B)
                dict: Arguments to pass to the decoder, updated for the current timestep.
        """

        data = [prev_nodes[batch_id].next_input_data() for batch_id in range(B)]  # [((1), (n_layers, H))...]
        if "prev_mod_hidden" in kwargs and kwargs["prev_mod_hidden"] is not None:
            input, hidden, mod_hidden = zip(*data, strict=True)
            kwargs["prev_mod_hidden"] = torch.stack(mod_hidden, dim=0).to(self.device)
        else:
            input, hidden = zip(*data, strict=True)
        input_tensor = torch.cat(input).to(self.device)  # (B)
        kwargs["prev_hidden"] = torch.stack(hidden, dim=1).to(self.device)  # (n_layers, B, H)
        return input_tensor, kwargs

    @torch.no_grad()
    def search(
        self,
        src: Tensor,
        enc_outputs: Tensor,
        enc_hidden: Tensor,
        config: BeamConfig | None = None,
    ) -> list[list[BeamSearchResult]]:
        """
        Performs a batched beam search to generate an approximation of the most likely target sequences for each source
        example in the given batch.

        Details:
            - For each source example in the batch, a priority queue of RNNBeamSearchNode objects is maintained, each
            representing a token (and its hidden state) at a certain timestep within a predicted sequence (i.e., a
            single beam).
            - These nodes are ordered so that the most likely target sequence for each source sequence is always located
            at the head of the queue.
            - Initially, only start nodes (each containing the start token) are enqueued.
            - Then, while there are some source examples that are "unfinished":
                - Dequeue the most likely node for each example.
                - From those nodes, assemble a batch (join node indices, hidden states into tensors).
                - Pass the constructed batch to the decoder.
                - Use the decoder's output to calculate log probabilities for the next tokens in the sequences.
                - For each unfinished example, enqueue `n` more nodes onto its priority queue, representing the `n` most
                likely next tokens as determined by the decoder.
                    - Value of `n` corresponds to `beam_width` configuration parameter.
            - What makes an example finished?
                - `n_best` configuration parameter controls how many complete target sequences should be generated for
                each example - once that number of sequences have been stored, the example is finished.
                - What makes a target sequence complete?
                    - If its last node contains the end token.
                - Alternatively, if there are still outstanding complete target sequences to be generated for an example,
                but it's already enqueued the maximum allowable amount of nodes (max_src_len * 10), then it is forced
                to be finished.
            - Once all examples are finished, for each example:
                - If there are less than `n_best` generated sequences, then dequeue the most likely (non-complete)
                sequences from the priority queue as needed.
                - For each sequence:
                    - Traverse backwards through the linked list formed by the nodes in the sequence and construct the
                    list of tokens and, optionally, the attention weights tensor, contained within the sequence.
                    - Use the target tokenizer to convert the tokens to a string.
                    - Save data within a BeamSearchResult.
                - Sort the BeamSearchResult objects in ascending order of their `val` attribute - i.e. most to least
                likely.

        Args:
            src (Tensor): The source sequence tensor, passed to the encoder as input.
                - Shape: (B, max_src_len)
            enc_outputs (Tensor): The outputs from the encoder's last layer.
                - Shape: (B, max_src_len, H)
            enc_hidden (Tensor): The final hidden states from each layer in the encoder.
                - Shape: (n_layers, B, H)
            config (BeamConfig, optional): Beam search config to override the class-level config.

        Returns:
            list[list[BeamSearchResult]]: List of search results for each example in the batch.
        """

        self.decoder.eval()

        config = config or self.config
        beam_width, n_best = config.beam_width, config.n_best

        B, max_src_len, H = enc_outputs.shape

        # list of priority queues (for each example in batch)
        nodes: list[list[RNNBeamSearchNode]] = [[] for _ in range(B)]
        # list of end nodes generated by decoder (for each example in batch)
        end_nodes: list[list[RNNBeamSearchNode]] = [[] for _ in range(B)]
        # how many nodes have been enqueued (for each example in batch)
        steps = [0 for _ in range(B)]
        # which examples have been finished - (generated n_best sequences, or max_steps reached)
        finished: set[int] = set()
        # max number of nodes that can be enqueued for an example
        max_steps = max_src_len * 10
        # list of nodes representing the output of the previous timestep (from the perspective of the decoder)
        # used to construct the inputs for a decoder step
        prev_nodes: list[RNNBeamSearchNode] = [None for _ in range(B)]  # type: ignore

        start_ix = self.dataset.tgt_specials["start"]
        eos_ix = self.dataset.tgt_specials["end"]

        input = torch.tensor([start_ix]).repeat(1, B)  # (1, B)
        kwargs = self.decoder.init_kwargs(src, enc_outputs, enc_hidden)

        for batch_idx in range(B):
            start_node = self._create_start_node(input, kwargs, batch_idx)
            heappush(nodes[batch_idx], start_node)

        while len(finished) < B:
            for batch_idx in range(B):
                if batch_idx in finished:
                    continue

                if steps[batch_idx] > max_steps:
                    finished.add(batch_idx)

                node = heappop(nodes[batch_idx])
                if node.prev is not None and node.ix.item() == eos_ix:
                    end_nodes[batch_idx].append(node)
                    if len(end_nodes[batch_idx]) >= n_best:
                        finished.add(batch_idx)
                prev_nodes[batch_idx] = node

            input, kwargs = self._assemble_batch(B, prev_nodes, kwargs)  # (B), {prev_hidden=(n_layers, B, H)}

            output, out_hidden, att = self.decoder(
                input, **kwargs
            )  # -> (B, tgt_vocab_size), (n_layers, B, H), (B, max_src_len)

            logprobs = F.softmax(output, dim=-1).log()  # (B, tgt_vocab_size)

            topk_vals, topk_indices = torch.topk(logprobs, beam_width, dim=-1)  # (B, beam_width), (B, beam_width)

            for batch_idx in range(B):
                if batch_idx in finished:
                    continue

                prev_node = prev_nodes[batch_idx]
                if prev_node is not None and prev_node.length > 1 and prev_node.ix == eos_ix:
                    continue

                for i in range(beam_width):
                    top_i_ix = topk_indices[batch_idx, i].view(-1)  # (1)
                    top_i_logp = cast(float, topk_vals[batch_idx, i].item())

                    new_node = self._create_node(top_i_ix, top_i_logp, prev_node, out_hidden, att, batch_idx)
                    heappush(nodes[batch_idx], new_node)
                steps[batch_idx] += beam_width

        results: list[list[BeamSearchResult]] = []
        for batch_idx in range(B):
            if len(end_nodes[batch_idx]) < n_best:
                end_nodes[batch_idx].extend(
                    [heappop(nodes[batch_idx]) for _ in range(n_best - len(end_nodes[batch_idx]))]
                )

            example_res: list[BeamSearchResult] = []
            for end_node in end_nodes[batch_idx]:
                ix = []
                att = None
                if end_node.att is not None:
                    att = torch.zeros((end_node.length - 1, max_src_len))
                node = end_node
                while node.prev is not None:
                    ix.append(cast(int, node.ix.item()))
                    if node.att is not None:
                        att[node.length - 2] = node.att.cpu()
                    node = node.prev
                ix = ix[::-1]
                s = self.dataset.tgt_tokenizer.decode(ix, strip_specials=True)
                example_res.append(BeamSearchResult(s, ix, end_node.val, att))
            example_res.sort(key=lambda r: r.val)
            results.append(example_res)

        return results
