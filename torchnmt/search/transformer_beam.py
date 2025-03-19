from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Self

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from torchnmt.config import BeamConfig
from torchnmt.data.dataset import NMTDataset
from torchnmt.model import TargetMasker, TransformerDecoder

from .beam import BeamSearchResult


@dataclass
class TransformerBeamSearchNode:
    """
    A node in the beam search process for a Transformer model, representing a single predicted sequence (i.e., a single beam).

    For a detailed explanation of how the nodes are ordered within their priority queues, see RNNBeamSearchNode.

    Attributes:
        prev (TransformerBeamSearchNode | None): A reference to the previous node in the sequence. `None` if this is the start node.
        ix (Tensor): The index tensor representing the tokens of the predicted sequence.
            - Shape: (seq_len,)
        logp (float): The log probability of the most recently predicted token, given the context of the previous tokens in the sequence.
        att (Tensor | None): The cross-attention weights used to predict the most recent token.
            - Shape: (n_layers, n_heads, max_src_len)
        length (int): The length of the sequence ending at this node.
        val (float): The evaluation metric used for sorting nodes during beam search.
    """

    prev: Self | None
    ix: torch.Tensor
    logp: float
    att: Tensor | None
    length: int
    val: float

    @classmethod
    def from_prev(cls, prev: Self | None, ix: Tensor, logp: float, att: Tensor | None) -> Self:
        if prev is None:
            length = 1
            val = 0.0
        else:
            length = prev.length + 1
            logp += prev.logp
            val = -(logp / (length - 1))
            ix = torch.cat((prev.ix, ix))
        return cls(prev, ix, logp, att, length, val)

    def __lt__(self, nxt: Self) -> bool:
        return self.val < nxt.val


class TransformerBeamSearcher:
    """
    Batched beam search implementation compatible with Transformer models.

    Attributes:
        - decoder (TransformerDecoder): The transformer decoder module that generates the target sequences.
        - dataset (NMTDataset): NMTDataset instance providing the start and end token indices.
        - device (str): The device the decoder module is located on.
        - config (BeamConfig): Beam search configuration object.
        - tgt_masker (TargetMasker): Masking module for target input sequences.
    """

    def __init__(self, decoder: TransformerDecoder, dataset: NMTDataset, device: str, config: BeamConfig):
        self.decoder = decoder
        self.dataset = dataset
        self.device = device
        self.config = config
        self.tgt_masker = TargetMasker.from_params(self.decoder.params)

    @torch.no_grad()
    def search(
        self,
        enc_out: Tensor,
        src_mask: Tensor,
        attention: bool = False,
        config: BeamConfig | None = None,
    ) -> list[list[BeamSearchResult]]:
        """
        Performs a batched beam search to generate an approximation of the most likely target sequences for each source
        example in the given batch.

        For a detailed explanation, see the RNNBeamSearcher::search method.

        Args:
            enc_out (Tensor): The outputs from the encoder.
                - Shape: (B, max_src_len, d_model)
            src_mask (Tensor): Generated mask for the source sequences (needed for cross-attention).
                - Shape: (B, 1, 1, max_src_len)
            attention (bool): Whether to keep track of attention weights and return them with the results. Defaults to `False`.
            config (BeamConfig, optional): Beam search config to override the class-level config.

        Returns:
            list[list[BeamSearchResult]]: List of search results for each example in the batch.
        """

        self.decoder.eval()

        config = config or self.config
        beam_width, n_best = config.beam_width, config.n_best

        B, max_src_len, d_model = enc_out.shape
        max_tgt_len = self.dataset.max_tgt_len

        nodes: list[list[TransformerBeamSearchNode]] = [[] for _ in range(B)]
        end_nodes: list[list[TransformerBeamSearchNode]] = [[] for _ in range(B)]
        steps = [0 for _ in range(B)]
        finished: set[int] = set()
        max_steps = max_src_len * 10
        prev_nodes: list[TransformerBeamSearchNode] = [None for _ in range(B)]  # type: ignore

        start_ix = self.dataset.tgt_specials["start"]
        end_ix = self.dataset.tgt_specials["end"]
        pad_ix = self.dataset.tgt_specials["pad"]

        input = torch.tensor([start_ix]).repeat(B, 1).to(self.device)  # (B, 1)

        for batch_idx in range(B):
            start_node = TransformerBeamSearchNode.from_prev(prev=None, ix=input[batch_idx], logp=0, att=None)
            heappush(nodes[batch_idx], start_node)

        while len(finished) < B:
            for batch_idx in range(B):
                if batch_idx in finished:
                    continue

                if steps[batch_idx] > max_steps:
                    finished.add(batch_idx)

                node = heappop(nodes[batch_idx])
                if node.length > 1 and (node.ix[-1].item() == end_ix or len(node.ix) == max_tgt_len):
                    end_nodes[batch_idx].append(node)
                    if len(end_nodes[batch_idx]) >= n_best:
                        finished.add(batch_idx)
                prev_nodes[batch_idx] = node

            tgt = [prev_nodes[batch_idx].ix for batch_idx in range(B)]
            tgt_last_ix = torch.tensor([len(tgt[batch_idx]) - 1 for batch_idx in range(B)]).to(self.device)  # (B,)

            input = pad_sequence(tgt, batch_first=True, padding_value=pad_ix)  # (B, max_tgt_len)
            tgt_mask = self.tgt_masker(input)  # (B, 1, max_tgt_len, max_tgt_len)

            output, _, cross_att = self.decoder(
                input, enc_out, src_mask, tgt_mask
            )  # -> (B, max_tgt_len, tgt_vocab_size), [(B, n_heads, max_tgt_len, max_tgt_len)] * n_layers, [(B, n_heads, max_tgt_len, max_src_len)] * n_layers

            logits = output[torch.arange(B), tgt_last_ix]  # (B, tgt_vocab_size)

            if attention:
                cross_att = torch.stack(cross_att, dim=1)  # (B, n_layers, n_heads, max_tgt_len, max_src_len)
                cross_att = cross_att[torch.arange(B), :, :, tgt_last_ix]  # (B, n_layers, n_heads, max_src_len)
            logprobs = F.softmax(logits, dim=-1).log()

            topk_vals, topk_indices = torch.topk(logprobs, beam_width, dim=-1)

            for batch_idx in range(B):
                if batch_idx in finished:
                    continue

                prev_node = prev_nodes[batch_idx]
                if (
                    prev_node is not None
                    and prev_node.length > 1
                    and (prev_node.ix[-1].item() == end_ix or len(prev_node.ix) == max_tgt_len)
                ):
                    continue

                for i in range(beam_width):
                    top_i_ix = topk_indices[batch_idx, i].view(-1)
                    top_i_logp = topk_vals[batch_idx, i].item()

                    new_node = TransformerBeamSearchNode.from_prev(
                        prev=prev_node, ix=top_i_ix, logp=top_i_logp, att=cross_att[batch_idx] if attention else None
                    )
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
                node = end_node
                s = self.dataset.tgt_tokenizer.decode(end_node.ix.tolist(), strip_specials=True)
                att: Tensor | None = None
                if attention:
                    att_list = []
                    while node.prev is not None and node.att is not None:
                        att_list.append(node.att.cpu())  # (n_layers, n_heads, max_src_len)
                        node = node.prev
                    att_list.reverse()
                    att = torch.stack(att_list, dim=2)  # (n_layers, n_heads, end_node.length-1, max_src_len)
                example_res.append(BeamSearchResult(s=s, ix=end_node.ix[1:].tolist(), val=end_node.val, att=att))

            example_res.sort(key=lambda r: r.val)
            results.append(example_res)

        return results
