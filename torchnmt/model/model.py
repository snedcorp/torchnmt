import math
from typing import Union

import torch.nn as nn

from torchnmt.data import NMTDataset

from .config import (
    BahdanauConfig,
    LuongConfig,
    MergeMethod,
    ModelConfig,
    RNNDecoderParams,
    RNNEncoderParams,
    ScoreMethod,
    SutskeverConfig,
    TransformerConfig,
    TransformerParams,
)
from .rnn import BahdanauRNNDecoder, LuongRNNDecoder, RNNEncoder, RNNSeq2Seq, SutskeverRNNDecoder
from .transformer import TransformerDecoder, TransformerEncoder, TransformerSeq2Seq


def _init_rnn_weights(m):
    for n, param in m.named_parameters():
        if "weight" in n:
            nn.init.uniform_(param.data, -0.1, 0.1)
        else:
            nn.init.constant_(param.data, 0)


def _get_init_transformer_weights(d_model: int):
    std = 1 / math.sqrt(d_model)

    def init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=std)

    return init_weights


def build_sutskever_model(config: SutskeverConfig, dataset: NMTDataset) -> RNNSeq2Seq:
    """
    Builds a Sutskever-style RNN-based sequence-to-sequence model for neural machine translation (see
    SutskeverRNNDecoder for details).

    Args:
        config (SutskeverConfig): Configuration object containing model parameters.
        dataset (NMTDataset): Dataset object providing vocabulary sizes and padding indices for the
                              source and target languages.

    Returns:
        RNNSeq2Seq: A fully constructed sequence-to-sequence model with initialized weights.
    """

    encoder_params = RNNEncoderParams(
        vocab_size=dataset.src_vocab_size,
        n_emb=config.n_emb,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        dropout=config.dropout,
        pad_ix=dataset.src_pad_ix,
        use_context=True,
        merge_method=MergeMethod.SUM,  # doesn't matter, no attention
    )
    encoder = RNNEncoder(encoder_params)

    decoder_params = RNNDecoderParams(
        vocab_size=dataset.tgt_vocab_size,
        n_emb=config.n_emb,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        dropout=config.dropout,
        pad_ix=dataset.tgt_pad_ix,
    )
    decoder = SutskeverRNNDecoder(decoder_params)

    model = RNNSeq2Seq(encoder, decoder)
    model.apply(_init_rnn_weights)
    return model


def build_bahdanau_model(config: BahdanauConfig, dataset: NMTDataset) -> RNNSeq2Seq:
    """
    Builds a Bahdanau-style RNN-based sequence-to-sequence model for neural machine translation (see BahdanauRNNDecoder
    for details).

    Args:
        config (BahdanauConfig): Configuration object containing model parameters.
        dataset (NMTDataset): Dataset object providing vocabulary sizes and padding indices for the
                              source and target languages.

    Returns:
        RNNSeq2Seq: A fully constructed sequence-to-sequence model with initialized weights.
    """

    encoder_params = RNNEncoderParams(
        vocab_size=dataset.src_vocab_size,
        n_emb=config.n_emb,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        dropout=config.dropout,
        pad_ix=dataset.src_pad_ix,
        use_context=True,
        merge_method=MergeMethod.CONCAT,
    )
    encoder = RNNEncoder(encoder_params)

    decoder_params = RNNDecoderParams(
        vocab_size=dataset.tgt_vocab_size,
        n_emb=config.n_emb,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        dropout=config.dropout,
        pad_ix=dataset.tgt_pad_ix,
        att_score_method=ScoreMethod.ADDITIVE,
    )
    decoder = BahdanauRNNDecoder(decoder_params)

    model = RNNSeq2Seq(encoder, decoder)
    model.apply(_init_rnn_weights)
    return model


def build_luong_model(config: LuongConfig, dataset: NMTDataset) -> RNNSeq2Seq:
    """
    Builds a Luong-style RNN-based sequence-to-sequence model for neural machine translation (see LuongRNNDecoder for
    details).

    Args:
        config (LuongConfig): Configuration object containing model parameters.
        dataset (NMTDataset): Dataset object providing vocabulary sizes and padding indices for the
                              source and target languages.

    Returns:
        RNNSeq2Seq: A fully constructed sequence-to-sequence model with initialized weights.
    """

    encoder_params = RNNEncoderParams(
        vocab_size=dataset.src_vocab_size,
        n_emb=config.n_emb,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        dropout=config.dropout,
        pad_ix=dataset.src_pad_ix,
        use_context=config.use_context,
        merge_method=MergeMethod.SUM,
    )
    encoder = RNNEncoder(encoder_params)

    decoder_params = RNNDecoderParams(
        vocab_size=dataset.tgt_vocab_size,
        n_emb=config.n_emb,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        dropout=config.dropout,
        pad_ix=dataset.tgt_pad_ix,
        att_score_method=config.att_score_method,
        input_feeding=config.input_feeding,
    )
    decoder = LuongRNNDecoder(decoder_params)

    model = RNNSeq2Seq(encoder, decoder)
    model.apply(_init_rnn_weights)
    return model


def build_transformer_model(config: TransformerConfig, dataset: NMTDataset) -> TransformerSeq2Seq:
    """
    Builds a Transformer-based sequence-to-sequence model for neural machine translation.

    Args:
        config (TransformerConfig): Configuration object containing model parameters.
        dataset (NMTDataset): Dataset object providing vocabulary sizes and padding indices for the
                              source and target languages.

    Returns:
        TransformerSeq2Seq: A fully constructed sequence-to-sequence model with initialized weights.
    """

    encoder_params = TransformerParams(
        vocab_size=dataset.src_vocab_size,
        max_seq_len=dataset.max_src_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        sine_pos=config.sine_pos,
        pre_norm=config.pre_norm,
        dropout=config.dropout,
        pad_ix=dataset.src_pad_ix,
        naive=config.naive,
    )
    encoder = TransformerEncoder(encoder_params)

    decoder_params = TransformerParams(
        vocab_size=dataset.tgt_vocab_size,
        max_seq_len=dataset.max_tgt_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        sine_pos=config.sine_pos,
        pre_norm=config.pre_norm,
        dropout=config.dropout,
        pad_ix=dataset.tgt_pad_ix,
        naive=config.naive,
    )
    decoder = TransformerDecoder(decoder_params)

    model = TransformerSeq2Seq(encoder, decoder)
    model.apply(_get_init_transformer_weights(config.d_model))
    return model


def build_model(config: ModelConfig, dataset: NMTDataset, device: str) -> Union[RNNSeq2Seq, TransformerSeq2Seq]:
    match config:
        case SutskeverConfig():
            return build_sutskever_model(config, dataset).to(device)
        case BahdanauConfig():
            return build_bahdanau_model(config, dataset).to(device)
        case LuongConfig():
            return build_luong_model(config, dataset).to(device)
        case TransformerConfig():
            return build_transformer_model(config, dataset).to(device)
        case _:
            raise ValueError(f"Invalid build model config: {config}")
