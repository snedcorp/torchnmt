import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from tests.utils import MockDataset
from torchnmt.model.config import TransformerConfig, TransformerParams
from torchnmt.model.model import build_model
from torchnmt.model.transformer import (
    AttentionHead,
    Embedding,
    FeedForward,
    MultiHeadAttention,
    NaiveMultiHeadAttention,
    NaivePadMasker,
    NaiveTargetMasker,
    PadMasker,
    PostNormResidualSubLayer,
    PreNormResidualSubLayer,
    SinePositionalEncoding,
    TargetMasker,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerSeq2Seq,
)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("max_seq_len", [5, 10, 15, 20])
def test_PadMasker(batch_size, max_seq_len):
    pad_ix = 0

    pad_mask = PadMasker(pad_ix)

    input = torch.randint(high=3, size=(batch_size, max_seq_len))

    mask = pad_mask(input)

    assert mask.shape == (batch_size, 1, 1, max_seq_len)
    for b in range(batch_size):
        for i in range(max_seq_len):
            if input[b, i].item() != pad_ix:
                assert mask[b, 0, 0, i].item()
            else:
                assert not mask[b, 0, 0, i].item()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("max_seq_len", [5, 10, 15, 20])
def test_NaivePadMasker(batch_size, max_seq_len):
    pad_ix = 0

    pad_mask = NaivePadMasker(pad_ix)

    input = torch.randint(high=3, size=(batch_size, max_seq_len))

    mask = pad_mask(input)

    assert mask.shape == (batch_size, 1, max_seq_len)
    for b in range(batch_size):
        for i in range(max_seq_len):
            if input[b, i].item() != pad_ix:
                assert mask[b, 0, i].item()
            else:
                assert not mask[b, 0, i].item()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("max_seq_len", [5, 10, 15, 20])
def test_TargetMasker_get_subsequent_mask(batch_size, max_seq_len):
    input = torch.randint(high=3, size=(batch_size, max_seq_len))

    mask = TargetMasker._get_subsequent_mask(input)

    assert mask.shape == (batch_size, max_seq_len, max_seq_len)
    for b in range(batch_size):
        for row_ix in range(max_seq_len):
            for col_ix in range(max_seq_len):
                if col_ix <= row_ix:
                    assert mask[b, row_ix, col_ix].item()
                else:
                    assert not mask[b, row_ix, col_ix].item()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("max_seq_len", [5, 10, 15, 20])
def test_TargetMasker(batch_size, max_seq_len):
    pad_ix = 0

    tgt_masker = TargetMasker(pad_ix)

    input = torch.randint(high=3, size=(batch_size, max_seq_len))

    mask = tgt_masker(input)

    assert mask.shape == (batch_size, 1, max_seq_len, max_seq_len)
    for b in range(batch_size):
        for row_ix in range(max_seq_len):
            for col_ix in range(max_seq_len):
                if col_ix <= row_ix and input[b, col_ix].item() != pad_ix:
                    assert mask[b, 0, row_ix, col_ix].item()
                else:
                    assert not mask[b, 0, row_ix, col_ix].item()


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("max_seq_len", [5, 10, 15, 20])
def test_NaiveTargetMasker(batch_size, max_seq_len):
    pad_ix = 0

    tgt_masker = NaiveTargetMasker(pad_ix)

    input = torch.randint(high=3, size=(batch_size, max_seq_len))

    mask = tgt_masker(input)

    assert mask.shape == (batch_size, max_seq_len, max_seq_len)
    for b in range(batch_size):
        for row_ix in range(max_seq_len):
            for col_ix in range(max_seq_len):
                if col_ix <= row_ix and input[b, col_ix].item() != pad_ix:
                    assert mask[b, row_ix, col_ix].item()
                else:
                    assert not mask[b, row_ix, col_ix].item()


@torch.no_grad()
@pytest.mark.parametrize("B", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [5, 10, 20])
@pytest.mark.parametrize("d_model", [4, 8, 16])
def test_SinePositionalEncoding(B, seq_len, d_model):
    pos_enc = SinePositionalEncoding(seq_len, d_model)

    pos = torch.arange(0, seq_len).unsqueeze(0).expand(B, -1)
    out = pos_enc(pos)

    assert out.shape == (B, seq_len, d_model)


def get_params(n_layers=2, dropout=0.1, max_seq_len=7, d_model=8, n_heads=4, sine_pos=True, pre_norm=True, naive=False):
    return TransformerParams(
        vocab_size=10,
        n_layers=n_layers,
        dropout=dropout,
        pad_ix=0,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_model * 4,
        sine_pos=sine_pos,
        pre_norm=pre_norm,
        naive=naive,
    )


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("max_seq_len", [5, 10, 20])
@pytest.mark.parametrize("d_model", [4, 8, 16])
@pytest.mark.parametrize("sine_pos", [True, False])
def test_Embedding(batch_size, max_seq_len, d_model, sine_pos):
    params = get_params(max_seq_len=max_seq_len, d_model=d_model, sine_pos=sine_pos)

    emb = Embedding(params)

    input = torch.randint(high=params.vocab_size, size=(batch_size, params.max_seq_len))

    out = emb(input)

    assert out.shape == (batch_size, params.max_seq_len, params.d_model)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("d_model", [64, 128, 256])
@pytest.mark.parametrize("d_head", [4, 8, 16])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("query_len", [6, 10, 15])
@pytest.mark.parametrize("kv_len", [6, 10, 15])
def test_AttentionHead(batch_size, d_model, d_head, dropout, query_len, kv_len):
    head = AttentionHead(d_model, d_head, dropout)

    query = torch.rand((batch_size, query_len, d_model))
    key = torch.rand((batch_size, kv_len, d_model))
    value = torch.rand((batch_size, kv_len, d_model))
    mask = torch.randint(high=2, size=(batch_size, 1, kv_len))

    context, attention = head(query, key, value, mask)

    assert context.shape == (batch_size, query_len, d_head)
    assert attention.shape == (batch_size, query_len, kv_len)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("d_model", [64, 128, 256])
@pytest.mark.parametrize("n_heads", [4, 8, 16])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("query_len", [6, 10, 15])
@pytest.mark.parametrize("kv_len", [6, 10, 15])
def test_NaiveMultiHeadAttention(batch_size, d_model, n_heads, dropout, query_len, kv_len):
    multi_head = NaiveMultiHeadAttention(d_model, n_heads, dropout)

    query = torch.rand((batch_size, query_len, d_model))
    key = torch.rand((batch_size, kv_len, d_model))
    value = torch.rand((batch_size, kv_len, d_model))
    mask = torch.randint(high=2, size=(batch_size, 1, kv_len))
    scores = []

    out = multi_head(query, key, value, mask, scores)

    assert out.shape == (batch_size, query_len, d_model)
    assert len(scores) == 1
    assert scores[0].shape == (batch_size, n_heads, query_len, kv_len)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("d_model", [64, 128, 256])
@pytest.mark.parametrize("n_heads", [4, 8, 16])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("query_len", [6, 10, 15])
@pytest.mark.parametrize("kv_len", [6, 10, 15])
def test_MultiHeadAttention(batch_size, d_model, n_heads, dropout, query_len, kv_len):
    multi_head = MultiHeadAttention(d_model, n_heads, dropout)

    query = torch.rand((batch_size, query_len, d_model))
    key = torch.rand((batch_size, kv_len, d_model))
    value = torch.rand((batch_size, kv_len, d_model))
    mask = torch.randint(high=2, size=(batch_size, 1, 1, kv_len))
    scores = []

    out = multi_head(query, key, value, mask, scores)

    assert out.shape == (batch_size, query_len, d_model)
    assert len(scores) == 1
    assert scores[0].shape == (batch_size, n_heads, query_len, kv_len)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("d_model", [8, 16])
@pytest.mark.parametrize("d_ff", [32, 64])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("max_seq_len", [7, 12, 20])
def test_FeedForward(batch_size, d_model, d_ff, dropout, max_seq_len):
    ff = FeedForward(d_model, d_ff, dropout)

    input = torch.rand((batch_size, max_seq_len, d_model))

    out = ff(input)

    assert out.shape == (batch_size, max_seq_len, d_model)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("d_model", [8, 16])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("max_seq_len", [7, 12, 20])
def test_PostNormResidualSubLayer(batch_size, d_model, dropout, max_seq_len):
    residual = PostNormResidualSubLayer(d_model, dropout)

    input = torch.rand((batch_size, max_seq_len, d_model))
    sub = torch.nn.Linear(d_model, d_model)

    out = residual(input, sub)

    assert out.shape == (batch_size, max_seq_len, d_model)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("d_model", [8, 16])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("max_seq_len", [7, 12, 20])
def test_PreNormResidualSubLayer(batch_size, d_model, dropout, max_seq_len):
    residual = PreNormResidualSubLayer(d_model, dropout)

    input = torch.rand((batch_size, max_seq_len, d_model))
    sub = torch.nn.Linear(d_model, d_model)

    out = residual(input, sub)

    assert out.shape == (batch_size, max_seq_len, d_model)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("d_model", [8, 16])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("max_seq_len", [7, 12, 20])
@pytest.mark.parametrize("pre_norm", [True, False])
def test_TransformerEncoderLayer(batch_size, d_model, dropout, max_seq_len, pre_norm):
    params = get_params(max_seq_len=max_seq_len, d_model=d_model, dropout=dropout, pre_norm=pre_norm)

    input = torch.rand((batch_size, max_seq_len, params.d_model))
    mask = torch.randint(high=2, size=(batch_size, 1, 1, max_seq_len))
    scores = []

    enc_layer = TransformerEncoderLayer(params)

    out = enc_layer(input, mask, scores)

    assert out.shape == (batch_size, max_seq_len, params.d_model)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
@pytest.mark.parametrize("d_model", [8, 16])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("max_seq_len", [7, 12, 20])
@pytest.mark.parametrize("pre_norm", [True, False])
@pytest.mark.parametrize("sine_pos", [True, False])
def test_TransformerEncoder_training(batch_size, n_layers, d_model, dropout, max_seq_len, pre_norm, sine_pos):
    params = get_params(
        n_layers=n_layers,
        d_model=d_model,
        dropout=dropout,
        max_seq_len=max_seq_len,
        pre_norm=pre_norm,
        sine_pos=sine_pos,
    )

    src = torch.randint(high=params.vocab_size, size=(batch_size, max_seq_len))
    mask = torch.randint(high=2, size=(batch_size, 1, 1, max_seq_len))

    enc = TransformerEncoder(params)
    enc.train()

    out, scores = enc(src, mask)

    assert out.shape == (batch_size, max_seq_len, params.d_model)
    assert scores is None


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
@pytest.mark.parametrize("d_model", [8, 16])
@pytest.mark.parametrize("max_seq_len", [5, 10, 15, 20])
def test_TransformerEncoder_eval(batch_size, n_layers, d_model, max_seq_len):
    params = get_params(
        n_layers=n_layers, d_model=d_model, max_seq_len=max_seq_len, pre_norm=True, sine_pos=False, naive=False
    )

    input = torch.randint(high=params.vocab_size, size=(batch_size, max_seq_len))
    mask = None
    while mask is None or torch.any(mask.sum(dim=-1) == 0):
        mask = torch.randint(high=2, size=(batch_size, 1, 1, max_seq_len))

    enc = TransformerEncoder(params)
    enc.eval()

    out, scores = enc(input, mask)

    assert out.shape == (batch_size, max_seq_len, params.d_model)
    assert_bidirectional_self_attention(scores, mask, params.n_layers, batch_size, params.n_heads, max_seq_len)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
@pytest.mark.parametrize("d_model", [8, 16])
@pytest.mark.parametrize("max_seq_len", [5, 10])
def test_TransformerEncoder_eval_naive(batch_size, n_layers, d_model, max_seq_len):
    params = get_params(
        n_layers=n_layers, d_model=d_model, max_seq_len=max_seq_len, pre_norm=True, sine_pos=False, naive=True
    )

    input = torch.randint(high=params.vocab_size, size=(batch_size, max_seq_len))
    mask = None
    while mask is None or torch.any(mask.sum(dim=-1) == 0):
        mask = torch.randint(high=2, size=(batch_size, 1, max_seq_len))

    enc = TransformerEncoder(params)
    enc.eval()

    out, scores = enc(input, mask)

    assert out.shape == (batch_size, max_seq_len, params.d_model)
    assert_bidirectional_self_attention(
        scores, mask, params.n_layers, batch_size, params.n_heads, max_seq_len, naive=True
    )


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
def test_TransformerDecoderLayer(batch_size):
    max_src_len = 7
    max_tgt_len = 9
    params = get_params(max_seq_len=max_tgt_len)

    input = torch.rand((batch_size, max_tgt_len, params.d_model))
    enc_out = torch.rand((batch_size, max_src_len, params.d_model))
    src_mask = torch.randint(high=2, size=(batch_size, 1, 1, max_src_len))
    tgt_mask = torch.randint(high=2, size=(batch_size, 1, max_tgt_len, max_tgt_len))
    self_scores = []
    cross_scores = []

    dec_layer = TransformerDecoderLayer(params)

    out = dec_layer(input, enc_out, src_mask, tgt_mask, self_scores, cross_scores)

    assert out.shape == (batch_size, max_tgt_len, params.d_model)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_layers", [1, 2])
def test_TransformerDecoder_training(batch_size, n_layers):
    max_src_len = 7
    max_tgt_len = 9
    params = get_params(n_layers=n_layers, max_seq_len=max_tgt_len)

    tgt = torch.randint(high=params.vocab_size, size=(batch_size, max_tgt_len))
    enc_out = torch.rand((batch_size, max_src_len, params.d_model))
    src_mask = torch.randint(high=2, size=(batch_size, 1, 1, max_src_len))
    tgt_mask = torch.randint(high=2, size=(batch_size, 1, max_tgt_len, max_tgt_len))

    dec = TransformerDecoder(params)
    dec.train()

    out, self_scores, cross_scores = dec(tgt, enc_out, src_mask, tgt_mask)

    assert out.shape == (batch_size, max_tgt_len, params.vocab_size)
    assert self_scores is None
    assert cross_scores is None


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
@pytest.mark.parametrize("d_model", [8, 16])
@pytest.mark.parametrize("max_src_len", [5, 10])
@pytest.mark.parametrize("max_tgt_len", [5, 10])
def test_TransformerDecoder_eval(batch_size, n_layers, d_model, max_src_len, max_tgt_len):
    params = get_params(n_layers=n_layers, d_model=d_model, max_seq_len=max_tgt_len)

    tgt = torch.randint(high=params.vocab_size, size=(batch_size, max_tgt_len))
    enc_out = torch.rand((batch_size, max_src_len, params.d_model))

    src_mask = None
    while src_mask is None or torch.any(src_mask.sum(dim=-1) == 0):
        src_mask = torch.randint(high=2, size=(batch_size, 1, 1, max_src_len))

    tgt_mask = None
    while tgt_mask is None or torch.any(tgt_mask.sum(dim=-1) == 0):
        tgt_mask = torch.randint(high=2, size=(batch_size, 1, max_tgt_len, max_tgt_len))

    dec = TransformerDecoder(params)
    dec.eval()

    out, self_scores, cross_scores = dec(tgt, enc_out, src_mask, tgt_mask)

    assert out.shape == (batch_size, max_tgt_len, params.vocab_size)

    assert_causal_self_attention(self_scores, tgt_mask, n_layers, batch_size, params.n_heads, max_tgt_len)
    assert_cross_attention(cross_scores, src_mask, n_layers, batch_size, params.n_heads, max_tgt_len, max_src_len)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
@pytest.mark.parametrize("d_model", [8, 16])
@pytest.mark.parametrize("max_src_len", [5, 10])
@pytest.mark.parametrize("max_tgt_len", [5, 10])
def test_TransformerDecoder_eval_naive(batch_size, n_layers, d_model, max_src_len, max_tgt_len):
    params = get_params(n_layers=n_layers, d_model=d_model, max_seq_len=max_tgt_len, naive=True)

    tgt = torch.randint(high=params.vocab_size, size=(batch_size, max_tgt_len))
    enc_out = torch.rand((batch_size, max_src_len, params.d_model))

    src_mask = None
    while src_mask is None or torch.any(src_mask.sum(dim=-1) == 0):
        src_mask = torch.randint(high=2, size=(batch_size, 1, max_src_len))

    tgt_mask = None
    while tgt_mask is None or torch.any(tgt_mask.sum(dim=-1) == 0):
        tgt_mask = torch.randint(high=2, size=(batch_size, max_tgt_len, max_tgt_len))

    dec = TransformerDecoder(params)
    dec.eval()

    out, self_scores, cross_scores = dec(tgt, enc_out, src_mask, tgt_mask)

    assert out.shape == (batch_size, max_tgt_len, params.vocab_size)

    assert_causal_self_attention(self_scores, tgt_mask, n_layers, batch_size, params.n_heads, max_tgt_len, naive=True)
    assert_cross_attention(
        cross_scores, src_mask, n_layers, batch_size, params.n_heads, max_tgt_len, max_src_len, naive=True
    )


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("d_model", [16, 32, 64])
@pytest.mark.parametrize("n_heads", [4, 8])
@pytest.mark.parametrize("sine_pos", [True, False])
@pytest.mark.parametrize("pre_norm", [True, False])
@pytest.mark.parametrize("naive", [True, False])
def test_TransformerSeq2Seq_training(batch_size, n_layers, dropout, d_model, n_heads, sine_pos, pre_norm, naive):
    config = TransformerConfig(
        n_layers=n_layers,
        dropout=dropout,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_model * 2,
        sine_pos=sine_pos,
        pre_norm=pre_norm,
        naive=naive,
    )
    dataset = MockDataset()

    model = build_model(config, dataset, "cpu")

    assert isinstance(model, TransformerSeq2Seq)
    assert isinstance(model.encoder, TransformerEncoder)
    assert isinstance(model.decoder, TransformerDecoder)

    src, tgt = get_batch(dataset, batch_size)

    max_tgt_len = tgt.shape[1]

    model.train()
    out = model(src, tgt)

    assert out.shape == (batch_size, max_tgt_len, model.decoder.params.vocab_size)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
@pytest.mark.parametrize("d_model", [16, 32])
@pytest.mark.parametrize("n_heads", [4, 8])
def test_TransformerSeq2Seq_eval(batch_size, n_layers, d_model, n_heads):
    config = TransformerConfig(
        n_layers=n_layers,
        dropout=0.0,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_model * 2,
        naive=False,
        sine_pos=False,
        pre_norm=True,
    )
    dataset = MockDataset()

    model = build_model(config, dataset, "cpu")

    assert isinstance(model, TransformerSeq2Seq)
    assert isinstance(model.encoder, TransformerEncoder)
    assert isinstance(model.decoder, TransformerDecoder)

    src, tgt = get_batch(dataset, batch_size)

    max_src_len = src.shape[1]
    max_tgt_len = tgt.shape[1]

    model.eval()

    src_mask = model.src_masker(src)
    tgt_mask = model.tgt_masker(tgt)

    enc_out, enc_self_scores = model.encoder(src, src_mask)
    dec_out, dec_self_scores, dec_cross_scores = model.decoder(tgt, enc_out, src_mask, tgt_mask)

    assert dec_out.shape == (batch_size, max_tgt_len, model.decoder.params.vocab_size)

    assert_bidirectional_self_attention(
        enc_self_scores, src_mask, config.n_layers, batch_size, config.n_heads, max_src_len
    )
    assert_causal_self_attention(dec_self_scores, tgt_mask, config.n_layers, batch_size, config.n_heads, max_tgt_len)
    assert_cross_attention(
        dec_cross_scores, src_mask, config.n_layers, batch_size, config.n_heads, max_tgt_len, max_src_len
    )


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
@pytest.mark.parametrize("d_model", [16, 32])
@pytest.mark.parametrize("n_heads", [4, 8])
def test_TransformerSeq2Seq_eval_naive(batch_size, n_layers, d_model, n_heads):
    config = TransformerConfig(
        n_layers=n_layers,
        dropout=0.0,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_model * 2,
        naive=True,
        sine_pos=False,
        pre_norm=True,
    )
    dataset = MockDataset()

    model = build_model(config, dataset, "cpu")

    assert isinstance(model, TransformerSeq2Seq)
    assert isinstance(model.encoder, TransformerEncoder)
    assert isinstance(model.decoder, TransformerDecoder)

    src, tgt = get_batch(dataset, batch_size)

    max_src_len = src.shape[1]
    max_tgt_len = tgt.shape[1]

    model.eval()

    src_mask = model.src_masker(src)
    tgt_mask = model.tgt_masker(tgt)

    enc_out, enc_self_scores = model.encoder(src, src_mask)
    dec_out, dec_self_scores, dec_cross_scores = model.decoder(tgt, enc_out, src_mask, tgt_mask)

    assert dec_out.shape == (batch_size, max_tgt_len, model.decoder.params.vocab_size)

    assert_bidirectional_self_attention(
        enc_self_scores, src_mask, config.n_layers, batch_size, config.n_heads, max_src_len, naive=True
    )
    assert_causal_self_attention(
        dec_self_scores, tgt_mask, config.n_layers, batch_size, config.n_heads, max_tgt_len, naive=True
    )
    assert_cross_attention(
        dec_cross_scores, src_mask, config.n_layers, batch_size, config.n_heads, max_tgt_len, max_src_len, naive=True
    )


def get_batch(mock_dataset: MockDataset, batch_size: int):
    srcs = []
    tgts = []
    for _ in range(batch_size):
        src_seq_len = torch.randint(low=3, high=mock_dataset.max_src_len, size=(1,)).item()
        src = torch.randint(low=1, high=mock_dataset.src_vocab_size, size=(src_seq_len,))
        srcs.append(src)

        tgt_seq_len = torch.randint(low=3, high=mock_dataset.max_tgt_len, size=(1,)).item()
        tgt = torch.randint(low=1, high=mock_dataset.tgt_vocab_size, size=(tgt_seq_len,))
        tgts.append(tgt)

    src = pad_sequence(srcs, batch_first=True, padding_value=mock_dataset.src_pad_ix)
    tgt = pad_sequence(tgts, batch_first=True, padding_value=mock_dataset.tgt_pad_ix)

    return src, tgt


def assert_bidirectional_self_attention(self_scores, src_mask, n_layers, batch_size, n_heads, max_src_len, naive=False):
    assert self_scores is not None
    assert len(self_scores) == n_layers

    for scores in self_scores:
        assert scores.shape == (batch_size, n_heads, max_src_len, max_src_len)
        for ex in range(batch_size):
            for head in range(n_heads):
                for query in range(max_src_len):
                    for key in range(max_src_len):
                        if not naive:
                            if src_mask[ex, 0, 0, key]:
                                assert scores[ex, head, query, key] > 0
                            else:
                                assert scores[ex, head, query, key] == 0
                        else:
                            if src_mask[ex, 0, key]:
                                assert scores[ex, head, query, key] > 0
                            else:
                                assert scores[ex, head, query, key] == 0


def assert_causal_self_attention(self_scores, tgt_mask, n_layers, batch_size, n_heads, max_tgt_len, naive=False):
    assert self_scores is not None
    assert len(self_scores) == n_layers

    for scores in self_scores:
        assert scores.shape == (batch_size, n_heads, max_tgt_len, max_tgt_len)
        for ex in range(batch_size):
            for head in range(n_heads):
                for query in range(max_tgt_len):
                    for key in range(max_tgt_len):
                        if not naive:
                            if tgt_mask[ex, 0, query, key]:
                                assert scores[ex, head, query, key] > 0
                            else:
                                assert scores[ex, head, query, key] == 0
                        else:
                            if tgt_mask[ex, query, key]:
                                assert scores[ex, head, query, key] > 0
                            else:
                                assert scores[ex, head, query, key] == 0


def assert_cross_attention(
    cross_scores, src_mask, n_layers, batch_size, n_heads, max_tgt_len, max_src_len, naive=False
):
    assert cross_scores is not None
    assert len(cross_scores) == n_layers

    for scores in cross_scores:
        assert scores.shape == (batch_size, n_heads, max_tgt_len, max_src_len)
        for ex in range(batch_size):
            for head in range(n_heads):
                for query in range(max_tgt_len):
                    for key in range(max_src_len):
                        if not naive:
                            if src_mask[ex, 0, 0, key]:
                                assert scores[ex, head, query, key] > 0
                            else:
                                assert scores[ex, head, query, key] == 0
                        else:
                            if src_mask[ex, 0, key]:
                                assert scores[ex, head, query, key] > 0
                            else:
                                assert scores[ex, head, query, key] == 0
