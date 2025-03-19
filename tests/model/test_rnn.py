from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from tests.utils import MockDataset
from torchnmt.model.config import (
    BahdanauConfig,
    LuongConfig,
    MergeMethod,
    RNNDecoderParams,
    RNNEncoderParams,
    ScoreMethod,
    SutskeverConfig,
)
from torchnmt.model.model import build_model
from torchnmt.model.rnn import (
    AdditiveScorer,
    Attention,
    BahdanauRNNDecoder,
    DecoderGRU,
    DotScorer,
    GeneralScorer,
    GRUCell,
    LuongRNNDecoder,
    RNNEncoder,
    RNNSeq2Seq,
    ScaledDotScorer,
    SutskeverRNNDecoder,
)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
def test_GRUCell(batch_size):
    n_emb = 4
    n_hidden = 8
    gru_cell = GRUCell(n_emb, n_hidden)

    X_t = torch.rand((batch_size, n_emb))
    h_prev = torch.rand((batch_size, n_hidden))
    out = gru_cell(X_t, h_prev)

    assert out.shape == (batch_size, n_hidden)


def get_enc_params(
    vocab_size=10,
    n_layers=2,
    dropout=0.1,
    pad_ix=0,
    n_emb=4,
    n_hidden=8,
    use_context=False,
    merge_method=MergeMethod.SUM,
):
    return RNNEncoderParams(
        vocab_size=vocab_size,
        n_layers=n_layers,
        dropout=dropout,
        pad_ix=pad_ix,
        n_emb=n_emb,
        n_hidden=n_hidden,
        use_context=use_context,
        merge_method=merge_method,
    )


@pytest.fixture
def enc_input_single():
    input = torch.tensor([[9, 7, 5, 3, 1]], dtype=torch.long)
    input_len = torch.tensor([5], dtype=torch.int)
    return input, input_len


@pytest.fixture
def enc_input_mult():
    input = torch.tensor([[1, 2, 3, 0, 0], [4, 2, 1, 4, 0], [9, 7, 5, 3, 1]], dtype=torch.long)
    input_len = torch.tensor([3, 4, 5], dtype=torch.int)
    return input, input_len


@torch.no_grad()
def test_RNNEncoder_get_ix_mapping():
    input = torch.tensor(
        [[1, 2, 3, 0, 0], [11, 12, 13, 14, 0], [21, 22, 23, 24, 25], [31, 32, 33, 0, 0]], dtype=torch.long
    )
    input_len = torch.tensor([3, 4, 5, 3], dtype=torch.int)
    input_emb = input.view(*input.shape[:2], 1).float()
    input_packed = pack_padded_sequence(input_emb, input_len, batch_first=True, enforce_sorted=False)
    ix_mapping = RNNEncoder.get_ix_mapping(input_packed, len(input_len))

    # per_example_ix = [ [0,4,8,12,14], [1,5,9,13], [2,6,10], [3,7,11] ]
    # ix_mapping = [14, 13, 10, 11, 12, 9, 6, 7, 8, 5, 2, 3, 4, 1, 0]
    assert len(ix_mapping) == sum(input_packed.batch_sizes)
    assert ix_mapping[0] == 14
    assert ix_mapping[1] == 13
    assert ix_mapping[2] == 10
    assert ix_mapping[3] == 11
    assert ix_mapping[4] == 12
    assert ix_mapping[5] == 9
    assert ix_mapping[6] == 6
    assert ix_mapping[7] == 7
    assert ix_mapping[8] == 8
    assert ix_mapping[9] == 5
    assert ix_mapping[10] == 2
    assert ix_mapping[11] == 3
    assert ix_mapping[12] == 4
    assert ix_mapping[13] == 1
    assert ix_mapping[14] == 0


@torch.no_grad()
def test_RNNEncoder_ix_mapping_sum():
    params = get_enc_params(n_layers=1, dropout=0, merge_method=MergeMethod.SUM)
    encoder = RNNEncoder(params)

    input = torch.tensor(
        [[1, 2, 3, 0, 0], [11, 12, 13, 14, 0], [21, 22, 23, 24, 25], [31, 32, 33, 0, 0]], dtype=torch.long
    )
    input_len = torch.tensor([3, 4, 5, 3], dtype=torch.int)
    B, max_src_len = input.shape

    with (
        patch.object(encoder, "emb", autospec=True) as mock_emb,
        patch.object(encoder, "dropout", autospec=True) as mock_drop,
    ):
        mock_emb.side_effect = lambda x: x.view(*x.shape[:2], 1).float()
        mock_drop.side_effect = lambda x: x

        mock_rnn = MagicMock(spec=nn.Module)
        mock_rnn.side_effect = lambda x, _: x

        encoder.rnn.forward_cells = nn.ModuleList([mock_rnn])
        encoder.rnn.backward_cells = nn.ModuleList([mock_rnn])

        out, context = encoder(input, input_len)

        assert out.shape[0] == B
        assert out.shape[1] == max_src_len
        assert out.shape[2] == 1

        assert torch.equal(input * 2, out.squeeze(2).int())
        assert context is None


@torch.no_grad()
def test_RNNEncoder_ix_mapping_concat():
    params = get_enc_params(n_layers=1, dropout=0, merge_method=MergeMethod.CONCAT)
    encoder = RNNEncoder(params)

    input = torch.tensor(
        [[1, 2, 3, 0, 0], [11, 12, 13, 14, 0], [21, 22, 23, 24, 25], [31, 32, 33, 0, 0]], dtype=torch.long
    )
    input_len = torch.tensor([3, 4, 5, 3], dtype=torch.int)
    B, max_src_len = input.shape

    with (
        patch.object(encoder, "emb", autospec=True) as mock_emb,
        patch.object(encoder, "dropout", autospec=True) as mock_drop,
    ):
        mock_emb.side_effect = lambda x: x.view(*x.shape[:2], 1).float()
        mock_drop.side_effect = lambda x: x

        mock_rnn = MagicMock(spec=nn.Module)
        mock_rnn.side_effect = lambda x, _: x

        encoder.rnn.forward_cells = nn.ModuleList([mock_rnn])
        encoder.rnn.backward_cells = nn.ModuleList([mock_rnn])

        out, context = encoder(input, input_len)

        assert out.shape[0] == B
        assert out.shape[1] == max_src_len
        assert out.shape[2] == 2

        assert torch.equal(input, out.mean(dim=2).int())
        assert context is None


@torch.no_grad()
def test_RNNEncoder_ix_mapping_sum_2_layers():
    params = get_enc_params(n_layers=2, dropout=0, n_hidden=1, merge_method=MergeMethod.SUM)
    encoder = RNNEncoder(params)

    input = torch.tensor(
        [[1, 2, 3, 0, 0], [11, 12, 13, 14, 0], [21, 22, 23, 24, 25], [31, 32, 33, 0, 0]], dtype=torch.long
    )
    input_len = torch.tensor([3, 4, 5, 3], dtype=torch.int)
    B, max_src_len = input.shape

    with (
        patch.object(encoder, "emb", autospec=True) as mock_emb,
        patch.object(encoder, "dropout", autospec=True) as mock_drop,
    ):
        mock_emb.side_effect = lambda x: x.view(*x.shape[:2], 1).float()
        mock_drop.side_effect = lambda x: x

        mock_rnn = MagicMock(spec=nn.Module)
        mock_rnn.side_effect = lambda x, _: x

        mock_rnn_layer2 = MagicMock(spec=nn.Module)
        mock_rnn_layer2.side_effect = lambda x, _: x[:, [0]]

        encoder.rnn.forward_cells = nn.ModuleList([mock_rnn, mock_rnn_layer2])
        encoder.rnn.backward_cells = nn.ModuleList([mock_rnn, mock_rnn_layer2])

        out, context = encoder(input, input_len)

        assert out.shape[0] == B
        assert out.shape[1] == max_src_len
        assert out.shape[2] == 1

        assert torch.equal(input * 2, out.squeeze(2).int())
        assert context is None


@torch.no_grad()
def test_RNNEncoder_ix_mapping_concat_2_layers():
    params = get_enc_params(n_layers=2, dropout=0, n_hidden=1, merge_method=MergeMethod.CONCAT)
    encoder = RNNEncoder(params)

    input = torch.tensor(
        [[1, 2, 3, 0, 0], [11, 12, 13, 14, 0], [21, 22, 23, 24, 25], [31, 32, 33, 0, 0]], dtype=torch.long
    )
    input_len = torch.tensor([3, 4, 5, 3], dtype=torch.int)
    B, max_src_len = input.shape

    with (
        patch.object(encoder, "emb", autospec=True) as mock_emb,
        patch.object(encoder, "dropout", autospec=True) as mock_drop,
    ):
        mock_emb.side_effect = lambda x: x.view(*x.shape[:2], 1).float()
        mock_drop.side_effect = lambda x: x

        mock_rnn = MagicMock(spec=nn.Module)
        mock_rnn.side_effect = lambda x, _: x

        mock_rnn_layer2 = MagicMock(spec=nn.Module)
        mock_rnn_layer2.side_effect = lambda x, _: x[:, [0]]

        encoder.rnn.forward_cells = nn.ModuleList([mock_rnn, mock_rnn_layer2])
        encoder.rnn.backward_cells = nn.ModuleList([mock_rnn, mock_rnn_layer2])

        out, context = encoder(input, input_len)

        assert out.shape[0] == B
        assert out.shape[1] == max_src_len
        assert out.shape[2] == 2

        assert torch.equal(input, out.mean(dim=2).int())
        assert context is None


@torch.no_grad()
def test_RNNEncoder_DropoutMixin_init_masks_not_training():
    params = get_enc_params(n_layers=2, dropout=0.2)
    encoder = RNNEncoder(params)
    encoder.eval()

    masks = encoder.rnn.init_masks(4, "cpu")
    assert masks is None


@torch.no_grad()
def test_RNNEncoder_DropoutMixin_init_masks_0():
    params = get_enc_params(n_layers=2, dropout=0)
    encoder = RNNEncoder(params)
    encoder.train()

    masks = encoder.rnn.init_masks(4, "cpu")
    assert masks is None


@torch.no_grad()
def test_RNNEncoder_DropoutMixin_init_masks_single_layer():
    params = get_enc_params(n_layers=1, dropout=0.2)
    encoder = RNNEncoder(params)
    encoder.train()

    masks = encoder.rnn.init_masks(4, "cpu")
    assert masks is None


@torch.no_grad()
@pytest.mark.parametrize("n_layers", [2, 3, 4])
@pytest.mark.parametrize("dropout", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
def test_RNNEncoder_DropoutMixin_init_masks(n_layers, dropout):
    B = 128
    n_hidden = 256
    params = get_enc_params(n_layers=n_layers, dropout=dropout, n_hidden=n_hidden)
    encoder = RNNEncoder(params)
    encoder.train()

    masks = encoder.rnn.init_masks(B, "cpu")
    assert masks is not None
    assert masks.shape[0] == n_layers - 1
    assert masks.shape[1] == B
    assert masks.shape[2] == n_hidden

    zero_ratio = (masks == 0).sum() / masks.numel()
    assert (zero_ratio > dropout - 0.01) and (zero_ratio < dropout + 0.01)


@torch.no_grad()
@pytest.mark.parametrize("n_layers", [2, 3, 4])
@pytest.mark.parametrize("dropout", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
def test_RNNEncoder_DropoutMixin_apply_dropout(n_layers, dropout):
    B = 128
    n_hidden = 256
    params = get_enc_params(n_layers=n_layers, dropout=dropout, n_hidden=n_hidden)
    encoder = RNNEncoder(params)
    encoder.train()

    masks = encoder.rnn.init_masks(B, "cpu")
    state = torch.rand((B, n_hidden))
    dropout_state = encoder.rnn.apply_dropout(state, masks[0])
    assert dropout_state is not None
    assert dropout_state.shape[0] == B
    assert dropout_state.shape[1] == n_hidden

    zero_ratio = (dropout_state == 0).sum() / dropout_state.numel()
    assert (zero_ratio > dropout - 0.01) and (zero_ratio < dropout + 0.01)

    orig_mean = state[dropout_state != 0].mean()
    dropout_mean = dropout_state[dropout_state != 0].mean()
    assert torch.allclose(orig_mean / dropout_mean, torch.tensor([1 - dropout]))


@torch.no_grad()
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("input_fixture", ["enc_input_single", "enc_input_mult"])
def test_RNNEncoder(n_layers, dropout, input_fixture, request):
    params = get_enc_params(n_layers=n_layers, dropout=dropout)
    encoder = RNNEncoder(params)

    input, input_len = request.getfixturevalue(input_fixture)
    B, max_src_len = input.shape

    out, context = encoder(input, input_len)

    assert out.shape == (B, max_src_len, params.n_hidden)
    assert context is None


@torch.no_grad()
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("input_fixture", ["enc_input_single", "enc_input_mult"])
def test_RNNEncoder_use_context(n_layers, input_fixture, request):
    params = get_enc_params(n_layers=n_layers, use_context=True)
    encoder = RNNEncoder(params)

    input, input_len = request.getfixturevalue(input_fixture)
    B, max_src_len = input.shape

    out, context = encoder(input, input_len)

    assert out.shape == (B, max_src_len, params.n_hidden)
    assert context.shape == (n_layers, B, params.n_hidden)


@torch.no_grad()
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("input_fixture", ["enc_input_single", "enc_input_mult"])
def test_RNNEncoder_concat(n_layers, input_fixture, request):
    params = get_enc_params(n_layers=n_layers, merge_method=MergeMethod.CONCAT)
    encoder = RNNEncoder(params)

    input, input_len = request.getfixturevalue(input_fixture)
    B, max_src_len = input.shape

    out, context = encoder(input, input_len)

    assert out.shape == (B, max_src_len, params.n_hidden * 2)
    assert context is None


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
def test_DotScorer(batch_size):
    n_hidden = 8
    max_src_len = 10
    scorer = DotScorer()

    queries = torch.rand((batch_size, 1, n_hidden))
    keys = torch.rand((batch_size, max_src_len, n_hidden))

    out = scorer(queries, keys)

    assert out.shape == (batch_size, 1, max_src_len)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
def test_GeneralScorer(batch_size):
    enc_n_hidden = 8
    dec_n_hidden = 8
    max_src_len = 10
    scorer = GeneralScorer(enc_n_hidden, dec_n_hidden)

    queries = torch.rand((batch_size, 1, dec_n_hidden))
    keys = torch.rand((batch_size, max_src_len, enc_n_hidden))

    out = scorer(queries, keys)

    assert out.shape == (batch_size, 1, max_src_len)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
def test_AdditiveScorer(batch_size):
    enc_n_hidden = 8
    dec_n_hidden = 8
    n_hidden = 8
    max_src_len = 10
    scorer = AdditiveScorer(enc_n_hidden, dec_n_hidden, n_hidden)

    queries = torch.rand((batch_size, 1, dec_n_hidden))
    keys = torch.rand((batch_size, max_src_len, enc_n_hidden))

    out = scorer(queries, keys)

    assert out.shape == (batch_size, 1, max_src_len)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
def test_ScaledDotScorer(batch_size):
    n_hidden = 8
    max_src_len = 10
    scorer = ScaledDotScorer(n_hidden)

    queries = torch.rand((batch_size, 1, n_hidden))
    keys = torch.rand((batch_size, max_src_len, n_hidden))

    out = scorer(queries, keys)

    assert out.shape == (batch_size, 1, max_src_len)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "score_method", [ScoreMethod.DOT, ScoreMethod.GENERAL, ScoreMethod.ADDITIVE, ScoreMethod.SCALED_DOT]
)
def test_Attention(batch_size, score_method):
    n_hidden = 8
    enc_n_hidden = 8
    max_src_len = 10

    att = Attention(score_method, n_hidden, enc_n_hidden)

    queries = torch.rand((batch_size, 1, n_hidden))
    keys = torch.rand((batch_size, max_src_len, enc_n_hidden))
    values = torch.rand((batch_size, max_src_len, enc_n_hidden))
    mask = torch.randint(high=2, size=(batch_size, max_src_len))

    context, attention = att(queries, keys, values, mask)

    assert context.shape == (batch_size, 1, n_hidden)
    assert attention.shape == (batch_size, 1, max_src_len)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_DecoderGRU(batch_size, n_layers, dropout):
    n_emb = 4
    n_hidden = 8
    max_tgt_len = 10

    decoder_gru = DecoderGRU(n_emb, n_hidden, n_layers, dropout)

    X = torch.rand((max_tgt_len, batch_size, n_emb))
    h_0 = torch.rand((n_layers, batch_size, n_hidden))

    output, h_n = decoder_gru(X, h_0)

    assert output.shape == (max_tgt_len, batch_size, n_hidden)
    assert h_n.shape == (n_layers, batch_size, n_hidden)


def get_dec_params(n_layers=2, dropout=0.1, att_score_method=None, input_feeding=None):
    return RNNDecoderParams(
        vocab_size=10,
        n_layers=n_layers,
        dropout=dropout,
        pad_ix=0,
        n_emb=4,
        n_hidden=8,
        att_score_method=att_score_method,
        input_feeding=input_feeding,
    )


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("hidden_size", [4, 8, 16])
@pytest.mark.parametrize("max_src_len", [12, 16, 20])
@pytest.mark.parametrize("num_queries", [1, 2])
def test_scorer_DotScorer(batch_size, hidden_size, max_src_len, num_queries):
    scorer = DotScorer()

    queries = torch.rand((batch_size, num_queries, hidden_size))
    keys = torch.rand((batch_size, max_src_len, hidden_size))

    score = scorer(queries, keys)
    assert score.shape == (batch_size, num_queries, max_src_len)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("hidden_size", [4, 8, 16])
@pytest.mark.parametrize("max_src_len", [12, 16, 20])
@pytest.mark.parametrize("num_queries", [1, 2])
def test_scorer_ScaledDotScorer(batch_size, hidden_size, max_src_len, num_queries):
    scorer = ScaledDotScorer(hidden_size)

    queries = torch.rand((batch_size, num_queries, hidden_size))
    keys = torch.rand((batch_size, max_src_len, hidden_size))

    score = scorer(queries, keys)
    assert score.shape == (batch_size, num_queries, max_src_len)


@torch.no_grad()
def test_Attention_matcher():
    methods = [ScoreMethod.DOT, ScoreMethod.GENERAL, ScoreMethod.ADDITIVE, ScoreMethod.SCALED_DOT]
    scorers = []

    for method in methods:
        att = Attention(method, 4, 4)
        scorers.append(att.scorer)

    assert isinstance(scorers[0], DotScorer)
    assert isinstance(scorers[1], GeneralScorer)
    assert isinstance(scorers[2], AdditiveScorer)
    assert isinstance(scorers[3], ScaledDotScorer)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_layers", [1, 2])
def test_SutskeverDecoder(batch_size, n_layers):
    params = get_dec_params(n_layers=n_layers)

    decoder = SutskeverRNNDecoder(params)

    input = torch.randint(high=params.vocab_size, size=(batch_size,))
    prev_hidden = torch.rand((n_layers, batch_size, params.n_hidden))

    fc_out, h_n, _ = decoder(input, prev_hidden)

    assert fc_out.shape == (batch_size, params.vocab_size)
    assert h_n.shape == (n_layers, batch_size, params.n_hidden)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_layers", [1, 2])
def test_BahdanauDecoder(batch_size, n_layers):
    max_src_len = 10
    params = get_dec_params(n_layers=n_layers, att_score_method=ScoreMethod.ADDITIVE)

    decoder = BahdanauRNNDecoder(params)

    input = torch.randint(high=params.vocab_size, size=(batch_size,))
    prev_hidden = torch.rand((n_layers, batch_size, params.n_hidden))
    enc_outputs = torch.rand((batch_size, max_src_len, params.n_hidden * 2))
    mask = torch.randint(high=2, size=(batch_size, max_src_len))

    fc_out, h_n, att = decoder(input, prev_hidden, enc_outputs, mask)

    assert fc_out.shape == (batch_size, params.vocab_size)
    assert h_n.shape == (n_layers, batch_size, params.n_hidden)
    assert att.shape == (batch_size, max_src_len)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("input_feeding", [False, True])
def test_LuongDecoder(batch_size, n_layers, input_feeding):
    max_src_len = 10
    params = get_dec_params(n_layers=n_layers, att_score_method=ScoreMethod.DOT, input_feeding=input_feeding)

    decoder = LuongRNNDecoder(params)

    input = torch.randint(high=params.vocab_size, size=(batch_size,))
    prev_hidden = torch.rand((n_layers, batch_size, params.n_hidden))
    enc_outputs = torch.rand((batch_size, max_src_len, params.n_hidden))
    mask = torch.randint(high=2, size=(batch_size, max_src_len))
    prev_mod_hidden = torch.rand((batch_size, params.n_hidden)) if input_feeding else None

    fc_out, (h_n, mod_hidden), att = decoder(input, prev_hidden, enc_outputs, mask, prev_mod_hidden)

    assert fc_out.shape == (batch_size, params.vocab_size)
    assert h_n.shape == (n_layers, batch_size, params.n_hidden)
    assert mod_hidden.shape == (batch_size, params.n_hidden)
    assert att.shape == (batch_size, max_src_len)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("teacher_forcing_rate", [0.0, 0.1])
def test_RNNSeq2Seq_Sutskever(batch_size, n_layers, teacher_forcing_rate):
    config = SutskeverConfig(n_layers=n_layers, dropout=0.1, n_emb=4, n_hidden=8)
    dataset = MockDataset()

    max_src_len = 7
    max_tgt_len = 9

    model = build_model(config, dataset, "cpu")

    assert isinstance(model, RNNSeq2Seq)
    assert isinstance(model.encoder, RNNEncoder)
    assert isinstance(model.decoder, SutskeverRNNDecoder)

    src = torch.randint(high=dataset.src_vocab_size, size=(batch_size, max_src_len))
    src_len = torch.randint(low=4, high=max_src_len, size=(batch_size,))
    tgt = torch.randint(high=dataset.tgt_vocab_size, size=(max_tgt_len, batch_size))

    out = model(src, src_len, tgt, teacher_forcing_rate)

    assert out.shape == (max_tgt_len, batch_size, dataset.tgt_vocab_size)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("teacher_forcing_rate", [0.0, 0.1])
@pytest.mark.parametrize("use_context", [True, False])
def test_RNNSeq2Seq_Bahdanau(batch_size, n_layers, teacher_forcing_rate, use_context):
    config = BahdanauConfig(n_layers=n_layers, dropout=0.1, n_emb=4, n_hidden=8, use_context=use_context)
    dataset = MockDataset()

    max_src_len = 7
    max_tgt_len = 9

    model = build_model(config, dataset, "cpu")

    assert isinstance(model, RNNSeq2Seq)
    assert isinstance(model.encoder, RNNEncoder)
    assert isinstance(model.decoder, BahdanauRNNDecoder)

    src = torch.randint(high=dataset.src_vocab_size, size=(batch_size, max_src_len))
    src_len = torch.full((batch_size,), max_src_len)
    tgt = torch.randint(high=dataset.tgt_vocab_size, size=(max_tgt_len, batch_size))

    out = model(src, src_len, tgt, teacher_forcing_rate)

    assert out.shape == (max_tgt_len, batch_size, dataset.tgt_vocab_size)


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_layers", [1, 2])
@pytest.mark.parametrize("teacher_forcing_rate", [0.0, 0.1])
@pytest.mark.parametrize("use_context", [True, False])
@pytest.mark.parametrize("att_score_method", ["dot", "general", "additive", "scaled_dot"])
@pytest.mark.parametrize("input_feeding", [True, False])
def test_RNNSeq2Seq_Luong(batch_size, n_layers, teacher_forcing_rate, use_context, att_score_method, input_feeding):
    config = LuongConfig(
        n_layers=n_layers,
        dropout=0.1,
        n_emb=4,
        n_hidden=8,
        use_context=use_context,
        att_score_method=att_score_method,
        input_feeding=input_feeding,
    )
    dataset = MockDataset()

    max_src_len = 7
    max_tgt_len = 9

    model = build_model(config, dataset, "cpu")

    assert isinstance(model, RNNSeq2Seq)
    assert isinstance(model.encoder, RNNEncoder)
    assert isinstance(model.decoder, LuongRNNDecoder)

    src = torch.randint(high=dataset.src_vocab_size, size=(batch_size, max_src_len))
    src_len = torch.full((batch_size,), max_src_len)
    tgt = torch.randint(high=dataset.tgt_vocab_size, size=(max_tgt_len, batch_size))

    out = model(src, src_len, tgt, teacher_forcing_rate)

    assert out.shape == (max_tgt_len, batch_size, dataset.tgt_vocab_size)
