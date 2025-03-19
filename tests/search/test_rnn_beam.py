import torch

from torchnmt.model.config import RNNDecoderParams, ScoreMethod
from torchnmt.model.rnn import LuongRNNDecoder, SutskeverRNNDecoder
from torchnmt.search.rnn_beam import RNNBeamSearcher, RNNBeamSearchNode


def test_node_creation_first():
    node = RNNBeamSearchNode(torch.rand(4), None, torch.tensor([3], dtype=torch.long), 0, None)

    assert node.length == 1
    assert node.val == 0


def test_node_creation():
    start_node = RNNBeamSearchNode(torch.rand(4), None, torch.tensor([3], dtype=torch.long), 0, None)

    node = RNNBeamSearchNode(torch.rand(4), start_node, torch.tensor([3], dtype=torch.long), -0.2, None)

    assert node.length == 2
    assert node.val == 0.2


def test_node_creation_chain():
    def create_node(ix, prev, logp):
        return RNNBeamSearchNode(torch.rand(4), prev, torch.tensor([ix], dtype=torch.long), logp, None)

    n0 = create_node(0, None, 0)
    n1 = create_node(1, n0, -0.2)
    n2 = create_node(2, n1, -0.5)
    n3 = create_node(3, n2, -0.9)

    assert n3.length == 4
    assert n3.val == 0.3


def get_dec_params(n_layers=2, att_score_method=None, input_feeding=None, n_hidden=8):
    return RNNDecoderParams(
        vocab_size=10,
        n_layers=n_layers,
        dropout=0.0,
        pad_ix=0,
        n_emb=4,
        n_hidden=n_hidden,
        att_score_method=att_score_method,
        input_feeding=input_feeding,
    )


def test_create_start_node():
    B = 4
    n_layers = 2
    H = 8
    start_ix = 1
    input = torch.tensor([start_ix]).repeat(1, B)
    prev_hidden = torch.rand((n_layers, H))
    decoder_kwargs = dict(prev_hidden=prev_hidden)
    batch_idx = 2

    params = get_dec_params(n_layers=n_layers, n_hidden=H)

    decoder = SutskeverRNNDecoder(params)

    searcher = RNNBeamSearcher(decoder, None, None, None)

    start_node = searcher._create_start_node(input, decoder_kwargs, batch_idx)

    assert torch.allclose(prev_hidden[:, batch_idx], start_node.hidden)
    assert start_node.prev is None
    assert start_node.ix.item() == start_ix
    assert start_node.logp == 0.0
    assert start_node.att is None
    assert not hasattr(start_node, "mod_hidden")


def test_create_start_node_luong_input_feeding():
    B = 4
    n_layers = 2
    H = 8
    start_ix = 1
    input = torch.tensor([start_ix]).repeat(1, B)
    prev_hidden = torch.rand((n_layers, H))
    mod_hidden = torch.rand((B, H))
    decoder_kwargs = dict(prev_hidden=prev_hidden, prev_mod_hidden=mod_hidden)
    batch_idx = 2

    params = get_dec_params(n_layers=n_layers, n_hidden=H, input_feeding=True, att_score_method=ScoreMethod.DOT)

    decoder = LuongRNNDecoder(params)

    searcher = RNNBeamSearcher(decoder, None, None, None)

    start_node = searcher._create_start_node(input, decoder_kwargs, batch_idx)

    assert torch.allclose(prev_hidden[:, batch_idx], start_node.hidden)
    assert start_node.prev is None
    assert start_node.ix.item() == start_ix
    assert start_node.logp == 0.0
    assert start_node.att is None
    assert torch.allclose(mod_hidden[batch_idx], start_node.mod_hidden)
