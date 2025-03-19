import random

import pytest
import torch
import torch.nn as nn

from tests.utils import MockDataset
from torchnmt.forward import RNNForward, TransformerForward
from torchnmt.model.config import BahdanauConfig, LuongConfig, ScoreMethod, SutskeverConfig, TransformerConfig
from torchnmt.model.model import build_model


@pytest.fixture
def sutskever_config():
    return SutskeverConfig(n_layers=2, dropout=0.1, n_emb=4, n_hidden=8)


@pytest.fixture
def bahdanau_config():
    return BahdanauConfig(n_layers=2, dropout=0.1, n_emb=4, n_hidden=8, use_context=False)


@pytest.fixture
def luong_config():
    return LuongConfig(
        n_layers=2,
        dropout=0.1,
        n_emb=4,
        n_hidden=8,
        use_context=False,
        att_score_method=ScoreMethod.DOT,
        input_feeding=False,
    )


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("max_src_len", [3, 6, 9])
@pytest.mark.parametrize("max_tgt_len", [4, 6, 8])
@pytest.mark.parametrize("config_fixture", ["sutskever_config", "bahdanau_config", "luong_config"])
@pytest.mark.parametrize("teacher_forcing_rate", [None, 0.1])
def test_RNNForward(batch_size, max_src_len, max_tgt_len, config_fixture, teacher_forcing_rate, request):
    config = request.getfixturevalue(config_fixture)
    dataset = MockDataset()

    model = build_model(config, dataset, "cpu")

    src = torch.randint(high=dataset.src_vocab_size, size=(batch_size, max_src_len))
    src_len = torch.randint(low=1, high=max_src_len + 1, size=(batch_size,))
    src_len[random.randint(0, len(src_len) - 1)] = torch.tensor(max_src_len)  # at least one has to be max len
    tgt = torch.randint(high=dataset.tgt_vocab_size, size=(max_tgt_len, batch_size))

    forward = RNNForward(model, "cpu", nn.CrossEntropyLoss(ignore_index=dataset.tgt_pad_ix))

    if teacher_forcing_rate is None:
        loss, cnt = forward((src, src_len, tgt, None))
    else:
        loss, cnt = forward((src, src_len, tgt, None), teacher_forcing_rate=teacher_forcing_rate)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    assert isinstance(cnt, int)
    assert cnt == (tgt[1:] != dataset.tgt_pad_ix).sum().item()


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("max_src_len", [3, 6, 9])
@pytest.mark.parametrize("max_tgt_len", [4, 6, 8])
def test_TransformerForward(batch_size, max_src_len, max_tgt_len):
    config = TransformerConfig(
        n_layers=2, dropout=0.1, d_model=8, n_heads=4, d_ff=16, sine_pos=True, pre_norm=True, naive=False
    )
    dataset = MockDataset()

    model = build_model(config, dataset, "cpu")

    src = torch.randint(high=dataset.src_vocab_size, size=(batch_size, max_src_len))
    tgt = torch.randint(high=dataset.tgt_vocab_size, size=(batch_size, max_tgt_len))

    forward = TransformerForward(model, "cpu", nn.CrossEntropyLoss(ignore_index=dataset.tgt_pad_ix))

    loss, cnt = forward((src, tgt, None))

    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    assert isinstance(cnt, int)
    assert cnt == (tgt[:, 1:] != dataset.tgt_pad_ix).sum().item()
