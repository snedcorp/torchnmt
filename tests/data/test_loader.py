import pytest
import torch

from torchnmt.data.loader import RNNCollator, TransformerCollator


@pytest.fixture
def raw_batch():
    ixs = [
        ([11, 12, 13, 14, 15], [11, 12, 13]),
        ([21, 22, 23, 24], [21, 22, 23, 24, 25, 26]),
        ([31, 32], [31, 32, 33, 34]),
        ([41, 42, 43, 44], [41, 42, 43, 44]),
        ([51, 52, 53, 54, 55, 56], [51, 52, 53, 54]),
    ]
    batch = []
    for ix in ixs:
        src_ix = torch.tensor(ix[0], dtype=torch.long)
        tgt_ix = torch.tensor(ix[1], dtype=torch.long)
        ex = dict(src_ix=ix[0], tgt_ix=ix[1])
        batch.append((src_ix, tgt_ix, ex))
    return batch


def test_rnn_collate(raw_batch):
    collator = RNNCollator(0, 0)
    batch = collator(raw_batch)

    assert len(batch) == 4

    src, src_len, tgt, examples = batch

    assert isinstance(src, torch.Tensor)
    assert src.shape == (5, 6)  # (B, max_src_len)

    assert isinstance(src_len, torch.Tensor)
    assert src_len.shape == (5,)
    assert src_len.tolist() == [5, 4, 2, 4, 6]

    for i, len_ in enumerate(src_len):
        assert torch.equal(src[i, :len_], raw_batch[i][0])
        assert torch.count_nonzero(src[i, len_:]) == 0

    assert isinstance(tgt, torch.Tensor)
    assert tgt.shape == (6, 5)  # (max_tgt_len, B)
    for i in range(tgt.shape[0]):
        for j in range(tgt.shape[1]):
            if tgt[i, j].item() == 0:
                with pytest.raises(IndexError):
                    assert tgt[i, j] == raw_batch[j][1][i]
            else:
                assert tgt[i, j] == raw_batch[j][1][i]

    for i, ex in enumerate(examples):
        assert ex == raw_batch[i][2]


def test_transformer_collate(raw_batch):
    collator = TransformerCollator(0, 0)
    batch = collator(raw_batch)

    assert len(batch) == 3

    src, tgt, examples = batch

    assert isinstance(src, torch.Tensor)
    assert src.shape == (5, 6)  # (B, max_src_len)

    src_len = [len(ex[0]) for ex in raw_batch]
    for i, len_ in enumerate(src_len):
        assert torch.equal(src[i, :len_], raw_batch[i][0])
        assert torch.count_nonzero(src[i, len_:]) == 0

    assert isinstance(tgt, torch.Tensor)
    assert tgt.shape == (5, 6)  # (B, max_tgt_len)

    tgt_len = [len(ex[1]) for ex in raw_batch]
    for i, len_ in enumerate(tgt_len):
        assert torch.equal(tgt[i, :len_], raw_batch[i][1])
        assert torch.count_nonzero(tgt[i, len_:]) == 0

    for i, ex in enumerate(examples):
        assert ex == raw_batch[i][2]
