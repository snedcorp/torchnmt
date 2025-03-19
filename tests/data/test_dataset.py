import torch
from torch import Tensor

from torchnmt.data.models import NMTDatasetSplit


def test_dataset_split():
    data = [
        {"src_ix": [1, 10, 20, 30, 40, 2], "tgt_ix": [1, 55, 45, 35, 25, 15, 2]},
        {"src_ix": [1, 100, 200, 300, 400, 2], "tgt_ix": [1, 550, 450, 350, 250, 2]},
    ]
    split = NMTDatasetSplit(data)

    assert len(split) == 2

    def is_correct(res, ex):
        assert len(res) == 3
        assert isinstance(res[0], Tensor)
        assert res[0].dtype == torch.long
        assert res[0].shape == (len(ex["src_ix"]),)
        assert torch.equal(res[0], torch.tensor(ex["src_ix"], dtype=torch.long))

        assert isinstance(res[1], Tensor)
        assert res[1].dtype == torch.long
        assert res[1].shape == (len(ex["tgt_ix"]),)
        assert torch.equal(res[1], torch.tensor(ex["tgt_ix"], dtype=torch.long))

        assert res[2] == ex

    is_correct(split[0], data[0])
    is_correct(split[1], data[1])
