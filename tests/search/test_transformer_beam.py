import torch

from torchnmt.search.transformer_beam import TransformerBeamSearchNode


def test_node_creation_first():
    node = TransformerBeamSearchNode.from_prev(None, torch.tensor([3], dtype=torch.long), 0, None)

    assert len(node.ix) == 1
    assert node.ix[0].item() == 3
    assert node.length == 1
    assert node.val == 0


def test_node_creation():
    start_node = TransformerBeamSearchNode.from_prev(None, torch.tensor([3], dtype=torch.long), 0, None)

    node = TransformerBeamSearchNode.from_prev(start_node, torch.tensor([4], dtype=torch.long), -0.2, None)

    assert len(node.ix) == 2
    assert torch.equal(node.ix, torch.tensor([3, 4], dtype=torch.long))
    assert node.length == 2
    assert node.val == 0.2


def test_node_creation_chain():
    def create_node(ix, prev, logp):
        return TransformerBeamSearchNode.from_prev(prev, torch.tensor([ix], dtype=torch.long), logp, None)

    n0 = create_node(0, None, 0)
    n1 = create_node(1, n0, -0.2)
    n2 = create_node(2, n1, -0.3)
    n3 = create_node(3, n2, -0.4)

    assert len(n3.ix) == 4
    assert torch.equal(n3.ix, torch.tensor([0, 1, 2, 3], dtype=torch.long))
    assert n3.length == 4
    assert n3.val == 0.3
