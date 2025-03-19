from dataclasses import dataclass


@dataclass
class MockDataset:
    src_vocab_size: int = 10
    src_pad_ix: int = 0
    tgt_vocab_size: int = 10
    tgt_pad_ix: int = 0
    max_src_len: int = 20
    max_tgt_len: int = 20
