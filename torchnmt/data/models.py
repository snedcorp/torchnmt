import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Self, TypedDict

import torch
import yaml
from pydantic import RootModel
from torch import Tensor
from torch.utils.data import Dataset

from .config import BuildDatasetConfig, LoadDatasetConfig


class Example(TypedDict):
    """
    A TypedDict representing a translation example in an NMTDataset, containing the source and target strings
    along with their tokenized and indexed forms.

    Attributes:
        src (str): The source language text.
        tgt (str): The target language text.
        src_tok (list[str]): A list of tokens derived from the source language text.
        tgt_tok (list[str]): A list of tokens derived from the target language text.
        src_ix (list[int]): A list of integer indices corresponding to the source tokens.
        tgt_ix (list[int]): A list of integer indices corresponding to the target tokens.
    """

    src: str
    tgt: str
    src_tok: list[str]
    tgt_tok: list[str]
    src_ix: list[int]
    tgt_ix: list[int]


class WordExample(Example):
    """
    A TypedDict representing a translation example in an NMTDataset when word-level tokenization is used.
    Extends the Example structure with additional fields containing the string tokens before vocabularies
    are constructed (i.e. before low-frequency words are replaced with unknown tokens).

    Attributes:
        src_tok_pre_vocab (list[str]): A list of tokens from the source text before vocabulary construction.
        tgt_tok_pre_vocab (list[str]): A list of tokens from the target text before vocabulary construction.
    """

    src_tok_pre_vocab: list[str]
    tgt_tok_pre_vocab: list[str]


class NMTDatasetSplit(Dataset):
    """
    A dataset split for neural machine translation (NMT), designed to work with PyTorch's data utilities.

    Attributes:
        data (list[Example]): A list of examples, each represented by the `Example` structure.
        len (int): The total number of examples in the split.
    """

    def __init__(self, data: list[Example]):
        self.data = data
        self.len = len(data)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Example]:
        """
        Retrieves the source and target token indices as tensors, along with the original example data.

        Args:
            i (int): The index of the example to retrieve.

        Returns:
            tuple[Tensor, Tensor, Example]:
                Tensor: Token indices of source sequence.
                    - Shape: (src_len,)
                Tensor: Token indices of target sequence.
                    - Shape: (tgt_len,)
                Example: The original example data corresponding to the specified index.
        """

        src_ix = torch.tensor(self.data[i]["src_ix"], dtype=torch.long)
        tgt_ix = torch.tensor(self.data[i]["tgt_ix"], dtype=torch.long)

        return src_ix, tgt_ix, self.data[i]

    def __len__(self):
        return self.len


class NMTTokenizer(ABC):
    """
    An abstract base class for neural machine translation (NMT) tokenizers,
    defining the interface for encoding and decoding tokens.

    Methods:
        encode(text: str) -> list[int]:
            Abstract method to encode a given text into a list of integer token IDs.
            Must be implemented in subclasses.

        decode(tokens: list[int], strip_specials: bool = False) -> str:
            Abstract method to decode a list of integer token IDs back into a string.
            Special tokens can be optionally stripped. Must be implemented in subclasses.

        decode_tokens(tokens: list[int], strip_specials: bool = False) -> list[str]:
            Abstract method to decode a list of integer token IDs into a list of corresponding tokens (strings).
            Special tokens can be optionally stripped. Must be implemented in subclasses.

        vocab_size() -> int:
            Abstract property that returns the size of the vocabulary. Must be implemented in subclasses.
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Encodes a text string into a list of integer tokens.

        Args:
            text (str): The input text to be encoded.

        Returns:
            list[int]: A list of integer tokens representing the input text,
            including the special "start" and "end" tokens.
        """

        pass

    @abstractmethod
    def decode(self, tokens: list[int], strip_specials: bool = False) -> str:
        """
        Decodes a list of integer tokens into its corresponding string representation.

        Args:
            tokens (list[int]): A list of integer tokens to decode.
            strip_specials (bool, optional): If `True`, special tokens are excluded from the output.
                                            Defaults to `False`.

        Returns:
            str: A single string composed of the decoded tokens, optionally excluding special tokens.
        """

        pass

    @abstractmethod
    def decode_tokens(self, tokens: list[int], strip_specials: bool = False) -> list[str]:
        """
        Decodes a list of integer tokens into a list of their corresponding string tokens.

        Args:
            tokens (list[int]): A list of integer tokens to decode.
            strip_specials (bool, optional): If `True`, special tokens are excluded from the output.
                                            Defaults to `False`.

        Returns:
            list[str]: A list of string tokens corresponding to the input integer tokens,
                    with special tokens optionally removed.
        """

        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        The size of the vocabulary.
        """

        pass


@dataclass
class NMTDataset:
    """
    Represents a complete dataset for neural machine translation (NMT), including tokenizers, data splits, and configuration.

    Attributes:
        name (str): The name of the dataset.
        src_tokenizer (NMTTokenizer): The tokenizer for the source language.
        tgt_tokenizer (NMTTokenizer): The tokenizer for the target language.
        train_set (NMTDatasetSplit): The training data split.
        val_set (NMTDatasetSplit): The validation data split.
        test_set (NMTDatasetSplit): The test data split.
        config (BuildDatasetConfig): The configuration used to build the dataset.
        src_specials (dict[str, int]): Special tokens and their indices for the source language.
        tgt_specials (dict[str, int]): Special tokens and their indices for the target language.
        max_src_len (int): The maximum length found within the source sequences.
        max_tgt_len (int): The maximum length found within the target sequences.
    """

    name: str
    src_tokenizer: NMTTokenizer
    tgt_tokenizer: NMTTokenizer
    train_set: NMTDatasetSplit
    val_set: NMTDatasetSplit
    test_set: NMTDatasetSplit
    config: BuildDatasetConfig
    src_specials: dict[str, int]
    tgt_specials: dict[str, int]
    max_src_len: int
    max_tgt_len: int

    @property
    def src_vocab_size(self):
        """
        The size of the source language vocabulary.
        """

        return self.src_tokenizer.vocab_size

    @property
    def tgt_vocab_size(self):
        """
        The size of the target language vocabulary.
        """

        return self.tgt_tokenizer.vocab_size

    @property
    def src_pad_ix(self):
        """
        The index of the padding token in the source vocabulary.
        """

        return self.src_specials["pad"]

    @property
    def tgt_pad_ix(self):
        """
        The index of the padding token in the target vocabulary.
        """

        return self.tgt_specials["pad"]

    def save(self):
        """
        Saves the dataset and its configuration to the specified directory.
        """

        parent_path = Path(self.config.dir_)
        if not parent_path.exists():
            parent_path.mkdir(parents=True)
        path = parent_path / self.name
        path.mkdir(exist_ok=False)

        print(f"\nSaving dataset to: {path}")
        dataset_path = path / "dataset.pt"
        with open(dataset_path, "wb") as f:
            pickle.dump(self, f)

        config_path = path / "config.yaml"
        with open(config_path, "w") as config_file:
            yaml.dump(
                RootModel[BuildDatasetConfig](self.config).model_dump(mode="json", exclude_none=True), config_file
            )
        print("Dataset saved!\n")

    @classmethod
    def load(cls, config: LoadDatasetConfig) -> Self:
        """
        Loads a dataset from the specified configuration.

        Args:
            config (LoadDatasetConfig): The directory and file to load the dataset from.

        Returns:
            NMTDataset: The loaded dataset.
        """

        path = Path(config.dir_) / config.name / "dataset.pt"
        print(f"Loading dataset from: {path}")
        with open(path, "rb") as f:
            dataset = pickle.load(f)
        print("Dataset loaded!")
        return dataset
