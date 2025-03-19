from abc import ABC, abstractmethod
from typing import Annotated, Literal, Self, Union

from pydantic import AfterValidator, BaseModel, Field, PositiveInt, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class Sub:
    """
    A class representing a substitution pattern and its replacement string.

    Attributes:
        ptn (str): The regex pattern to be matched.
        repl (str): The replacement string for the matched pattern.
    """

    ptn: str
    repl: str


@pydantic_dataclass
class LangPatterns:
    """
    A class for managing language-specific regex patterns and substitutions.

    Attributes:
        omit (str | None): An optional regex pattern to identify translation examples that should be omitted.
        subs (list[Sub] | None): An optional list of `Sub` objects representing substitution patterns
                                 and their corresponding replacement strings.
    """

    omit: str | None = None
    subs: list[Sub] | None = None


class Filter(ABC):
    """
    An abstract base class representing a filter for tokenized translation examples.

    Subclasses must implement the `filter` method to define custom filtering logic.
    """

    @abstractmethod
    def filter(self, tok: list[str]) -> bool:
        """
        Determines whether the given list of tokens meets the filter's criteria.

        Args:
            tok (list[str]): A list of strings representing tokens to be evaluated.

        Returns:
            bool: True if the tokens satisfy the filter criteria, False otherwise.
        """
        pass


class LenFilter(Filter, BaseModel):
    """
    A filter that checks whether the length of a list of tokens is within a specified maximum limit.

    Inherits from:
        - Filter: Abstract base class for token filters.
        - BaseModel: Pydantic base model for data validation.

    Attributes:
        max_len (PositiveInt): The maximum allowed length of the list of tokens.
    """

    max_len: PositiveInt

    def filter(self, tok: list[str]) -> bool:
        return len(tok) <= self.max_len


class StartFilter(Filter, BaseModel):
    """
    A filter that checks whether the first token in a list starts with any of the specified strings.

    Inherits from:
        - Filter: Abstract base class for token filters.
        - BaseModel: Pydantic base model for data validation.

    Attributes:
        start_with (list[str]): A list of strings that the first token should match.
    """

    start_with: list[str]

    def filter(self, tok: list[str]) -> bool:
        return tok[0] in self.start_with


class FilterChain(BaseModel):
    """
    A class that applies a chain of filters to a list of tokens, ensuring all filters are satisfied.

    Attributes:
        filters (list[Union[LenFilter, StartFilter]]):
            A list of filter objects that will be applied to the token list.

    Methods:
        __call__(tok: list[str]) -> bool:
            Evaluates the token list against all filters in the chain.
            Returns `False` if any filter fails, otherwise `True`.

    Usage:
        Create an instance of `FilterChain` by providing a list of filter objects.
        Use the instance as a callable to evaluate token lists against the chain of filters.
    """

    filters: list[Union[LenFilter, StartFilter]]

    def __call__(self, tok: list[str]) -> bool:
        """
        Evaluates a list of tokens against all filters in the chain.

        Args:
            tok (list[str]): A list of tokens.

        Returns:
            bool: True if all specified filters are satisfied, False otherwise.
        """

        for f in self.filters:
            if not f.filter(tok):
                return False
        return True


@pydantic_dataclass
class Specials:
    """
    A class representing special tokens.

    Attributes:
        pad (str): The special token used for padding.
        start (str): The special token indicating the start of a sequence.
        end (str): The special token indicating the end of a sequence.
    """

    pad: str
    start: str
    end: str

    @model_validator(mode="after")
    def check(self) -> Self:
        """
        Validates that all special tokens are unique.

        Raises:
            ValueError: If any duplicates are found.
        """

        if len(self.__dict__) != len(set(self.__dict__.values())):
            raise ValueError("duplicate special token(s)")
        return self


@pydantic_dataclass
class WordSpecials(Specials):
    """
    A subclass of `Specials` that includes an additional special token for unknown words, to
    be used with word-level tokenization.

    Attributes:
        unk (str): The special token used for unknown words.
    """

    unk: str


@pydantic_dataclass(kw_only=True)
class TokenizerConfig:
    """
    Base configuration class for all tokenizers.

    Attributes:
        regex (str): A regex pattern used for tokenizing the text.
        take_first (PositiveInt | None):
            An optional parameter specifying the maximum number of examples to retain.
            If `None`, all examples are retained. Defaults to `None`.
    """

    regex: str
    take_first: PositiveInt | None = None


@pydantic_dataclass(kw_only=True)
class WordTokenizerConfig(TokenizerConfig):
    """
    Configuration class for a word-level tokenizer.

    Attributes:
        specials (WordSpecials): A collection of special tokens - padding, start, end, and unknown tokens.
        min_freq (PositiveInt): The minimum frequency a word must appear to be included in the vocabulary.
        lower (bool): Whether to convert all input text to lowercase during tokenization.
        unicode_normalize (bool): Whether to normalize Unicode characters.
        patterns (dict[str, LangPatterns] | None):
            A dictionary mapping language codes to language-specific patterns and substitution rules.
            Defaults to `None` if no patterns are specified.
        kind (Literal["word"]): Specifies the tokenizer type. Always set to "word" for this configuration.
        src_filters (FilterChain | None):
            An optional chain of filters applied to the source text during tokenization. Defaults to `None`.
        tgt_filters (FilterChain | None):
            An optional chain of filters applied to the target text during tokenization. Defaults to `None`.

    Inherits:
        TokenizerConfig: Base configuration for tokenization.
    """

    specials: WordSpecials
    min_freq: PositiveInt
    lower: bool
    unicode_normalize: bool
    patterns: dict[str, LangPatterns] | None = None
    kind: Literal["word"] = "word"
    src_filters: FilterChain | None = None
    tgt_filters: FilterChain | None = None


@pydantic_dataclass(kw_only=True)
class BPETokenizerConfig(TokenizerConfig):
    """
    Configuration class for a Byte Pair Encoding (BPE) tokenizer.

    Attributes:
        specials (Specials): A collection of special tokens - padding, start, and end tokens.
        src_vocab_size (PositiveInt): The size of the source language vocabulary to be generated by the tokenizer.
        tgt_vocab_size (PositiveInt): The size of the target language vocabulary to be generated by the tokenizer.
        kind (Literal["bpe"]): Specifies the tokenizer type. Always set to "bpe" for this configuration.

    Inherits:
        TokenizerConfig: Base configuration for tokenization.
    """

    specials: Specials
    src_vocab_size: PositiveInt
    tgt_vocab_size: PositiveInt
    verbose: bool = False
    kind: Literal["bpe"] = "bpe"


@pydantic_dataclass(kw_only=True)
class LoadDatasetConfig:
    """
    Configuration class for loading a dataset from a specified directory.

    Attributes:
        dir_ (str): The directory where the dataset is located.
        name (str): The name of the dataset to be loaded.
        kind (Literal["load"]): Specifies the configuration type. Always set to "load" for this configuration.
    """

    dir_: str
    name: str
    kind: Literal["load"] = "load"


VALID_SOURCES = {"tatoeba", "TED2020", "SciELO"}


def validate_sources(sources):
    if sources is None or len(sources) == 0 or len(set(sources)) < len(sources):
        raise ValueError("Invalid sources list")
    for source in sources:
        if source not in VALID_SOURCES:
            raise ValueError("Invalid sources list")
    return sources


Sources = Annotated[list[str], AfterValidator(validate_sources)]


@pydantic_dataclass(kw_only=True)
class BuildDatasetConfig:
    """
    Configuration class for building an NMTDataset with specified parameters, including source and target languages,
    tokenizer settings, and other options.

    Attributes:
        dir_ (str): The directory where the dataset will be saved.
        name (str): The name of the dataset being built.
        src_lang (str): The source language code.
        tgt_lang (str): The target language code.
        source (str): The Source from which the dataset will be built.
        save (bool): Whether to save the built dataset to the specified directory.
        seed (PositiveInt): A seed value for reproducibility in dataset creation.
        tokenizer (Union[WordTokenizerConfig, BPETokenizerConfig]):
            The tokenizer configuration, either word-level or BPE, with type differentiation
            via the `kind` field.
        kind (Literal["build"]): Specifies the configuration type. Always set to "build" for this configuration.
    """

    dir_: str
    name: str
    src_lang: str
    tgt_lang: str
    sources: Sources
    save: bool
    seed: PositiveInt
    tokenizer: Union[WordTokenizerConfig, BPETokenizerConfig] = Field(discriminator="kind")
    kind: Literal["build"] = "build"


DatasetConfig = Union[LoadDatasetConfig, BuildDatasetConfig]
