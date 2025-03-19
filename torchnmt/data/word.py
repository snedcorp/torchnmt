import re
import unicodedata
from typing import TypedDict

import regex
from sklearn.model_selection import train_test_split  # type: ignore

from .config import BuildDatasetConfig, FilterChain, LangPatterns, WordTokenizerConfig
from .models import NMTDataset, NMTDatasetSplit, NMTTokenizer, WordExample
from .source import get_examples


class NormalizedExample(TypedDict):
    """
    A TypedDict representing a normalized example with source and target strings.

    Attributes:
        src (str): The normalized source string.
        tgt (str): The normalized target string.
    """

    src: str
    tgt: str


class PreVocabExample(NormalizedExample):
    """
    A TypedDict extending NormalizedExample to include the tokenized words from the source and target strings.

    Attributes:
        src_tok_pre_vocab (list[str]): The tokenized words for the source string.
        tgt_tok_pre_vocab (list[str]): The tokenized words for the target string.
    """

    src_tok_pre_vocab: list[str]
    tgt_tok_pre_vocab: list[str]


class Normalizer:
    """
    Utility class for normalizing translation examples.

    Supports case conversion, Unicode normalization, and pattern-based substitution and omissions.

    Attributes:
        src_lang (str): The source language code.
        tgt_lang (str): The target language code.
        lower (bool): Whether to lowercase text.
        unicode_normalize (bool): Whether to apply Unicode normalization and remove accents.
        patterns (dict[str, LangPatterns] | None): Optional language-specific patterns for substitution and omission.
    """

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        lower: bool,
        unicode_normalize: bool,
        patterns: dict[str, LangPatterns] | None = None,
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.lower = lower
        self.unicode_normalize = unicode_normalize
        self.patterns = patterns

    def normalize_example(self, example: tuple[str, str]) -> NormalizedExample | None:
        """
        Normalizes a pair of source and target strings.

        Args:
            example (tuple[str, str]): A tuple containing the raw source and target strings to be normalized.

        Returns:
            NormalizedExample | None: A dictionary with the normalized source and target strings
            (`{"src": src_str, "tgt": tgt_str}`) if both strings can be normalized; otherwise, `None`.
        """

        raw_src_str, raw_tgt_str = example
        src_str = self.normalize_str(raw_src_str, self.src_lang)
        if src_str is None:
            return None

        tgt_str = self.normalize_str(raw_tgt_str, self.tgt_lang)
        if tgt_str is not None:
            normalized_example: NormalizedExample = {"src": src_str, "tgt": tgt_str}
            return normalized_example
        return None

    def normalize_str(self, s: str, lang: str) -> str | None:
        """
        Normalizes a single string based, in part, on language-specific rules and patterns.

        Args:
            s (str): The string to be normalized.
            lang (str): The string's language code.

        Returns:
            str | None: The normalized string, or `None` if the string matches an omission pattern.
        """

        if self.lower:
            s = s.lower()
        if self.unicode_normalize:
            s = self._remove_accents(s)
        if self.patterns is None or lang not in self.patterns:
            return s

        lang_patterns = self.patterns[lang]
        if lang_patterns.omit:
            found = re.search(lang_patterns.omit, s)
            if found is not None:
                return None

        if lang_patterns.subs:
            for sub in lang_patterns.subs:
                s = re.sub(sub.ptn, sub.repl, s)
        return s

    def _remove_accents(self, s: str) -> str:
        """
        Performs Unicode normalization and accent removal on the given string.

        Args:
            s (str): The string to be normalized.

        Returns:
            str: The normalized string.
        """

        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"  # omit if non-spacing mark
        )


class FilterTokenizer:
    """
    A tokenizer that applies a regex-based, word-level tokenization strategy with optional filtering for source and target strings.

    Attributes:
        start (str): The token to prepend at the start of the tokenized sequence.
        end (str): The token to append at the end of the tokenized sequence.
        pattern (regex.Pattern): The compiled regex pattern for tokenization.
        src_filters (FilterChain | None): An optional filter chain applied to the source tokens.
        tgt_filters (FilterChain | None): An optional filter chain applied to the target tokens.
    """

    def __init__(
        self,
        start: str,
        end: str,
        pat_str: str,
        src_filters: FilterChain | None = None,
        tgt_filters: FilterChain | None = None,
    ):
        self.start = start
        self.end = end
        self.pattern = regex.compile(pat_str)
        self.src_filters = src_filters
        self.tgt_filters = tgt_filters

    def tokenize_example(self, norm_example: NormalizedExample) -> PreVocabExample | None:
        """
        Tokenizes the source and target strings of a given normalized example.

        Args:
            norm_example (NormalizedExample): A dictionary containing normalized source and target strings.

        Returns:
            PreVocabExample | None: A dictionary extending `NormalizedExample` with the tokenized versions
            of the source and target strings, or `None` if tokenization or filtering fails.
        """

        src_tok = self.tokenize(norm_example["src"], self.src_filters)
        if src_tok is None:
            return None

        tgt_tok = self.tokenize(norm_example["tgt"], self.tgt_filters)
        if tgt_tok is None:
            return None

        example: PreVocabExample = {**norm_example, "src_tok_pre_vocab": src_tok, "tgt_tok_pre_vocab": tgt_tok}
        return example

    def tokenize(self, s: str, filters: FilterChain | None = None) -> list[str] | None:
        """
        Tokenizes a given string using a regex pattern and applies optional filtering.

        Args:
            s (str): The input string to tokenize.
            filters (FilterChain | None): An optional filter chain to validate the tokenized sequence.

        Returns:
            list[str] | None: The tokenized sequence with `start` and `end` tokens included,
            or `None` if filtering fails.
        """

        tok = [tok for tok in (tok.strip() for tok in self.pattern.findall(s)) if tok]
        if filters is None or filters(tok):
            return [self.start, *tok, self.end]
        return None


class WordTokenizer(NMTTokenizer):
    """
    A tokenizer that converts text into integer token sequences and vice versa,
    based on a predefined vocabulary and regex pattern.

    Attributes:
        stoi (dict[str, int]): A mapping from string tokens to their integer representations.
        itos (dict[int, str]): A mapping from integer tokens to their string representations.
        default_ix (int): The default integer token to use for unknown tokens.
        specials (dict[str, int]): A dictionary of special tokens (e.g., "start", "end") and their integer values.
        specials_ix (set[int]): A set of integer values corresponding to the special tokens.
        pattern (regex.Pattern): A compiled regex pattern for tokenizing input text.
    """

    def __init__(
        self, stoi: dict[str, int], itos: dict[int, str], default_ix: int, specials: dict[str, int], pat_str: str
    ):
        self.stoi = stoi
        self.itos = itos
        self.default_ix = default_ix
        self.specials = specials
        self.specials_ix = {ix for ix in specials.values() if ix != default_ix}
        self.pattern = regex.compile(pat_str)

    def encode(self, text: str) -> list[int]:
        tokens = [self.stoi.get(token.strip(), self.default_ix) for token in self.pattern.findall(text) if token]
        return [self.specials["start"], *tokens, self.specials["end"]]

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        """
        Encodes a list of string tokens into a list of integer tokens.

        Args:
            tokens (list[str]): A list of string tokens to be encoded.

        Returns:
            list[int]: A list of integer tokens corresponding to the input string tokens.
        """

        return [self.stoi.get(token, self.default_ix) for token in tokens]

    def decode(self, tokens: list[int], strip_specials: bool = False) -> str:
        return " ".join(self.decode_tokens(tokens, strip_specials))

    def decode_tokens(self, tokens: list[int], strip_specials: bool = False) -> list[str]:
        return [self.itos[token] for token in tokens if not strip_specials or token not in self.specials_ix]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


def build_word_dataset(config: BuildDatasetConfig) -> NMTDataset:
    """
    Builds an NMT (Neural Machine Translation) dataset using word-level tokenization.

    This function takes a dataset configuration and processes source data into a tokenized dataset suitable
    for training, validating, and testing an NMT model. It performs normalization, tokenization,
    vocabulary construction, and data splitting. Tokenizers are created for both the source
    and target languages, and examples are tokenized into integer-indexed and string-tokenized formats.

    Args:
        config (BuildDatasetConfig): The configuration object specifying dataset parameters,
                                     including tokenization settings, normalization rules,
                                     and source/target languages.

    Returns:
        NMTDataset: An object containing the processed dataset, including train, validation,
                    and test splits, along with tokenizers, vocabulary details, and special tokens.

    Raises:
        ValueError: If the provided tokenizer configuration is not compatible with word-level tokenization.
    """

    if type(config.tokenizer) is not WordTokenizerConfig:
        raise ValueError("Invalid configuration for word dataset")
    tok_config: WordTokenizerConfig = config.tokenizer
    specials = tok_config.specials.__dict__

    normalizer = Normalizer(
        config.src_lang, config.tgt_lang, tok_config.lower, tok_config.unicode_normalize, tok_config.patterns
    )
    tokenizer = FilterTokenizer(
        specials["start"],
        specials["end"],
        tok_config.regex,
        src_filters=tok_config.src_filters,
        tgt_filters=tok_config.tgt_filters,
    )

    # normalize examples
    normalized_gen = (ex for ex in map(normalizer.normalize_example, get_examples(config)) if ex is not None)
    # perform word-level tokenization
    examples: list[PreVocabExample] = [ex for ex in map(tokenizer.tokenize_example, normalized_gen) if ex is not None]

    if tok_config.take_first is not None:
        examples = examples[: tok_config.take_first]

    # get indices for training, validation, and testing splits
    train_ix, test_ix = train_test_split(range(len(examples)), train_size=0.8, test_size=0.2, random_state=config.seed)
    val_ix, test_ix = train_test_split(test_ix, train_size=0.5, test_size=0.5, random_state=config.seed)

    # group src and tgt tokens from training set together
    train_src_tokens, train_tgt_tokens = zip(
        *[(examples[ix]["src_tok_pre_vocab"], examples[ix]["tgt_tok_pre_vocab"]) for ix in train_ix], strict=True
    )

    # build src vocab and tokenizer
    src_stoi, src_itos = _build_vocab(train_src_tokens, specials, tok_config.min_freq)
    src_specials = {name: src_stoi[tok] for name, tok in specials.items()}
    src_tokenizer = WordTokenizer(src_stoi, src_itos, src_specials["unk"], src_specials, tok_config.regex)

    # build tgt vocab and tokenizer
    tgt_stoi, tgt_itos = _build_vocab(train_tgt_tokens, specials, tok_config.min_freq)
    tgt_specials = {name: tgt_stoi[tok] for name, tok in specials.items()}
    tgt_tokenizer = WordTokenizer(tgt_stoi, tgt_itos, tgt_specials["unk"], tgt_specials, tok_config.regex)

    # tokenize examples
    tokenized_examples, max_src_len, max_tgt_len = _build_examples(examples, src_tokenizer, tgt_tokenizer)

    # generate splits
    train_data = [tokenized_examples[ix] for ix in train_ix]
    val_data = [tokenized_examples[ix] for ix in val_ix]
    test_data = [tokenized_examples[ix] for ix in test_ix]

    train_set = NMTDatasetSplit(train_data)
    val_set = NMTDatasetSplit(val_data)
    test_set = NMTDatasetSplit(test_data)

    dataset = NMTDataset(
        config.name,
        src_tokenizer,
        tgt_tokenizer,
        train_set,
        val_set,
        test_set,
        config,
        src_specials,
        tgt_specials,
        max_src_len,
        max_tgt_len,
    )

    return dataset


def _build_vocab(
    examples: tuple[list[str]], specials: dict[str, str], min_freq: int
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Builds a vocabulary from tokenized examples, including special tokens and frequency-based filtering.

    Args:
        examples (tuple[list[str]]): A collection of tokenized examples where each example is a list of tokens.
        specials (dict[str, str]): A dictionary of special tokens.
        min_freq (int): The minimum frequency a token must have to be included in the vocabulary.

    Returns:
        tuple[dict[str, int], dict[int, str]]:
            - `stoi` (dict[str, int]): A mapping of tokens to unique integer indices.
            - `itos` (dict[int, str]): A reverse mapping of integer indices to tokens.
    """

    stoi = {tok: i for i, tok in enumerate(specials.values())}
    itos = {i: s for s, i in stoi.items()}

    counts: dict[str, int] = {}
    for example in examples:
        for token in example:
            if token not in stoi:
                counts[token] = counts.get(token, 0) + 1
    sorted_counts = sorted(counts.items(), key=lambda c: c[1], reverse=True)

    for token, cnt in sorted_counts:
        if cnt < min_freq:
            break
        i = len(stoi)
        stoi[token] = i
        itos[i] = token

    return stoi, itos


def _build_examples(
    examples: list[PreVocabExample], src_tokenizer: WordTokenizer, tgt_tokenizer: WordTokenizer
) -> tuple[list[WordExample], int, int]:
    """
    Constructs tokenized examples with integer and string representations for source and target sequences.

    Args:
        examples (list[PreVocabExample]): A list of examples containing pre-tokenized source and target sequences.
        src_tokenizer (WordTokenizer): A tokenizer for encoding and decoding tokens in the source language.
        tgt_tokenizer (WordTokenizer): A tokenizer for encoding and decoding tokens in the target language.

    Returns:
        tuple[list[WordExample], int, int]:
            - A list of `WordExample` dictionaries with tokenized and indexed source and target sequences.
            - The maximum length of the tokenized source sequences.
            - The maximum length of the tokenized target sequences.
    """

    max_src_len, max_tgt_len = 0, 0
    tokenized_examples: list[WordExample] = []
    for ex in examples:
        src_ix = src_tokenizer.encode_tokens(ex["src_tok_pre_vocab"])
        src_tok = src_tokenizer.decode_tokens(src_ix)
        tgt_ix = tgt_tokenizer.encode_tokens(ex["tgt_tok_pre_vocab"])
        tgt_tok = tgt_tokenizer.decode_tokens(tgt_ix)
        tokenized_example: WordExample = dict(**ex, src_ix=src_ix, src_tok=src_tok, tgt_ix=tgt_ix, tgt_tok=tgt_tok)
        tokenized_examples.append(tokenized_example)

        if (src_len := len(src_ix)) > max_src_len:
            max_src_len = src_len
        if (tgt_len := len(tgt_ix)) > max_tgt_len:
            max_tgt_len = tgt_len
    return tokenized_examples, max_src_len, max_tgt_len
