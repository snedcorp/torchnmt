from collections import defaultdict
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import chain, takewhile
from typing import Self, TypeAlias, Union

import regex
from sklearn.model_selection import train_test_split  # type: ignore

from .config import BPETokenizerConfig, BuildDatasetConfig
from .models import (
    Example,
    NMTDataset,
    NMTDatasetSplit,
    NMTTokenizer,
)
from .source import get_examples


Pair: TypeAlias = tuple[int, int]
"""
Represents a consecutive pair of tokens within a sequence.
The integer values represent the tokens' unique identifiers / indices within the vocabulary.
"""


@dataclass
class TrainToken:
    """
    Represents a token in a sequence with connections to its neighbors,
    forming a doubly linked list of tokens.

    Attributes:
        id (int): The unique identifier for the token within the training vocabulary.
        prev (TrainToken | None): A reference to the previous token in the sequence, or `None` if it is the first token.
        next (TrainToken | None): A reference to the next token in the sequence, or `None` if it is the last token.
    """

    id: int
    prev: Self | None = None
    next: Self | None = None


@dataclass
class PairStatus:
    """
    Represents the current status for a unique Pair of tokens within the BPE training process.

    Attributes:
        count (int): The current total number of occurrences of the Pair across every TrainWord derived
        from the training text.
        pos (set[int]): A set of indices corresponding to every TrainWord derived from the training text
        that currently contains the Pair.
    """

    count: int
    pos: set[int]


class TrainWord:
    """
    Contains the tokens currently comprising a unique "word" (or chunk) derived from the training text using a
    splitting regex pattern.

    The tokens are stored as a doubly linked list of TrainToken nodes, to facilitate the merging of consecutive
    tokens as part of the BPE training process.

    Attributes:
        start_token (TrainToken): The head of the TrainToken linked list.
        count (int): The total number of occurrences of this word/chunk within the training text.
    """

    def __init__(self, start_token: TrainToken, count: int):
        self.start_token = start_token
        self.count = count

    @classmethod
    def build(cls, text: str, count: int):
        """
        Static factory method to build a TrainWord from a given string.

        Steps:
            1) Converts the given string into a list of bytes corresponding to its UTF-8 encoding.
            2) Converts those bytes into integers, to be used as the unique identifiers within the vocabulary
            for the tokens they represent.
                - This is possible b/c the first 256 entries in the vocabulary correspond directly to their
                byte equivalents.
            3) Then, constructs the doubly linked list of TrainTokens representing those single-byte tokens.

        Args:
            text (str): A unique word/chunk within the training text.
            count (int): The total number of occurrences of this word/chunk within the training text.

        Returns:
            (TrainWord): The built TrainWord instance.
        """

        if not text:
            raise ValueError("Empty train word")

        ids = list(map(int, text.encode("utf-8")))

        start_token = TrainToken(id=ids[0])
        if len(ids) > 1:
            prev_token = start_token
            for id_ in ids[1:]:
                token = TrainToken(id_, prev=prev_token)
                prev_token.next = token
                prev_token = token
        return cls(start_token=start_token, count=count)

    def merge(self, target_pair: Pair, new_token_id: int) -> tuple[dict[Pair, int], dict[Pair, int], dict[Pair, int]]:
        """
        Traverses the TrainToken linked list and replaces any occurrences of the provided Pair with a new TrainToken
        representing the new, merged token.

        To enable the calling BPETrainer to update its status as a result of merging this word, this method returns
        three Pair -> int dictionaries detailing:
            - Pairs that have been added, and their count deltas (`positive`)
            - Pairs that have been removed, and their count deltas (`negative`)
            - All the pairs in the merged word, and their counts (`pairs`)

        It's important to keep in mind that a pair being added to the `positive` dict does not mean that it didn't exist
        on the word previously, just that there's now there's new occurrences of it. Similarly, a pair being added to the
        `negative` dict does not mean that it no longer exists on the merged word at all, just that there's now fewer
        occurrences of it.

        Furthermore, because each TrainWord represents a unique word/chunk within the training text, each time
        a pair is added or removed, the count delta is actually multiplied by the number of occurrences of that word
        within the training text. So if the word is 'hahaha' and it appears 100 times in the text, and we then merge 'ha',
        the negative count delta for the pair 'ah' is actually 200 (2 in word * 100 occurrences of the word in the text).

        Also, note that when traversing the TrainToken linked list, though pairs are checked with the current token and
        the next token, a pair isn't added to the final `pairs` map until the current token is the second in the pair.
        This makes intuitive sense - in the example above, when merging the first 'ha' into a new token, we cannot
        add a pair for it until after we keep traversing, because the next token might itself get merged.

        Args:
            target_pair: The token Pair to be merged.
            new_token_id: The unique identifier for the new token replacing the given Pair.

        Returns:
            dict[Pair, int]: A map of all the new token pairs that have been created as a result of the merge, to their
            corresponding count deltas.
            dict[Pair, int]: A map of all the token pairs that have been removed as a result of the merge, to their
            corresponding count deltas.
            dict[Pair, int]: A map of all the token pairs that are present in the merged TrainWord, to their
            corresponding counts.
        """

        positive: defaultdict[Pair, int] = defaultdict(int)
        negative: defaultdict[Pair, int] = defaultdict(int)
        pairs: defaultdict[Pair, int] = defaultdict(int)

        def add_pair(token1: TrainToken, token2: TrainToken):
            new_pair = (token1.id, token2.id)
            pairs[new_pair] += self.count
            if new_token_id in new_pair:
                positive[new_pair] += self.count

        token = self.start_token
        while token.next is not None:
            if token.id == target_pair[0] and token.next.id == target_pair[1]:
                first, second = token, token.next
                new_token = TrainToken(id=new_token_id, prev=first.prev, next=second.next)

                if new_token.prev is not None:
                    new_token.prev.next = new_token
                    if new_token.prev.id != new_token_id:
                        negative[(new_token.prev.id, first.id)] += self.count
                    add_pair(new_token.prev, new_token)
                else:
                    self.start_token = new_token

                if new_token.next is None:
                    break
                new_token.next.prev = new_token

                negative[(second.id, new_token.next.id)] += self.count
                token = new_token
            else:
                if token.prev is not None:
                    add_pair(token.prev, token)

            next_token = token.next
            assert next_token  # placate mypy
            if next_token.next is None:
                add_pair(token, next_token)
            token = next_token

        return positive, negative, pairs


@dataclass
class TrainMergeCandidate:
    """
    Represents a potential merge candidate in the training process, used to populate the priority queue (implemented as
    a min-heap).

    Attributes:
        pair (Pair): The Pair of tokens to be merged.
        count (int): The number of occurrences of this Pair, at the time of instantiation.
    """

    pair: Pair
    count: int

    def __lt__(self, nxt: Self):
        """
        Ensures that the min-heap will return the highest-count candidates first.

        Args:
            nxt: Another TrainMergeCandidate being compared.
        """
        return self.count > nxt.count


def _get_word_counts(words_iter: Iterator[str]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """
    Computes word frequencies from an iterator of words.

    Args:
        words_iter (Iterator[str]): An iterator that yields words to be counted.

    Returns:
        tuple[str, ...]: A tuple of unique words.
        tuple[int, ...]: A corresponding tuple of their frequencies.
    """

    word_counts: defaultdict[str, int] = defaultdict(int)
    for word in words_iter:
        word_counts[word] += 1
    words, counts = zip(*word_counts.items(), strict=True)
    return words, counts


class BPETrainer:
    """
    This class implements the byte-level Byte Pair Encoding (BPE) algorithm to construct a vocabulary and a list of merges
    from the given input text, to be used by the BPETokenizer to encode and decode text to and from its corresponding
    integer tokens.

    It applies the BPE algorithm by iteratively identifying and merging the most frequent adjacent byte pairs in the text.
    By processing text at the byte level, it enables the creation of subword representations that are highly efficient for
    tokenization.

    Note that, prior to training, a regex pattern is used to split the text up into different "words", or chunks, to
    ensure that no merges occur between unrelated bytes.

    Additionally, because this is the "byte-level" variant of BPE, this class doesn't care about Unicode characters and
    instead operates directly on the byte-level representation of the text, thereby removing the potential for any
    out-of-vocabulary issues when tokenizing. As a result, the first 256 tokens in the resulting vocabulary correspond
    directly to their byte equivalents.

    Naive implementation approach:
        - At each step, iterate across all words to find the most common pair, and then iterate across them all again to
        merge the pair where needed.

    Improved, vastly more efficient approach:
        - Each unique word (chunk) is transformed into a TrainWord object containing:
            - A linked list of TrainToken objects, each representing a token.
            - Its frequency within the text.
        - A priority queue (implemented as a min-heap) is used to sort all potential pairs and always return the pair with
        the highest count.
        - The `pairs` dictionary is also used to store the current state of every pair in the training process, including:
            - Its current count.
            - The positions (indices) of the words it's currently found within.
        - Initially, all pairs found within the original text are inserted into the heap.
        - Then, until the heap is exhausted or the desired vocabulary size has been reached:
            - The pair with the highest count is popped from the heap
            - For every TrainWord that contains that pair (obtained from the `pairs` dictionary):
                - The TrainWord is merged, and returns information about the merge's effects:
                    - Pairs that were added, pairs that were removed, all current pairs
                - The `pairs` dictionary is then updated based on this information, to remain in sync
            - Once every TrainWord has been processed, every new, unique, pair that was created as a result of the merge
            is pushed onto the heap.
            - Repeat
        - Finally, add the desired special tokens to the vocabulary.
        - A caveat:
            - To avoid having to re-populate the heap after each merge, it is accepted that some of the pairs in the heap
            will inevitably have a now-inaccurate count after a merge occurs. To guard against this issue, when a pair
            is popped from the heap, it is first checked to see its count matches the value in the `pairs` dictionary -
            if not, it should either re-pushed with the correct count, or ignored entirely.

    Why is this approach so much faster?
        - Because by always possessing an accurate view of a pair's current state, we can be sure we're only visiting
        the words that actually need to be processed for a given pair.

    Attributes:
        vocab_size (int): The maximum size of the vocabulary to be generated during training.
        pattern (regex.Pattern): A compiled regular expression pattern used for splitting the input text into "words"
        (or chunks).
        specials (dict[str, str]): A dictionary of special tokens, mapping token names (e.g., "pad", "unk") to their
        string representations.
        verbose (bool): Whether to log token merges as they occur.
        pairs (dict[Pair, PairStatus]): A mapping of each unique Pair to its current status, which includes the total
        count of all its occurrences, as well as the positions of the TrainWord objects (within the `words` list) that it
        can be found within.
        words (list[TrainWord]): A list of TrainWord objects representing the training data, each corresponding to a
        unique word or chunk within the original text.
    """

    def __init__(
        self,
        text: Union[str, Sequence[str]],
        vocab_size: int,
        pat_str: str,
        specials: dict[str, str],
        verbose: bool = False,
    ):
        self.vocab_size = vocab_size
        self.pattern = regex.compile(pat_str)
        self.specials = specials
        self.verbose = verbose
        self.pairs: dict[Pair, PairStatus] = {}
        self.words: list[TrainWord] = self._build_words(text)
        self._populate_pairs(self.words)

    def _build_words(self, text: Union[str, Sequence[str]]) -> list[TrainWord]:
        """
        Builds a list of TrainWord objects from the given input text or sequence of texts.

        Args:
            text (Union[str, Sequence[str]]): The input text or sequence of texts from which to extract words.

        Returns:
            list[TrainWord]: The TrainWord objects derived from the input text.
        """

        words_str, counts = _get_word_counts(self._get_words(text))
        return [TrainWord.build(word, counts[i]) for i, word in enumerate(words_str)]

    def _get_words(self, text: Union[str, Sequence[str]]) -> Iterator[str]:
        """
        Extracts words/chunks from the given text or sequence of texts using a regex pattern.

        Args:
            text (Union[str, Sequence[str]]):
                The input text or sequence of texts from which to extract words.
                - If a single string is provided, words are extracted directly from it.
                - If a sequence of strings is provided, words are extracted from each string in the sequence.

        Returns:
            Iterator[str]: An iterator over the words extracted from the input text(s).
        """

        if isinstance(text, str):
            yield from self.pattern.finditer(text)
        for example in text:
            yield from self.pattern.findall(example)

    def _upsert_pair(self, pair: Pair, count: int, pos: int):
        """
        Updates the count and positions for an existing Pair in the `pairs` dictionary, or instead inserts a new Pair
        with the provided count and position.

        Args:
            pair (Pair): The key representing the pair to be added or updated.
            count (int): The number to increment the pair's count by.
            pos (int): The position to add to the pair's set of positions.
        """

        if pair in self.pairs:
            pair_status = self.pairs[pair]
            pair_status.count += count
            pair_status.pos.add(pos)
        else:
            self.pairs[pair] = PairStatus(count, {pos})

    def _populate_pairs(self, words: Sequence[TrainWord]):
        """
        Constructs pairs of adjacent token IDs from a sequence of TrainWord objects and sets/updates their counts and
        positions as needed.

        Args:
            words (Sequence[TrainWord]): A sequence of TrainWord objects representing the words to process.
        """

        for i, word in enumerate(words):
            token = word.start_token
            while token.next is not None:
                pair = (token.id, token.next.id)
                self._upsert_pair(pair, word.count, i)
                token = token.next

    def train(self) -> tuple[dict[Pair, int], dict[int, bytes], dict[str, int]]:
        """
        Uses a priority queue and the `pairs` dictionary to iteratively build up a vocabulary to the desired size, in
        the manner described above.

        Returns:
            dict[Pair, int]: The mapping of merged token pairs to their corresponding token indices.
            dict[int, bytes]: The mapping of token indices to their corresponding byte sequences.
            dict[str, int]: The mapping of special tokens to their corresponding token indices.
        """

        queue: list[TrainMergeCandidate] = []
        # populate heap with every pair
        for pair, status in self.pairs.items():
            heappush(queue, TrainMergeCandidate(pair, status.count))

        merges: dict[Pair, int] = {}
        # populate vocab with initial byte mappings
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        while len(queue) > 0 and len(vocab) < self.vocab_size:
            top_candidate = heappop(queue)

            curr_status = self.pairs[top_candidate.pair]
            # if candidate's count is out of date, move to next candidate
            if top_candidate.count != curr_status.count:
                # if candidate hasn't been merged yet, re-push with correct count
                if curr_status.count > 0:
                    top_candidate.count = curr_status.count
                    heappush(queue, top_candidate)
                continue

            new_token_id = 256 + len(merges)

            positive_pairs = set()
            # merge pair in every word that contains it
            for pos in curr_status.pos:
                positive, negative, new_pairs = self.words[pos].merge(top_candidate.pair, new_token_id)

                # update or insert status for new pairs
                for pair, count in positive.items():
                    self._upsert_pair(pair, count, pos)
                    positive_pairs.add(pair)

                # update status for removed pairs
                for pair, count in negative.items():
                    if pair == top_candidate.pair:
                        continue
                    self.pairs[pair].count -= count
                    # if pair no longer found on word, can remove its position from status
                    if new_pairs[pair] == 0:
                        self.pairs[pair].pos.remove(pos)

            # update pair status to account for merge
            curr_status.count = 0
            curr_status.pos = set()

            # mint new token
            merges[top_candidate.pair] = new_token_id
            vocab[new_token_id] = vocab[top_candidate.pair[0]] + vocab[top_candidate.pair[1]]

            # push all new pairs onto the heap
            for pair in positive_pairs:
                heappush(queue, TrainMergeCandidate(pair, self.pairs[pair].count))

            if self.verbose:
                print(
                    f"merged {vocab[top_candidate.pair[0]]!r} ({top_candidate.pair[0]}) , {vocab[top_candidate.pair[1]]!r} ({top_candidate.pair[1]}) -> {vocab[new_token_id]!r} ({new_token_id})"
                )

        # add special tokens to vocab
        specials_ix = {}
        for name, val in self.specials.items():
            new_token_id = len(vocab)
            vocab[new_token_id] = val.encode("utf-8")
            specials_ix[name] = new_token_id

        return merges, vocab, specials_ix


@dataclass
class Token:
    """
    Represents a token in a linked list structure used for Byte Pair Encoding.

    Attributes:
        id (int): The unique identifier for the token.
        prev (int): The index of the previous token in the sequence. Defaults to -1.
        next (int): The index of the next token in the sequence. Defaults to -1.
        active (bool): A flag indicating whether the token is currently active in the sequence (default is True).
    """

    id: int
    prev: int = -1
    next: int = -1
    active: bool = True


@dataclass
class MergeCandidate:
    """
    Represents a potential merge candidate in the encoding process, used to populate the priority queue (implemented as
    a min-heap).

    Attributes:
        pos (int): The leading position (index) of the pair within the `tokens` list.
        new_id (int): The identifier for the new token that results from this merge.
    """

    pos: int
    new_id: int

    def __lt__(self, nxt: Self):
        """
        Ensures that the min-heap will return the merge candidates whose new token indices are the smallest.
        If the new tokens are the same, then return whichever comes first in the sequence.

        Args:
            nxt: Another MergeCandidate being compared.
        """

        if self.new_id != nxt.new_id:
            return self.new_id < nxt.new_id
        return self.pos < nxt.pos


class BPETokenizer(NMTTokenizer):
    """
    A tokenizer that converts text into integer token sequences and vice versa, leveraging a vocabulary derived from the
    Byte Pair Encoding (BPE) process, which iteratively merges the most frequent adjacent tokens to create subword
    representations.

    Attributes:
        merges (dict[Pair, int]): A mapping of merged token pairs to their corresponding token indices.
        vocab (dict[int, bytes]): A mapping of token indices to their corresponding byte sequences.
        pattern (regex.Pattern): A compiled regex pattern for pre-tokenizing input text.
        specials (dict[str, int]): A dictionary of special tokens (e.g., "start", "end") and their corresponding token
        indices.
        specials_ix (set[int]): A set of integer token indices corresponding to the special tokens.
    """

    def __init__(self, merges: dict[Pair, int], vocab: dict[int, bytes], pat_str: str, specials: dict[str, int]):
        self.merges = merges
        self.vocab = vocab
        self.pattern = regex.compile(pat_str)
        self.specials = specials
        self.specials_ix = {ix for ix in specials.values()}

    def encode(self, text: str) -> list[int]:
        tokens = chain.from_iterable(self.encode_chunk(chunk) for chunk in self.pattern.findall(text))
        return [self.specials["start"], *tokens, self.specials["end"]]

    def decode(self, tokens: list[int], strip_specials: bool = False) -> str:
        text_bytes = b"".join(
            [self.vocab[token] for token in tokens if not strip_specials or token not in self.specials_ix]
        )
        return text_bytes.decode("utf-8", errors="replace")

    def decode_tokens(self, tokens: list[int], strip_specials: bool = False) -> list[str]:
        return [
            self.vocab[token].decode("utf-8", errors="replace")
            for token in tokens
            if not strip_specials or token not in self.specials_ix
        ]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode_chunk(self, text: str) -> Generator[int, None, None]:
        """
        Encodes a text chunk into a sequence of integer tokens based on Byte Pair Encoding (BPE) rules.

        This function converts a given string into a sequence of integers representing byte-level tokens,
        and iteratively applies relevant BPE merge operations defined in a `merges` dictionary.

        Args:
            text (str): The input text chunk to encode.

        Yields:
            int: The integer representation of each token in the encoded sequence, in order.

        Raises:
            ValueError: If the input `text` is an empty string.

        Details:
            - The function starts by encoding the input text into its UTF-8 byte representation,
              converting each byte into an integer ID.
            - If the text consists of a single byte, the corresponding integer is yielded immediately.
            - For longer sequences, it constructs a list of Token objects, each representing a byte
              with pointers (`prev` and `next`) to maintain a linked list structure.
            - Using a priority queue (`queue`), it executes merge operations by combining adjacent tokens
              according to the highest priority (most frequently found in source text) relevant pairs defined in
              `merges`.
                - When a potential merge is found for two adjacent tokens, the MergeCandidate added to the priority queue
                contains both the leading position in the pair, and the identifier for the resulting merged token.
            - The function ensures active tokens are updated, merged, and properly linked, while inactive tokens are
            skipped.

        Merging:
            - Once a merge pair is determined to be valid (more on that below), a new Token is created to represent the
            merged token, with the `prev` pointer set to the first token's `prev` pointer, and the `next` pointer set to
            the second token's `next` pointer.
            - That new token is then placed within the `token` list at the first token's position, and the second token
            is updated to an inactive status.
            - Finally, if a potential merge candidate is found for the new token combined with its neighbor to the
            left or right, those new candidates (if any) are added to the priority queue.

        Also, note that because the contents of the `tokens` list change as merges occur, a merge candidate procured
        from the priority queue might not be valid anymore. For a merge candidate from the queue to be considered valid:
            - The token at the candidate's position must:
                - Not have been set to inactive.
                - Still have a valid `next` pointer.
            - If those conditions are satisfied, then the pair comprising the token at the candidate's position and its
            `next` neighbor must still combine for a merge that:
                - Is valid (i.e. found within `merges`), and
                - Results in the same exact token id as stored in the candidate.
        """

        if not text:
            raise ValueError("Empty word")

        ids = list(map(int, text.encode("utf-8")))

        if len(ids) == 1:
            yield ids[0]
            return

        tokens: list[Token] = []
        queue: list[MergeCandidate] = []

        # add all initial candidates to queue
        for i, id_ in enumerate(ids):
            token = Token(id_)
            if i > 0:
                token.prev = i - 1
                tokens[i - 1].next = i
                pair = (tokens[i - 1].id, token.id)
                if pair in self.merges:
                    new_id = self.merges[pair]
                    heappush(queue, MergeCandidate(pos=i - 1, new_id=new_id))
            tokens.append(token)

        while len(queue) > 0:
            top = heappop(queue)

            token = tokens[top.pos]
            # check if active pair still exists at the location
            if not token.active or token.next < 0:
                continue

            next_token = tokens[token.next]

            pair = (token.id, next_token.id)
            # check if same pair still exists at the location
            if pair not in self.merges or self.merges[pair] != top.new_id:
                continue

            # overwrite merged token with first token
            new_token = Token(top.new_id, prev=token.prev, next=next_token.next)
            tokens[top.pos] = new_token

            token.active = False  # not technically necessary, no longer in list
            next_token.active = False

            if new_token.prev > -1:
                prev_token = tokens[new_token.prev]
                # add (prev, new_token) pair if valid
                if new_id := self.merges.get((prev_token.id, new_token.id)):  # type: ignore
                    heappush(queue, MergeCandidate(pos=new_token.prev, new_id=new_id))

            if new_token.next > -1:
                next_token = tokens[new_token.next]
                next_token.prev = top.pos
                # add (new_token, next) pair if valid
                if new_id := self.merges.get((new_token.id, next_token.id)):  # type: ignore
                    heappush(queue, MergeCandidate(pos=top.pos, new_id=new_id))

        yield from (token.id for token in tokens if token.active)


def build_bpe_dataset(config: BuildDatasetConfig) -> NMTDataset:
    """
    Builds an NMT (Neural Machine Translation) dataset using byte-level Byte Pair Encoding (BPE) tokenization.

    This function takes a dataset configuration and processes source data into a tokenized dataset suitable
    for training, validating, and testing an NMT model. It performs vocabulary construction, tokenization, and data
    splitting. Tokenizers are created for both the source and target languages, and examples are tokenized into
    integer-indexed and string-tokenized formats.

    Args:
        config (BuildDatasetConfig): The configuration object specifying dataset parameters,
                                     including tokenization settings, data sources,
                                     and source/target languages.

    Returns:
        NMTDataset: An object containing the processed dataset, including train, validation,
                    and test splits, along with tokenizers, vocabulary details, and special tokens.

    Raises:
        ValueError: If the provided tokenizer configuration is not compatible with BPE tokenization.
    """

    if type(config.tokenizer) is not BPETokenizerConfig:
        raise ValueError("Invalid configuration for BPE dataset")
    tok_config = config.tokenizer
    specials = tok_config.specials.__dict__

    # retrieve examples
    examples_gen = get_examples(config)
    # truncate if desired
    if (take_first := tok_config.take_first) is not None:  # assign to local var for mypy
        examples_gen = (ex for _, ex in takewhile(lambda e: e[0] < take_first, enumerate(examples_gen)))  # type: ignore

    # group src and tgt tokens together
    src, tgt = zip(*[ex for ex in examples_gen], strict=True)

    # build src vocab and tokenizer
    src_trainer = BPETrainer(src, tok_config.src_vocab_size, tok_config.regex, specials, tok_config.verbose)
    src_merges, src_vocab, src_specials = src_trainer.train()
    src_tokenizer = BPETokenizer(src_merges, src_vocab, tok_config.regex, src_specials)

    # build tgt vocab and tokenizer
    tgt_trainer = BPETrainer(tgt, tok_config.tgt_vocab_size, tok_config.regex, specials, tok_config.verbose)
    tgt_merges, tgt_vocab, tgt_specials = tgt_trainer.train()
    tgt_tokenizer = BPETokenizer(tgt_merges, tgt_vocab, tok_config.regex, tgt_specials)

    # tokenize examples
    examples, max_src_len, max_tgt_len = _build_examples(src, tgt, src_tokenizer, tgt_tokenizer)

    # get indices for training, validation, and testing splits
    train_ix, test_ix = train_test_split(range(len(src)), train_size=0.8, test_size=0.2, random_state=config.seed)
    val_ix, test_ix = train_test_split(test_ix, train_size=0.5, test_size=0.5, random_state=config.seed)

    # generate splits
    train_data = [examples[ix] for ix in train_ix]
    val_data = [examples[ix] for ix in val_ix]
    test_data = [examples[ix] for ix in test_ix]

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


def _build_examples(
    src: Sequence[str], tgt: Sequence[str], src_tokenizer: BPETokenizer, tgt_tokenizer: BPETokenizer
) -> tuple[list[Example], int, int]:
    """
    Constructs tokenized examples with integer and string representations for source and target sequences.

    Args:
        src (Sequence[str]): A list of source sequences.
        tgt (Sequence[str]): A list of target sequences.
        src_tokenizer (BPETokenizer): A tokenizer for encoding and decoding tokens in the source language.
        tgt_tokenizer (BPETokenizer): A tokenizer for encoding and decoding tokens in the target language.

    Returns:
        tuple[list[Example], int, int]:
            - A list of `Example` dictionaries with tokenized and indexed source and target sequences.
            - The maximum length of the tokenized source sequences.
            - The maximum length of the tokenized target sequences.
    """

    examples: list[Example] = []
    max_src_len, max_tgt_len = 0, 0
    for i in range(len(src)):
        src_ix = src_tokenizer.encode(src[i])
        src_tok = src_tokenizer.decode_tokens(src_ix)
        tgt_ix = tgt_tokenizer.encode(tgt[i])
        tgt_tok = tgt_tokenizer.decode_tokens(tgt_ix)
        ex: Example = dict(src=src[i], tgt=tgt[i], src_ix=src_ix, src_tok=src_tok, tgt_ix=tgt_ix, tgt_tok=tgt_tok)
        examples.append(ex)

        if (src_len := len(ex["src_ix"])) > max_src_len:
            max_src_len = src_len
        if (tgt_len := len(ex["tgt_ix"])) > max_tgt_len:
            max_tgt_len = tgt_len
    return examples, max_src_len, max_tgt_len
