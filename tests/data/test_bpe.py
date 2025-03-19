import pytest

from torchnmt.data.bpe import BPETokenizer, TrainWord


@pytest.fixture
def gpt4_regex():
    return (
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
    )


def test_TrainWord_build_one_byte():
    word = "a"
    train_word = TrainWord.build(word, 1)
    assert train_word.start_token
    assert train_word.count == 1
    start_token = train_word.start_token
    assert start_token.id == 97
    assert start_token.prev is None
    assert start_token.next is None


def test_TrainWord_build():
    word = "bumi"
    train_word = TrainWord.build(word, 2)
    assert train_word.start_token
    assert train_word.count == 2

    t0 = train_word.start_token
    assert t0.id == 98

    t1 = t0.next
    assert t1.id == 117

    t2 = t1.next
    assert t2.id == 109

    t3 = t2.next
    assert t3.id == 105

    assert t3.next is None
    assert t3.prev == t2
    assert t2.prev == t1
    assert t1.prev == t0
    assert t0.prev is None


def test_TrainWord_build_unicode():
    word = "bumí"
    train_word = TrainWord.build(word, 3)
    assert train_word.start_token
    assert train_word.count == 3

    t0 = train_word.start_token
    assert t0.id == 98

    t1 = t0.next
    assert t1.id == 117

    t2 = t1.next
    assert t2.id == 109

    t3 = t2.next
    assert t3.id == 195

    t4 = t3.next
    assert t4.id == 173

    assert t4.next is None
    assert t4.prev == t3
    assert t3.prev == t2
    assert t2.prev == t1
    assert t1.prev == t0
    assert t0.prev is None


def test_TrainWord_merge_single_token():
    word = "b"
    train_word = TrainWord.build(word, 3)
    pos, neg, pairs = train_word.merge((117, 109), 256)
    assert len(pos) == 0
    assert len(neg) == 0
    assert len(pairs) == 0


def test_TrainWord_merge_once_middle():
    word = "bumi"
    train_word = TrainWord.build(word, 3)
    positive, negative, pairs = train_word.merge((117, 109), 256)

    t0 = train_word.start_token
    assert t0.id == ord("b")

    t1 = t0.next
    assert t1.id == 256

    t2 = t1.next
    assert t2.id == ord("i")

    assert t2.next is None
    assert t2.prev == t1
    assert t1.prev == t0
    assert t0.prev is None

    assert len(positive) == 2
    assert positive[(ord("b"), 256)] == 3
    assert positive[(256, ord("i"))] == 3

    assert len(negative) == 2
    assert negative[(ord("b"), ord("u"))]
    assert negative[(ord("m"), ord("i"))]

    assert len(pairs) == 2
    assert pairs[(ord("b"), 256)] == 3
    assert pairs[(256, ord("i"))] == 3


def test_TrainWord_merge_once_middle_untouched_pairs():
    word = "bummi"
    count = 3
    new_id = 256
    train_word = TrainWord.build(word, count)
    positive, negative, pairs = train_word.merge((ord("u"), ord("m")), new_id)

    t0 = train_word.start_token
    assert t0.id == ord("b")

    t1 = t0.next
    assert t1.id == new_id

    t2 = t1.next
    assert t2.id == ord("m")

    t3 = t2.next
    assert t3.id == ord("i")

    assert t3.next is None
    assert t3.prev == t2
    assert t2.prev == t1
    assert t1.prev == t0
    assert t0.prev is None

    assert len(positive) == 2
    assert positive[(ord("b"), new_id)] == count
    assert positive[(new_id, ord("m"))] == count

    assert len(negative) == 2
    assert negative[(ord("b"), ord("u"))] == count
    assert negative[(ord("m"), ord("m"))] == count

    assert len(pairs) == 3
    assert pairs[(ord("b"), new_id)] == count
    assert pairs[(new_id, ord("m"))] == count
    assert pairs[(ord("m"), ord("i"))] == count


# bumbumb


def test_TrainWord_merge_twice_middle():
    word = "bumpumi"
    count = 3
    new_id = 256
    train_word = TrainWord.build(word, count)
    positive, negative, pairs = train_word.merge((ord("u"), ord("m")), new_id)

    t0 = train_word.start_token
    assert t0.id == ord("b")

    t1 = t0.next
    assert t1.id == new_id

    t2 = t1.next
    assert t2.id == ord("p")

    t3 = t2.next
    assert t3.id == new_id

    t4 = t3.next
    assert t4.id == ord("i")

    assert t4.next is None
    assert t4.prev == t3
    assert t3.prev == t2
    assert t2.prev == t1
    assert t1.prev == t0
    assert t0.prev is None

    assert len(positive) == 4
    assert positive[(ord("b"), new_id)] == count
    assert positive[(new_id, ord("p"))] == count
    assert positive[(ord("p"), new_id)] == count
    assert positive[(new_id, ord("i"))] == count

    assert len(negative) == 4
    assert negative[(ord("b"), ord("u"))] == count
    assert negative[(ord("m"), ord("p"))] == count
    assert negative[(ord("p"), ord("u"))] == count
    assert negative[(ord("m"), ord("i"))] == count

    assert len(pairs) == 4
    assert pairs[(ord("b"), new_id)] == count
    assert pairs[(new_id, ord("p"))] == count
    assert pairs[(ord("p"), new_id)] == count
    assert pairs[(new_id, ord("i"))] == count


def test_TrainWord_merge_twice_middle_repeats():
    word = "bumbumb"
    count = 3
    new_id = 256
    train_word = TrainWord.build(word, count)
    positive, negative, pairs = train_word.merge((ord("u"), ord("m")), new_id)

    t0 = train_word.start_token
    assert t0.id == ord("b")

    t1 = t0.next
    assert t1.id == new_id

    t2 = t1.next
    assert t2.id == ord("b")

    t3 = t2.next
    assert t3.id == new_id

    t4 = t3.next
    assert t4.id == ord("b")

    assert t4.next is None
    assert t4.prev == t3
    assert t3.prev == t2
    assert t2.prev == t1
    assert t1.prev == t0
    assert t0.prev is None

    assert len(positive) == 2
    assert positive[(ord("b"), new_id)] == count * 2
    assert positive[(new_id, ord("b"))] == count * 2

    assert len(negative) == 2
    assert negative[(ord("b"), ord("u"))] == count * 2
    assert negative[(ord("m"), ord("b"))] == count * 2

    assert len(pairs) == 2
    assert pairs[(ord("b"), new_id)] == count * 2
    assert pairs[(new_id, ord("b"))] == count * 2


def test_TrainWord_merge_first():
    word = "bumi"
    count = 3
    new_id = 256
    train_word = TrainWord.build(word, count)
    positive, negative, pairs = train_word.merge((ord("b"), ord("u")), new_id)

    t0 = train_word.start_token
    assert t0.id == new_id

    t1 = t0.next
    assert t1.id == ord("m")

    t2 = t1.next
    assert t2.id == ord("i")

    assert t2.next is None
    assert t2.prev == t1
    assert t1.prev == t0
    assert t0.prev is None

    assert len(positive) == 1
    assert positive[(new_id, ord("m"))] == count

    assert len(negative) == 1
    assert negative[(ord("u"), ord("m"))] == count

    assert len(pairs) == 2
    assert pairs[(new_id, ord("m"))] == count
    assert pairs[(ord("m"), ord("i"))] == count


def test_TrainWord_merge_last():
    word = "bumi"
    count = 3
    new_id = 256
    train_word = TrainWord.build(word, count)
    positive, negative, pairs = train_word.merge((ord("m"), ord("i")), new_id)

    t0 = train_word.start_token
    assert t0.id == ord("b")

    t1 = t0.next
    assert t1.id == ord("u")

    t2 = t1.next
    assert t2.id == new_id

    assert t2.next is None
    assert t2.prev == t1
    assert t1.prev == t0
    assert t0.prev is None

    assert len(positive) == 1
    assert positive[(ord("u"), new_id)] == count

    assert len(negative) == 1
    assert negative[(ord("u"), ord("m"))] == count

    assert len(pairs) == 2
    assert pairs[(ord("b"), ord("u"))] == count
    assert pairs[(ord("u"), new_id)] == count


def test_TrainWord_merge_complete():
    word = "um"
    count = 3
    new_id = 256
    train_word = TrainWord.build(word, count)
    positive, negative, pairs = train_word.merge((ord("u"), ord("m")), new_id)

    t0 = train_word.start_token
    assert t0.id == new_id

    assert t0.next is None
    assert t0.prev is None

    assert len(positive) == 0
    assert len(negative) == 0
    assert len(pairs) == 0


def test_TrainWord_merge_repeated():
    word = "umumum"
    count = 3
    new_id = 256
    train_word = TrainWord.build(word, count)
    positive, negative, pairs = train_word.merge((ord("u"), ord("m")), new_id)

    t0 = train_word.start_token
    assert t0.id == new_id

    t1 = t0.next
    assert t1.id == new_id

    t2 = t1.next
    assert t2.id == new_id

    assert t2.next is None
    assert t2.prev == t1
    assert t1.prev == t0
    assert t0.prev is None

    assert len(positive) == 1
    assert positive[(new_id, new_id)] == count * 2

    assert len(negative) == 1
    assert negative[(ord("m"), ord("u"))] == count * 2

    assert len(pairs) == 1
    assert pairs[(new_id, new_id)] == count * 2


def test_encode_chunk_merge_then_no_merge(gpt4_regex):
    text = "Bumi"
    merges = {(ord("u"), ord("m")): 256, (ord("B"), ord("u")): 257}
    tokenizer = BPETokenizer(merges, None, gpt4_regex, {})
    tokens = [*tokenizer.encode_chunk(text)]
    assert len(tokens) == 3
    assert tokens[0] == ord("B")
    assert tokens[1] == 256
    assert tokens[2] == ord("i")


def test_encode_chunk_merge_then_inactive(gpt4_regex):
    text = "Bumi"
    merges = {(ord("u"), ord("m")): 256, (ord("m"), ord("i")): 257}
    tokenizer = BPETokenizer(merges, None, gpt4_regex, {})
    tokens = [*tokenizer.encode_chunk(text)]
    assert len(tokens) == 3
    assert tokens[0] == ord("B")
    assert tokens[1] == 256
    assert tokens[2] == ord("i")


def test_encode_chunk_merge_then_cand_before(gpt4_regex):
    text = "Bumi"
    merges = {(ord("u"), ord("m")): 256, (ord("B"), 256): 257}
    tokenizer = BPETokenizer(merges, None, gpt4_regex, {})
    tokens = [*tokenizer.encode_chunk(text)]
    assert len(tokens) == 2
    assert tokens[0] == 257
    assert tokens[1] == ord("i")


def test_encode_chunk_merge_then_cand_after(gpt4_regex):
    text = "Bumi"
    merges = {(ord("u"), ord("m")): 256, (256, ord("i")): 257}
    tokenizer = BPETokenizer(merges, None, gpt4_regex, {})
    tokens = [*tokenizer.encode_chunk(text)]
    assert len(tokens) == 2
    assert tokens[0] == ord("B")
    assert tokens[1] == 257


def test_encode_chunk_merge_then_wrong_pair_merge_before(gpt4_regex):
    text = "Bumi"
    merges = {(ord("m"), ord("i")): 256, (ord("u"), ord("m")): 257, (ord("u"), 256): 258}
    tokenizer = BPETokenizer(merges, None, gpt4_regex, {})
    tokens = [*tokenizer.encode_chunk(text)]
    assert len(tokens) == 2
    assert tokens[0] == ord("B")
    assert tokens[1] == 258


def test_encode_chunk_merge_then_merge_with_no_next(gpt4_regex):
    text = "Bumi"
    merges = {(ord("m"), ord("i")): 256, (ord("u"), 256): 257, (ord("B"), 257): 258, (ord("B"), ord("u")): 259}
    tokenizer = BPETokenizer(merges, None, gpt4_regex, {})
    tokens = [*tokenizer.encode_chunk(text)]
    assert len(tokens) == 1
    assert tokens[0] == 258


def test_encode_chunk_merge_multiple_same(gpt4_regex):
    text = "hahaha"
    merges = {(ord("h"), ord("a")): 256, (256, 256): 257}
    tokenizer = BPETokenizer(merges, None, gpt4_regex, {})
    tokens = [*tokenizer.encode_chunk(text)]
    assert len(tokens) == 2
    assert tokens[0] == 257
    assert tokens[1] == 256


def test_encode_chunk_single_token(gpt4_regex):
    text = "B"
    tokenizer = BPETokenizer({}, None, gpt4_regex, {})
    tokens = [*tokenizer.encode_chunk(text)]
    assert len(tokens) == 1
    assert tokens[0] == ord("B")
