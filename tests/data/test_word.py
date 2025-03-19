import pytest

from torchnmt.data.config import FilterChain, LangPatterns, LenFilter, StartFilter, Sub
from torchnmt.data.word import FilterTokenizer, Normalizer, WordTokenizer, _build_vocab


def test_normalizer_untouched():
    normalizer = Normalizer("spa", "eng", False, False, None)

    example = "Quiero ver una película", "I want to see a movie"

    res = normalizer.normalize_example(example)

    assert res is not None
    assert res["src"] == "Quiero ver una película"
    assert res["tgt"] == "I want to see a movie"


def test_normalizer_lower():
    normalizer = Normalizer("spa", "eng", True, False, None)

    example = "Quiero ver una película", "I want to see a movie"

    res = normalizer.normalize_example(example)

    assert res is not None
    assert res["src"] == "quiero ver una película"
    assert res["tgt"] == "i want to see a movie"


def test_normalizer_unicode():
    normalizer = Normalizer("spa", "eng", False, True, None)

    example = "Quiero ver una película", "I want to see a movie"

    res = normalizer.normalize_example(example)

    assert res is not None
    assert res["src"] == "Quiero ver una pelicula"
    assert res["tgt"] == "I want to see a movie"


def test_normalizer_patterns_omit_src():
    patterns = {"spa": LangPatterns(omit="soy")}
    normalizer = Normalizer("spa", "eng", False, False, patterns)

    example = "Quiero ver una película", "I want to see a movie"

    res = normalizer.normalize_example(example)

    assert res is not None
    assert res["src"] == "Quiero ver una película"
    assert res["tgt"] == "I want to see a movie"


def test_normalizer_patterns_omit_src_fail():
    patterns = {"spa": LangPatterns(omit="ver")}
    normalizer = Normalizer("spa", "eng", False, False, patterns)

    example = "Quiero ver una película", "I want to see a movie"

    res = normalizer.normalize_example(example)

    assert res is None


def test_normalizer_patterns_omit_tgt():
    patterns = {"eng": LangPatterns(omit="go")}
    normalizer = Normalizer("spa", "eng", False, False, patterns)

    example = "Quiero ver una película", "I want to see a movie"

    res = normalizer.normalize_example(example)

    assert res is not None
    assert res["src"] == "Quiero ver una película"
    assert res["tgt"] == "I want to see a movie"


def test_normalizer_patterns_omit_tgt_fail():
    patterns = {"eng": LangPatterns(omit="want")}
    normalizer = Normalizer("spa", "eng", False, False, patterns)

    example = "Quiero ver una película", "I want to see a movie"

    res = normalizer.normalize_example(example)

    assert res is None


def test_normalizer_patterns_omit_src_tgt_fail():
    patterns = {"spa": LangPatterns(omit="ver"), "eng": LangPatterns(omit="want")}
    normalizer = Normalizer("spa", "eng", False, False, patterns)

    example = "Quiero ver una película", "I want to see a movie"

    res = normalizer.normalize_example(example)

    assert res is None


def test_normalizer_patterns_sub_src():
    patterns = {"spa": LangPatterns(subs=[Sub(ptn="a", repl="as")])}
    normalizer = Normalizer("spa", "eng", False, False, patterns)

    example = "Quiero ver una película", "I want to see a movie"

    res = normalizer.normalize_example(example)

    assert res is not None
    assert res["src"] == "Quiero ver unas películas"
    assert res["tgt"] == "I want to see a movie"


def test_normalizer_patterns_sub_tgt():
    patterns = {"eng": LangPatterns(subs=[Sub(ptn="nt", repl="nted")])}
    normalizer = Normalizer("spa", "eng", False, False, patterns)

    example = "Quiero ver una película", "I want to see a movie"

    res = normalizer.normalize_example(example)

    assert res is not None
    assert res["src"] == "Quiero ver una película"
    assert res["tgt"] == "I wanted to see a movie"


def test_normalizer_all():
    patterns = {
        "spa": LangPatterns(subs=[Sub(ptn="a", repl="as")]),
        "eng": LangPatterns(subs=[Sub(ptn="nt", repl="nted")]),
    }
    normalizer = Normalizer("spa", "eng", True, True, patterns)

    example = "Quiero ver una película", "I want to see a movie"

    res = normalizer.normalize_example(example)

    assert res is not None
    assert res["src"] == "quiero ver unas peliculas"
    assert res["tgt"] == "i wanted to see a movie"


@pytest.fixture
def gpt4_regex():
    return (
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
    )


def test_filter_tokenizer_no_filters(gpt4_regex):
    tokenizer = FilterTokenizer("<sos>", "<eos>", gpt4_regex, None, None)

    example = dict(src="Quiero ver una película", tgt="I want to see a movie")

    res = tokenizer.tokenize_example(example)

    assert res is not None
    assert res["src_tok_pre_vocab"] == ["<sos>", "Quiero", "ver", "una", "película", "<eos>"]
    assert res["tgt_tok_pre_vocab"] == ["<sos>", "I", "want", "to", "see", "a", "movie", "<eos>"]


def test_filter_tokenizer_filters(gpt4_regex):
    src_filters = FilterChain(filters=[LenFilter(max_len=4), StartFilter(start_with=["Quiero"])])
    tgt_filters = FilterChain(filters=[LenFilter(max_len=6), StartFilter(start_with=["I"])])
    tokenizer = FilterTokenizer("<sos>", "<eos>", gpt4_regex, src_filters, tgt_filters)

    example = dict(src="Quiero ver una película", tgt="I want to see a movie")

    res = tokenizer.tokenize_example(example)

    assert res is not None
    assert res["src_tok_pre_vocab"] == ["<sos>", "Quiero", "ver", "una", "película", "<eos>"]
    assert res["tgt_tok_pre_vocab"] == ["<sos>", "I", "want", "to", "see", "a", "movie", "<eos>"]


def test_filter_tokenizer_src_len_fail(gpt4_regex):
    src_filters = FilterChain(filters=[LenFilter(max_len=3), StartFilter(start_with=["Quiero"])])
    tgt_filters = FilterChain(filters=[LenFilter(max_len=6), StartFilter(start_with=["I"])])
    tokenizer = FilterTokenizer("<sos>", "<eos>", gpt4_regex, src_filters, tgt_filters)

    example = dict(src="Quiero ver una película", tgt="I want to see a movie")

    res = tokenizer.tokenize_example(example)

    assert res is None


def test_filter_tokenizer_src_start_fail(gpt4_regex):
    src_filters = FilterChain(filters=[LenFilter(max_len=4), StartFilter(start_with=["Yo"])])
    tgt_filters = FilterChain(filters=[LenFilter(max_len=6), StartFilter(start_with=["I"])])
    tokenizer = FilterTokenizer("<sos>", "<eos>", gpt4_regex, src_filters, tgt_filters)

    example = dict(src="Quiero ver una película", tgt="I want to see a movie")

    res = tokenizer.tokenize_example(example)

    assert res is None


def test_filter_tokenizer_tgt_len_fail(gpt4_regex):
    src_filters = FilterChain(filters=[LenFilter(max_len=4), StartFilter(start_with=["Quiero"])])
    tgt_filters = FilterChain(filters=[LenFilter(max_len=5), StartFilter(start_with=["I"])])
    tokenizer = FilterTokenizer("<sos>", "<eos>", gpt4_regex, src_filters, tgt_filters)

    example = dict(src="Quiero ver una película", tgt="I want to see a movie")

    res = tokenizer.tokenize_example(example)

    assert res is None


def test_filter_tokenizer_tgt_start_fail(gpt4_regex):
    src_filters = FilterChain(filters=[LenFilter(max_len=4), StartFilter(start_with=["Quiero"])])
    tgt_filters = FilterChain(filters=[LenFilter(max_len=6), StartFilter(start_with=["You"])])
    tokenizer = FilterTokenizer("<sos>", "<eos>", gpt4_regex, src_filters, tgt_filters)

    example = dict(src="Quiero ver una película", tgt="I want to see a movie")

    res = tokenizer.tokenize_example(example)

    assert res is None


@pytest.fixture
def tokenizer(gpt4_regex):
    specials = {"pad": 0, "start": 1, "end": 2, "unk": 3}
    stoi = {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        ".": 4,
        "?": 5,
        "Tom": 6,
        "a": 7,
        "es": 8,
        "!": 9,
        "No": 10,
        "un": 11,
        "Me": 12,
        "Yo": 13,
        "está": 14,
        "de": 15,
        "Él": 16,
        "la": 17,
        "lo": 18,
        "el": 19,
        "Estoy": 20,
        "Es": 21,
        "se": 22,
        "Soy": 23,
        "una": 24,
        "en": 25,
        "me": 26,
        "¿Quién": 27,
        "Lo": 28,
        ",": 29,
        "no": 30,
        "que": 31,
        "mi": 32,
        "esto": 33,
        "aquí": 34,
        "Ella": 35,
        "te": 36,
        "bien": 37,
        "gusta": 38,
        "eso": 39,
        "Tomás": 40,
        "los": 41,
        "estoy": 42,
        "Se": 43,
        "Eso": 44,
        "casa": 45,
        "Esto": 46,
        "Te": 47,
        "¿Es": 48,
        "Ellos": 49,
        "Tengo": 50,
    }
    itos = {
        0: "<pad>",
        1: "<sos>",
        2: "<eos>",
        3: "<unk>",
        4: ".",
        5: "?",
        6: "Tom",
        7: "a",
        8: "es",
        9: "!",
        10: "No",
        11: "un",
        12: "Me",
        13: "Yo",
        14: "está",
        15: "de",
        16: "Él",
        17: "la",
        18: "lo",
        19: "el",
        20: "Estoy",
        21: "Es",
        22: "se",
        23: "Soy",
        24: "una",
        25: "en",
        26: "me",
        27: "¿Quién",
        28: "Lo",
        29: ",",
        30: "no",
        31: "que",
        32: "mi",
        33: "esto",
        34: "aquí",
        35: "Ella",
        36: "te",
        37: "bien",
        38: "gusta",
        39: "eso",
        40: "Tomás",
        41: "los",
        42: "estoy",
        43: "Se",
        44: "Eso",
        45: "casa",
        46: "Esto",
        47: "Te",
        48: "¿Es",
        49: "Ellos",
        50: "Tengo",
    }
    default_ix = 3
    return WordTokenizer(stoi, itos, default_ix, specials, gpt4_regex)


def test_tokenizer_encode(tokenizer):
    res = tokenizer.encode("No me gusta Tom")
    assert res == [1, 10, 26, 38, 6, 2]


def test_tokenizer_encode_unk(tokenizer):
    res = tokenizer.encode("No me gusta Tom ahora")
    assert res == [1, 10, 26, 38, 6, 3, 2]


def test_tokenizer_encode_tokens(tokenizer):
    res = tokenizer.encode_tokens(["<sos>", "Tengo", "una", "casa", "<eos>"])
    assert res == [1, 50, 24, 45, 2]


def test_tokenizer_encode_tokens_unk(tokenizer):
    res = tokenizer.encode_tokens(["<sos>", "Tengo", "una", "coche", "<eos>"])
    assert res == [1, 50, 24, 3, 2]


def test_tokenizer_decode_tokens(tokenizer):
    res = tokenizer.decode_tokens([1, 10, 26, 38, 6, 2], strip_specials=False)
    assert res == ["<sos>", "No", "me", "gusta", "Tom", "<eos>"]


def test_tokenizer_decode_tokens_strip(tokenizer):
    res = tokenizer.decode_tokens([1, 10, 26, 38, 6, 2], strip_specials=True)
    assert res == ["No", "me", "gusta", "Tom"]


def test_tokenizer_decode(tokenizer):
    res = tokenizer.decode([1, 10, 26, 38, 6, 2], strip_specials=False)
    assert res == "<sos> No me gusta Tom <eos>"


def test_tokenizer_decode_strip(tokenizer):
    res = tokenizer.decode([1, 10, 26, 38, 6, 2], strip_specials=True)
    assert res == "No me gusta Tom"


def test_tokenizer_vocab_size(tokenizer):
    assert tokenizer.vocab_size == 51


def test_build_vocab():
    examples = (
        ["yo", "soy"],
        ["yo", "tengo"],
        ["me", "gusta"],
        ["te", "gusta"],
        ["soy", "un", "abogado"],
        ["quiero", "uno"],
        ["quiero", "una"],
        ["es", "un", "perro"],
        ["tengo", "un", "gato"],
        ["tengo", "una", "coche"],
    )
    specials = {"pad": "<pad>", "start": "<sos>", "end": "<eos>", "unk": "<unk>"}
    min_freq = 2
    # yo 2, soy 2, tengo 3, me 1, gusta 2, te 1, un 3, abogado 1, quiero 2,
    # uno 1, una 2, coche 1
    stoi, itos = _build_vocab(examples, specials, min_freq)

    assert len(stoi) == 11
    assert stoi["<pad>"] == 0
    assert stoi["<sos>"] == 1
    assert stoi["<eos>"] == 2
    assert stoi["<unk>"] == 3
    assert stoi["tengo"] == 4
    assert stoi["un"] == 5
    assert stoi["yo"] == 6
    assert stoi["soy"] == 7
    assert stoi["gusta"] == 8
    assert stoi["quiero"] == 9
    assert stoi["una"] == 10

    assert len(itos) == 11
    assert itos[0] == "<pad>"
    assert itos[1] == "<sos>"
    assert itos[2] == "<eos>"
    assert itos[3] == "<unk>"
    assert itos[4] == "tengo"
    assert itos[5] == "un"
    assert itos[6] == "yo"
    assert itos[7] == "soy"
    assert itos[8] == "gusta"
    assert itos[9] == "quiero"
    assert itos[10] == "una"
