import gzip
import re
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Generator
from itertools import chain
from pathlib import Path
from xml.etree.ElementTree import iterparse

import requests
from iso639 import Lang
from requests import HTTPError

from .config import BuildDatasetConfig


class Source(ABC):
    """
    An abstract base class representing a source of bilingual text translations.

    This class defines the interface for fetching translation examples - subclasses
    must implement the `get_examples` method to provide a generator of translation examples.

    Attributes:
        src_lang (Lang): The source language.
        tgt_lang (Lang): The target language.
        dir_ (Path): The directory to store or retrieve the raw dataset.
    """

    def __init__(self, config: BuildDatasetConfig, dataset: str):
        self.src_lang = Lang(config.src_lang)
        self.tgt_lang = Lang(config.tgt_lang)

        self.dir_ = Path(config.dir_) / "raw" / dataset
        self.dir_.mkdir(parents=True, exist_ok=True)

        if self.src_lang.name != "English" and self.tgt_lang.name != "English":
            raise ValueError("One of the languages must be English.")

    @abstractmethod
    def get_examples(self) -> Generator[tuple[str, str], None, None]:
        """
        Yields bilingual text translations as tuples of source and target texts.
        """

        pass


class TatoebaSource(Source):
    def __init__(self, config: BuildDatasetConfig):
        super().__init__(config, "tatoeba")
        self.lang, self.reverse = (
            (self.tgt_lang.pt2b, False) if self.src_lang.name == "English" else (self.src_lang.pt2b, True)
        )

    def download_file(self, zip_path):
        file_name = f"{self.lang}-eng.zip"
        url = f"https://www.manythings.org/anki/{file_name}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }

        r = requests.get(url, headers=headers)
        r.raise_for_status()

        with open(zip_path, "wb") as f:
            f.write(r.content)

        extract_path = zip_path.with_suffix("")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_path)

    def ensure_file(self):
        dir_name = f"{self.lang}-eng"
        extract_path = self.dir_ / dir_name

        file_path = extract_path / f"{self.lang}.txt"
        if not file_path.exists():
            self.download_file(extract_path.with_suffix(".zip"))
        return file_path

    def get_examples(self) -> Generator[tuple[str, str], None, None]:
        data_path = self.ensure_file()
        if not data_path.exists():
            raise FileNotFoundError

        with open(data_path, encoding="utf-8") as f:
            for line in f:
                str_1, str_2, *_ = line.split("\t")
                yield (str_1, str_2) if not self.reverse else (str_2, str_1)


class OpusSource(Source):
    def __init__(self, config: BuildDatasetConfig, dataset: str):
        super().__init__(config, dataset)
        self.dataset = dataset
        self.lang = self.tgt_lang.pt1 if self.src_lang.name == "English" else self.src_lang.pt1
        self._parans_pattern = re.compile(r"\([^\)]+\)\s?")
        self._multi_dash_pattern = re.compile(r"(--(?: --)+)")

    def download_file(self, file_path: Path, file_name: str):
        url = f"https://object.pouta.csc.fi/OPUS-{self.dataset}/v1/tmx/{file_name}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Host": "object.pouta.csc.fi",
        }

        r = requests.get(url, headers=headers)
        r.raise_for_status()

        with open(file_path, "wb") as f:
            f.write(r.content)

    def ensure_file(self) -> tuple[Path, bool]:
        file_names = [f"en-{self.lang}.tmx.gz", f"{self.lang}-en.tmx.gz"]
        file_paths = [self.dir_ / file_name for file_name in file_names]

        for i, file_path in enumerate(file_paths):
            if file_path.exists():
                return file_path, i == 1

        try:
            self.download_file(file_paths[0], file_names[0])
            return file_paths[0], False
        except HTTPError:
            self.download_file(file_paths[1], file_names[1])

        return file_paths[1], True

    def get_examples(self) -> Generator[tuple[str, str], None, None]:
        file_path, is_reversed = self.ensure_file()

        last = None
        with gzip.open(file_path) as f:
            for _, elem in iterparse(f, events=("end",)):
                if elem.tag != "seg":
                    continue
                if elem.text is None:
                    if last is not None:
                        last = None
                    continue

                curr = elem.text
                curr = re.sub(self._parans_pattern, "", curr)
                curr = re.sub(self._multi_dash_pattern, "--", curr)
                curr = curr.strip()

                if last is None:
                    last = curr
                    continue
                if len(last) < 2 or len(curr) < 2 or len(last) > 200 or len(curr) > 200:
                    last = None
                    continue

                if self.src_lang.name == "English":
                    res = (last, curr) if not is_reversed else (curr, last)
                else:
                    res = (curr, last) if not is_reversed else (last, curr)
                last = None
                yield res


def get_source(source: str, config: BuildDatasetConfig) -> Source:
    match source:
        case "tatoeba":
            return TatoebaSource(config)
        case _ as dataset:
            return OpusSource(config, dataset)


def get_examples(config: BuildDatasetConfig) -> chain[tuple[str, str]]:
    return chain.from_iterable(get_source(source, config).get_examples() for source in config.sources)
