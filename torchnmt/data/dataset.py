from .bpe import build_bpe_dataset
from .config import BuildDatasetConfig, DatasetConfig, LoadDatasetConfig
from .models import NMTDataset
from .word import build_word_dataset


_builders = dict(word=build_word_dataset, bpe=build_bpe_dataset)

_load_cache: dict[str, NMTDataset] = {}


def get_dataset(config: DatasetConfig) -> NMTDataset:
    """
    Retrieves or builds an NMT dataset based on the provided configuration.

    Args:
        config (DatasetConfig): The configuration for the dataset. This can be either:
            - `LoadDatasetConfig`: To load a pre-existing dataset from disk.
            - `BuildDatasetConfig`: To build a new dataset from specified parameters.

    Returns:
        NMTDataset: The neural machine translation dataset, either loaded from disk or newly built.
    """

    match config:
        case LoadDatasetConfig():
            if config.name in _load_cache:
                return _load_cache[config.name]
            dataset = NMTDataset.load(config)
            _load_cache[config.name] = dataset
            return dataset
        case BuildDatasetConfig():
            dataset = _builders[config.tokenizer.kind](config)
            if config.save:
                dataset.save()
            return dataset
