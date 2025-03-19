import os
from pathlib import Path
from typing import Any

import torch
import yaml
from hydra import compose, initialize
from omegaconf import OmegaConf

from torchnmt.data import NMTDataset, get_dataset
from torchnmt.eval import NMTEval, RecurrentNMTEval, TransformerNMTEval
from torchnmt.model import LoadModelConfig, TransformerSeq2Seq, build_model
from torchnmt.train import Config, NMTTrainer, RecurrentNMTTrainer, TransformerNMTTrainer


def get_config(overrides: list[str] | None = None):
    if overrides is None:
        overrides = []
    overrides.append(f"dataset.dir_={os.getenv('DATA_DIR')}")
    with initialize(version_base=None, config_path="../config", job_name="config"):
        cfg = compose(config_name="config", overrides=overrides)
        print(cfg)
        return Config(**OmegaConf.to_container(cfg))  # type: ignore


def load_model(config: LoadModelConfig, dataset: NMTDataset, train: bool) -> tuple[Config, Any, Any]:
    model_dir_path = Path(config.dir_) / config.experiment / config.name
    config_path = model_dir_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    with open(config_path) as config_file:
        loaded_config = Config(**yaml.safe_load(config_file))

    if dataset.name != loaded_config.dataset.name:
        raise ValueError(
            f"Loaded model's dataset {loaded_config.dataset.name} does not match specified dataset {dataset.name}"
        )

    if config.epoch is None:
        checkpoints = [child for child in model_dir_path.iterdir() if child.is_dir()]
        if not checkpoints:
            raise ValueError(f"No checkpoints in dir: {model_dir_path}")
        checkpoints.sort()
        checkpoint_path = checkpoints[-1]
    else:
        checkpoint_path = model_dir_path / (f"0{config.epoch}" if config.epoch < 10 else str(config.epoch))

    print(f"Loading checkpoint from: {checkpoint_path}")
    model_dict = torch.load(checkpoint_path / "model.pt")
    train_dict = None
    if train:
        train_dict = torch.load(checkpoint_path / "train.pt")
    print("Checkpoint loaded!")

    return loaded_config, model_dict, train_dict


def load_trainer(config: LoadModelConfig, dataset: NMTDataset):
    loaded_config, model_dict, train_dict = load_model(config, dataset, train=True)

    trainer = build_trainer(loaded_config, dataset)

    trainer.model.load_state_dict(model_dict)
    trainer.opt.load_state_dict(train_dict["opt"])
    trainer.epoch = train_dict["epoch"] + 1
    trainer.losses = train_dict["losses"]

    if "fields" in train_dict:
        for field, val in train_dict["fields"].items():
            setattr(trainer, field, val)

    return trainer


def build_trainer(config: Config, dataset: NMTDataset) -> NMTTrainer:
    model = build_model(config.model, dataset, config.device)
    if isinstance(model, TransformerSeq2Seq):
        return TransformerNMTTrainer(config, dataset, model)
    return RecurrentNMTTrainer(config, dataset, model)


def get_trainer(config: Config) -> NMTTrainer:
    dataset = get_dataset(config.dataset)
    if type(config.model) is LoadModelConfig:
        return load_trainer(config.model, dataset)
    return build_trainer(config, dataset)


def get_evaluator(config: Config) -> NMTEval:
    if type(config.model) is not LoadModelConfig:
        raise ValueError(f"Invalid model config for evaluator: {type(config.model)}")
    dataset = get_dataset(config.dataset)

    loaded_config, model_dict, _ = load_model(config.model, dataset, train=False)

    model = build_model(loaded_config.model, dataset, config.device)

    if isinstance(model, TransformerSeq2Seq):
        evaluator: NMTEval = TransformerNMTEval(
            name=loaded_config.name, config=config.eval_, dataset=dataset, model=model, device=config.device
        )
    else:
        evaluator = RecurrentNMTEval(
            name=loaded_config.name, config=config.eval_, dataset=dataset, model=model, device=config.device
        )

    evaluator.model.load_state_dict(model_dict)
    return evaluator
