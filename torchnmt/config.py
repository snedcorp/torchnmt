from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Protocol, Union

from pydantic import BaseModel, Field, NonNegativeInt, PositiveFloat, PositiveInt
from pydantic.dataclasses import dataclass as pydantic_dataclass

from torchnmt.data import DatasetConfig
from torchnmt.model import ModelConfig


class Metric:
    def __init__(self):
        self.vals = []
        self.avgs = []
        self.running_val = 0.0
        self.running_cnt = 0

    def add(self, val: float, cnt: int):
        self.vals.append(val)
        self.running_val += val * cnt
        self.running_cnt += cnt

    def avg(self):
        avg = self.running_val / self.running_cnt
        self.avgs.append(avg)
        self.running_val = 0.0
        self.running_cnt = 0

        return avg


@dataclass
class Losses:
    train_loss: Metric
    val_loss: list[tuple[int, float]]


class Trainer(Protocol):
    epoch: int
    losses: Losses


class SavePolicy(ABC):
    @abstractmethod
    def __call__(self, trainer: Trainer) -> bool:
        pass


class EpochFreqPolicy(SavePolicy, BaseModel):
    freq: PositiveInt

    def __call__(self, trainer: Trainer) -> bool:
        return trainer.epoch % self.freq == 0


class BestValLossPolicy(SavePolicy, BaseModel):
    threshold: PositiveFloat
    _curr_best: Optional[float] = None

    def __call__(self, trainer: Trainer) -> bool:
        val_loss = trainer.losses.val_loss[-1][1]
        if val_loss <= self.threshold and (self._curr_best is None or val_loss < self._curr_best):
            self._curr_best = val_loss
            return True
        return False


@pydantic_dataclass
class TrainConfig:
    checkpoint_dir: str
    n_epochs: PositiveInt
    batch_size: PositiveInt
    sample_size: PositiveInt
    report_per_epoch: PositiveInt
    sample_epoch_freq: PositiveInt
    lr: PositiveFloat
    save_policy: Union[BestValLossPolicy, EpochFreqPolicy]
    num_workers: NonNegativeInt = 0


class TeacherForcingPolicy(BaseModel):
    start_rate: float = Field(ge=0, le=1)
    decay_rate: float = Field(ge=0, le=1)


@pydantic_dataclass
class RNNTrainConfig(TrainConfig):
    grad_clip: PositiveFloat = 1.0
    teacher_forcing_policy: TeacherForcingPolicy | None = None


@pydantic_dataclass
class BeamConfig:
    """
    Configuration class for beam search.

    Attributes:
        beam_width (PositiveInt): The number of beams to maintain during the search process.
        n_best (PositiveInt): The desired number of target sequences to generate for each source example.
    """

    beam_width: PositiveInt
    n_best: PositiveInt


@pydantic_dataclass
class EvalConfig:
    batch_size: PositiveInt
    sample_size: PositiveInt
    beam: BeamConfig
    num_workers: NonNegativeInt = 0


@pydantic_dataclass(kw_only=True)
class Config:
    dataset: DatasetConfig = Field(discriminator="kind")
    model: ModelConfig = Field(discriminator="kind")
    eval_: EvalConfig
    train: Optional[Union[TrainConfig, RNNTrainConfig]] = None
    experiment: str
    name: str
    device: str
