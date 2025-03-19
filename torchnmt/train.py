from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from pydantic import RootModel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from torchnmt.config import Config, Losses, Metric, RNNTrainConfig, TrainConfig
from torchnmt.data import Batch, Collator, NMTDataset, RNNBatch, RNNCollator, TransformerBatch, TransformerCollator
from torchnmt.eval import NMTEval, RecurrentNMTEval, TransformerNMTEval
from torchnmt.forward import Forward, RNNForward, TransformerForward
from torchnmt.model import RNNSeq2Seq, TransformerSeq2Seq


C = TypeVar("C", bound=TrainConfig)
B = TypeVar("B", bound=Batch)
M = TypeVar("M", bound=nn.Module)


class NMTTrainer(ABC, Generic[C, B, M]):
    def __init__(
        self,
        config: Config,
        train_config: C,
        dataset: NMTDataset,
        model: M,
        collator_type: type[Collator],
        forward_type: type[Forward],
        evaluator: NMTEval,
    ):
        self.config = config
        self.train_config = train_config
        self.dataset = dataset
        self.device = self.config.device
        self.model = model
        self.evaluator = evaluator

        self.epoch = 1
        self.opt = Adam(self.model.parameters(), lr=self.train_config.lr)

        self.ignore_index = self.dataset.tgt_specials["pad"]
        self.loss_fn = CrossEntropyLoss(ignore_index=self.ignore_index)

        self.forward = forward_type(self.model, self.device, self.loss_fn)
        self.collate_fn = collator_type(self.dataset.src_specials["pad"], self.dataset.tgt_specials["pad"])

        self.train_loader = DataLoader(
            self.dataset.train_set,
            batch_size=self.train_config.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            num_workers=self.train_config.num_workers,
        )

        self.report_iters = int(len(self.train_loader) / self.train_config.report_per_epoch)

        self.losses = Losses(train_loss=Metric(), val_loss=[])

    def train_summary(self):
        return self.evaluator.summary(next(iter(self.train_loader)))

    def overfit_batch(self, batch: B, n: int = 200):
        self.model.train()
        for _ in range(n):
            loss, _ = self.train_batch(batch)
            print(f"{loss:.3f}")
        self.evaluator.translate_batch(batch)

    def train(self):
        self.train_n_epochs(self.train_config.n_epochs)

    def train_n_epochs(self, n_epochs: int):
        for e in range(self.epoch, self.epoch + n_epochs):
            self.model.train()
            train_batches = tqdm(self.train_loader, unit="batch")
            train_batches.set_description(f"Epoch {e}")

            self.train_epoch(train_batches)

            val_loss = self.evaluator.eval("val")
            self.losses.val_loss.append((self.epoch * len(self.train_loader), val_loss))

            if e % self.train_config.sample_epoch_freq == 0:
                print()
                self.evaluator.translate_sample("train")
                print()

            self.post_epoch_hook()
            if self.train_config.save_policy(self):
                self.save_checkpoint()
            self.epoch += 1

    def train_epoch(self, batches: tqdm):
        for i, batch in enumerate(batches):
            loss, cnt = self.train_batch(batch)

            self.losses.train_loss.add(loss, cnt)

            if i % self.report_iters == self.report_iters - 1:
                avg_train_loss = self.losses.train_loss.avg()
                tqdm.write(
                    f"loss: {avg_train_loss:.3f} [{(i + 1) * self.train_config.batch_size:>5d}/{len(self.train_loader.dataset)}]"  # type: ignore [arg-type]
                )

    @abstractmethod
    def train_batch(self, batch: B) -> tuple[float, int]:
        pass

    def plot_losses(self):
        plt.plot(self.losses.train_loss.vals)
        z = list(zip(*self.losses.val_loss, strict=True))
        plt.plot(z[0], z[1])
        plt.show()

    def save_checkpoint(self, manual: bool = False):
        path = Path(self.train_config.checkpoint_dir)
        path.mkdir(exist_ok=True)
        experiment_path = path / self.config.experiment
        experiment_path.mkdir(exist_ok=True)

        model_path = experiment_path / self.config.name
        if not model_path.exists():
            model_path.mkdir()

            with open(model_path / "config.yaml", "w") as config_file:
                yaml.dump(RootModel[Config](self.config).model_dump(mode="json", exclude_none=True), config_file)

        epoch = self.epoch - 1 if manual else self.epoch
        checkpoint_name = f"0{epoch}" if epoch < 10 else str(epoch)

        checkpoint_path = model_path / checkpoint_name
        checkpoint_path.mkdir()

        print(f"\nSaving checkpoint to: {checkpoint_path}")

        torch.save(self.model.state_dict(), checkpoint_path / "model.pt")

        train_dict = {"opt": self.opt.state_dict(), "epoch": epoch, "losses": self.losses}
        if save_fields := self.get_save_fields():
            train_dict["fields"] = save_fields
        torch.save(train_dict, checkpoint_path / "train.pt")

        print("Checkpoint saved!\n")

    def post_epoch_hook(self):
        pass

    def get_save_fields(self) -> dict | None:
        pass


class RecurrentNMTTrainer(NMTTrainer[RNNTrainConfig, RNNBatch, RNNSeq2Seq]):
    def __init__(self, config: Config, dataset: NMTDataset, model: RNNSeq2Seq):
        evaluator = RecurrentNMTEval(config.name, config.eval_, dataset, model, config.device)
        if type(config.train) is not RNNTrainConfig:
            raise ValueError("Invalid RNN training config")
        super().__init__(config, config.train, dataset, model, RNNCollator, RNNForward, evaluator)
        tf_policy = self.config.train.teacher_forcing_policy  # type: ignore
        self.teacher_forcing_rate = 0.0 if tf_policy is None else tf_policy.start_rate

    def train_batch(self, batch: RNNBatch) -> tuple[float, int]:
        loss, cnt = self.forward(batch, teacher_forcing_rate=self.teacher_forcing_rate)
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)

        self.opt.step()
        self.opt.zero_grad()

        return loss.item(), cnt

    def post_epoch_hook(self):
        if self.teacher_forcing_rate > 0:
            tf_policy = self.config.train.teacher_forcing_policy
            self.teacher_forcing_rate = tf_policy.start_rate * pow(tf_policy.decay_rate, self.epoch)
        return None

    def get_save_fields(self) -> dict | None:
        if self.teacher_forcing_rate > 0:
            return dict(teacher_forcing_rate=self.teacher_forcing_rate)
        return None


class TransformerNMTTrainer(NMTTrainer[TrainConfig, TransformerBatch, TransformerSeq2Seq]):
    def __init__(self, config: Config, dataset: NMTDataset, model: TransformerSeq2Seq):
        evaluator = TransformerNMTEval(config.name, config.eval_, dataset, model, config.device)
        if type(config.train) is not TrainConfig:
            raise ValueError("Invalid training config")
        super().__init__(config, config.train, dataset, model, TransformerCollator, TransformerForward, evaluator)

    def train_batch(self, batch: TransformerBatch) -> tuple[float, int]:
        loss, cnt = self.forward(batch)
        loss.backward()

        self.opt.step()
        self.opt.zero_grad()

        return loss.item(), cnt
