from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
from bertviz import head_view, model_view  # type: ignore
from rich.console import Console
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from torchinfo import ModelStatistics, summary

from torchnmt.config import BeamConfig, EvalConfig
from torchnmt.data import Batch, Collator, NMTDataset, RNNBatch, RNNCollator, TransformerBatch, TransformerCollator
from torchnmt.forward import Forward, RNNForward, TransformerForward
from torchnmt.model import PadMasker, RNNSeq2Seq, TransformerSeq2Seq
from torchnmt.search import BeamSearchResult, RNNBeamSearcher, TransformerBeamSearcher


B = TypeVar("B", bound=Batch)
M = TypeVar("M", bound=nn.Module)


class NMTEval(ABC, Generic[B, M]):
    def __init__(
        self,
        name: str,
        config: EvalConfig,
        dataset: NMTDataset,
        model: M,
        device: str,
        collator_type: type[Collator],
        forward_type: type[Forward],
    ):
        self.name = name
        self.dataset = dataset
        self.device = device
        self.model = model
        self.config = config

        self.ignore_index = self.dataset.tgt_specials["pad"]
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction="sum")

        self.forward = forward_type(self.model, self.device, self.loss_fn)

        self.collate_fn = collator_type(self.dataset.src_specials["pad"], self.dataset.tgt_specials["pad"])

        train_loader = DataLoader(
            self.dataset.train_set,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
        )
        val_loader = DataLoader(
            self.dataset.val_set,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
        )
        test_loader = DataLoader(
            self.dataset.test_set,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.config.num_workers,
        )
        self.loaders = {"train": train_loader, "val": val_loader, "test": test_loader}

        train_sampler = DataLoader(
            self.dataset.train_set, batch_size=self.config.sample_size, shuffle=True, collate_fn=self.collate_fn
        )
        val_sampler = DataLoader(
            self.dataset.val_set, batch_size=self.config.sample_size, shuffle=True, collate_fn=self.collate_fn
        )
        test_sampler = DataLoader(
            self.dataset.test_set, batch_size=self.config.sample_size, shuffle=True, collate_fn=self.collate_fn
        )
        self.samplers = {"train": train_sampler, "val": val_sampler, "test": test_sampler}

        self.console = Console()

    @torch.no_grad()
    def eval(self, split: str):
        if split not in self.loaders:
            raise ValueError(f"Invalid split: {split}")

        loader = self.loaders[split]

        self.model.eval()

        running_loss = 0.0
        total_cnt = 0
        for _, batch in enumerate(loader):
            loss, cnt = self.forward(batch)
            running_loss += loss.item()
            total_cnt += cnt

        avg_loss = running_loss / total_cnt

        print(f"\n{split.capitalize()}: ")
        print(f" loss: {avg_loss:.3f}")

        return avg_loss

    @torch.no_grad()
    def translate_sample(self, split: str, indices: Optional[list[int]] = None):
        if split not in self.samplers:
            raise ValueError(f"Invalid split: {split}")

        if indices:
            dataset = getattr(self.dataset, f"{split}_set")
            batch = self.collate_fn([dataset[i] for i in indices])
        else:
            sampler = self.samplers[split]
            batch = next(iter(sampler))

        self.translate_batch(batch)

    @abstractmethod
    def summary(self, batch: B):
        raise NotImplementedError

    @abstractmethod
    def translate_sentence(self, *args, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def translate_batch(self, batch: B):
        raise NotImplementedError

    @abstractmethod
    def translate(self, *args, **kwargs) -> list[BeamSearchResult]:
        raise NotImplementedError


class RecurrentNMTEval(NMTEval[RNNBatch, RNNSeq2Seq]):
    def __init__(self, name: str, config: EvalConfig, dataset: NMTDataset, model: RNNSeq2Seq, device: str):
        super().__init__(name, config, dataset, model, device, RNNCollator, RNNForward)
        self.beam_searcher = RNNBeamSearcher(model.decoder, self.dataset, self.device, config.beam)

    @torch.no_grad()
    def summary(self, batch: RNNBatch) -> ModelStatistics:
        src, src_len, tgt, _ = batch
        input_data = (src.to(self.device), src_len, tgt.to(self.device))
        return summary(self.model, input_data=input_data)

    @torch.no_grad()
    def translate_sentence(self, s: str, attention: bool = False, beam_config: Optional[BeamConfig] = None) -> str:
        src_ix = self.dataset.src_tokenizer.encode(s)

        src = torch.tensor(src_ix).unsqueeze(0).to(self.device)
        src_len = torch.tensor([len(src_ix)], dtype=torch.int)

        preds = self.translate(src, src_len, beam_config)
        pred = preds[0]

        if attention and pred.att is not None:
            self.show_attention(src_ix, pred.ix, pred.att)

        return pred.s

    @torch.no_grad()
    def translate_batch(self, batch: RNNBatch, beam_config: BeamConfig | None = None):
        src, src_len, tgt, examples = batch

        preds = self.translate(src, src_len, beam_config)

        table = Table(show_header=True, show_lines=True)
        table.add_column("Source")
        table.add_column("Target")
        table.add_column("Prediction")

        for i in range(len(preds)):
            table.add_row(examples[i]["src"], examples[i]["tgt"], preds[i].s)

        self.console.print(table)

    @torch.no_grad()
    def translate(self, src: Tensor, src_len: Tensor, beam_config: BeamConfig | None = None) -> list[BeamSearchResult]:
        self.model.eval()
        src = src.to(self.device)

        enc_out, enc_hidden = self.model.encoder(src, src_len)

        preds = [ex_results[0] for ex_results in self.beam_searcher.search(src, enc_out, enc_hidden, beam_config)]
        return preds

    def show_attention(self, src_ix: list[int], pred_ix: list[int], attention: Tensor):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        attention = attention.numpy()

        ax.matshow(attention)

        ax.tick_params(labelsize=15)

        x_ticks = ["", *self.dataset.src_tokenizer.decode_tokens(src_ix)]
        y_ticks = ["", *self.dataset.tgt_tokenizer.decode_tokens(pred_ix)]

        ax.set_xticklabels(x_ticks, rotation=45)
        ax.set_yticklabels(y_ticks)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()


class TransformerNMTEval(NMTEval[TransformerBatch, TransformerSeq2Seq]):
    def __init__(self, name: str, config: EvalConfig, dataset: NMTDataset, model: TransformerSeq2Seq, device: str):
        super().__init__(name, config, dataset, model, device, TransformerCollator, TransformerForward)
        self.beam_searcher = TransformerBeamSearcher(model.decoder, self.dataset, self.device, config.beam)
        self.naive = model.encoder.params.naive
        self.src_masker = PadMasker.from_params(self.model.encoder.params)

    @torch.no_grad()
    def summary(self, batch: TransformerBatch) -> ModelStatistics:
        src, tgt, _ = batch
        input_data = (src.to(self.device), tgt.to(self.device))
        return summary(self.model, input_data=input_data)

    @torch.no_grad()
    def translate_sentence(self, s: str, attention: Optional[str] = None, beam_config: BeamConfig | None = None) -> str:
        src_ix = self.dataset.src_tokenizer.encode(s)

        src = torch.tensor(src_ix).unsqueeze(0).to(self.device)

        preds = self.translate(src, attention=attention is not None, beam_config=beam_config)
        pred = preds[0]

        if pred.att is not None:
            cross_attention = [pred.att[i].unsqueeze(0) for i in range(len(pred.att))]
            encoder_tokens = self.dataset.src_tokenizer.decode_tokens(src_ix)
            decoder_tokens = self.dataset.tgt_tokenizer.decode_tokens(pred.ix)
            if attention == "head":
                head_view(cross_attention=cross_attention, encoder_tokens=encoder_tokens, decoder_tokens=decoder_tokens)
            elif attention == "model":
                model_view(
                    cross_attention=cross_attention, encoder_tokens=encoder_tokens, decoder_tokens=decoder_tokens
                )

        return pred.s

    @torch.no_grad()
    def translate_batch(self, batch: TransformerBatch, beam_config: BeamConfig | None = None):
        src, tgt, examples = batch

        preds = self.translate(src, beam_config=beam_config)

        table = Table(show_header=True, show_lines=True)
        table.add_column("Source")
        table.add_column("Target")
        table.add_column("Prediction")

        for i in range(len(preds)):
            table.add_row(examples[i]["src"], examples[i]["tgt"], preds[i].s)

        self.console.print(table)

    @torch.no_grad()
    def translate(
        self, src: Tensor, attention: bool = False, beam_config: BeamConfig | None = None
    ) -> list[BeamSearchResult]:
        self.model.eval()
        src = src.to(self.device)

        src_mask = self.src_masker(src)

        enc_out, _ = self.model.encoder(src, src_mask)

        return [
            ex_results[0]
            for ex_results in self.beam_searcher.search(enc_out, src_mask, attention=attention, config=beam_config)
        ]
