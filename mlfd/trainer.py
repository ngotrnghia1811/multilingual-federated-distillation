"""
MlFD Trainer — HuggingFace-based training with federated knowledge distillation.

Two classes are defined:
  - LogitTracker: accumulates per-label logit sums during local training, and
    computes the local contribution to global distillation targets.
  - Trainer: extends Plato's basic Trainer to use a custom HuggingFace Trainer
    (SampledHuggingFaceTrainer) that incorporates the KD regularization loss.

Loss formulation (each training step):
    loss = classification_loss + reg_alpha * KL(local_log_probs || global_log_probs)

where global_log_probs are derived from the server-aggregated per-label logits
sent at the start of each round.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, Sampler

import evaluate
from easydict import EasyDict as edict
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)
from transformers import Trainer as HuggingFaceTrainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from plato.config import Config
from plato.trainers import basic


class LogitTracker:
    """Tracks per-label mean logit vectors for federated knowledge distillation.

    Each client accumulates the sum of logits grouped by ground-truth label during
    local training. These per-label averages are sent to the server, which aggregates
    them across clients. The resulting global logits are used as soft distillation
    targets in the next round.
    """

    def __init__(self, unique_labels: int):
        self.unique_labels = unique_labels
        self.label_counts = torch.ones(unique_labels)
        self.logit_sums = torch.zeros((unique_labels, unique_labels))
        self.num_agg_clients = 1
        self.skip_kd = False
        self.global_logits = None

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """Accumulate logit sums grouped by label.

        Args:
            logits: Log-probabilities of shape (batch_size, num_labels).
            labels: Ground-truth label indices of shape (batch_size,).
        """
        batch_unique, batch_counts = labels.unique(dim=0, return_counts=True)
        self.label_counts[batch_unique.cpu()] += batch_counts.cpu()

        expanded_labels = labels.view(-1, 1).expand(-1, logits.size(1)).cpu()
        bucket = torch.zeros((self.unique_labels, self.unique_labels))
        bucket.scatter_add_(0, expanded_labels, logits.cpu())
        self.logit_sums += bucket

    def avg(self) -> torch.Tensor:
        """Return per-label average logit tensor of shape (num_labels, num_labels)."""
        return self.logit_sums / self.label_counts.float().unsqueeze(1)

    def server_avg(self, sum_logits: torch.Tensor):
        """Compute global distillation targets from the server's aggregated sum.

        The server sends sum_logits = sum of avg() across all clients. To exclude
        the local client's contribution:
            global = (sum_logits - self.avg()) / (num_agg_clients - 1)
        """
        self.global_logits = (sum_logits - self.avg()) / (self.num_agg_clients - 1)


@dataclass
class LabelSmoother:
    """Label-smoothed cross-entropy that also returns log-probabilities.

    Returns an EasyDict with keys:
        - loss: scalar smoothed cross-entropy loss
        - log_probs: log-probability tensor of shape (batch_size, num_labels)
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels: bool = False):
        logits = (
            model_output["logits"]
            if isinstance(model_output, dict)
            else model_output[0]
        )
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -F.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        n_active = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / n_active
        smoothed_loss = smoothed_loss.sum() / (n_active * log_probs.shape[-1])

        return edict(
            {
                "loss": (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss,
                "log_probs": log_probs,
            }
        )


class SampledHuggingFaceTrainer(HuggingFaceTrainer):
    """HuggingFace Trainer extended with:
      - a custom sampler (for Plato's partitioning)
      - KD regularization in compute_loss
    """

    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        tokenizer,
        data_collator,
        sampler,
        callbacks,
        compute_metrics=None,
        logit_tracker=None,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )
        self.sampler = sampler
        self.logit_tracker = logit_tracker

        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(
                epsilon=self.args.label_smoothing_factor
            )
        else:
            self.label_smoother = None

        self.kd_loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.reg_alpha = Config().trainer.reg_alpha

    def _get_train_sampler(self) -> Optional[Sampler]:
        if self.sampler is None:
            return RandomSampler(self.train_dataset)
        return self.sampler

    def _get_eval_sampler(self, eval_dataset) -> Optional[Sampler]:
        if self.sampler is None:
            return super()._get_eval_sampler(eval_dataset)
        return self.sampler

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        """Compute classification loss plus optional KD regularization.

        KD is applied only when global_logits are available (after the first round).
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            is_causal = (
                unwrap_model(model)._get_name()
                in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()
            )
            smoothed = self.label_smoother(outputs, labels, shift_labels=is_causal)
            clss_loss, log_probs = smoothed.loss, smoothed.log_probs
        else:
            clss_loss = (
                outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            )
            log_probs = None

        if model.training and not self.logit_tracker.skip_kd and log_probs is not None:
            global_log_probs = self.logit_tracker.global_logits.to(log_probs.device)[
                labels, :
            ]
            output_logits = (
                outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            )
            output_log_probs = F.log_softmax(output_logits, dim=1)

            self.logit_tracker.update(output_log_probs, labels)
            kd_loss = self.kd_loss_fn(output_log_probs, global_log_probs)
            loss = clss_loss + self.reg_alpha * kd_loss
        else:
            loss = clss_loss

        return (loss, outputs) if return_outputs else loss


class Trainer(basic.Trainer):
    """Plato Trainer for MlFD using HuggingFace transformer models."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model)

        self.unique_labels = len(Config().data.label_list.split(", "))
        self.logit_tracker = LogitTracker(self.unique_labels)

        self.hf_trainer = None
        self.trainer_callbacks = []
        if callbacks:
            self.add_callbacks(callbacks)

        self.model.train()

        parser = HfArgumentParser(TrainingArguments)
        (self.training_args,) = parser.parse_args_into_dataclasses(
            args=["--output_dir=/tmp", "--report_to=none"]
        )

        model_name = DataSource._resolve_model_name(Config().trainer.model_name)
        self.max_seq_len = Config().trainer.max_seq_len

        config_kwargs = {
            "cache_dir": Config().params["model_path"],
            "revision": "main",
            "use_auth_token": None,
            "max_length": self.max_seq_len,
        }
        tokenizer_kwargs = {
            "cache_dir": Config().params["data_path"],
            "use_fast": True,
            "revision": "main",
            "use_auth_token": None,
            "max_length": self.max_seq_len,
        }

        self.hf_config = AutoConfig.from_pretrained(model_name, **config_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, config=self.hf_config, **tokenizer_kwargs
        )
        self.metric = evaluate.load(Config().data.dataset_config)

        self.training_args.label_smoothing_factor = 0.1
        self.training_args._n_gpu = (
            Config().trainer._n_gpu
            if hasattr(Config().trainer, "_n_gpu")
            else torch.cuda.device_count()
        )
        self.training_args.disable_tqdm = True
        self.training_args.fp16 = True
        self.training_args.bf16 = False
        self.training_args.optim = "adamw_torch"

    def compute_metric(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return self.metric.compute(predictions=preds, references=p.label_ids)

    def train_model(self, config, trainset, sampler, **kwargs):
        """Run one round of local training.

        Args:
            config: Dict with training hyperparameters (epochs, batch_size).
            trainset: The local training dataset (HuggingFace Dataset).
            sampler: Plato sampler that defines the data partition.
        """
        self.training_args.num_train_epochs = config["epochs"]
        self.training_args.per_device_train_batch_size = config["batch_size"]

        self.hf_trainer = SampledHuggingFaceTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=trainset,
            eval_dataset=None,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            sampler=sampler,
            callbacks=self.trainer_callbacks,
            compute_metrics=self.compute_metric,
            logit_tracker=self.logit_tracker,
        )
        self.hf_trainer.logit_tracker = self.logit_tracker

        if (
            self.client_id != 0
            and hasattr(Config().clients, "speed_simulation")
            and Config().clients.speed_simulation
        ):
            self.simulate_sleep_time()

        self.hf_trainer.train()
        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)

    def test_model(self, config, testset, sampler=None, **kwargs):
        """Evaluate the model on the local test set.

        Returns:
            Accuracy (float between 0 and 1).
        """
        self.training_args.per_device_eval_batch_size = config["batch_size"]

        self.hf_trainer = SampledHuggingFaceTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=testset,
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            sampler=sampler,
            callbacks=None,
            compute_metrics=self.compute_metric,
            logit_tracker=self.logit_tracker,
        )

        metrics = self.hf_trainer.evaluate()
        return metrics["eval_accuracy"]

    def add_callbacks(self, callbacks):
        """Register HuggingFace-compatible trainer callbacks."""
        for cb in callbacks:
            if not issubclass(cb, TrainerCallback):
                raise ValueError(
                    f"Expected subclass of {TrainerCallback}, got {cb}."
                )
        self.trainer_callbacks.extend(callbacks)


from mlfd.data import DataSource
