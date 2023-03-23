"""
MlFD Data Source — XNLI from XGLUE with personalized tokenization.

Each client is assigned a language tier based on its client_id. The local
training set is the per-language validation split; the test set is the
per-language test split. After tokenization, the set of unique input_ids
across train+test is stored as `self.token_ids`, which represents the client's
personalized vocabulary subset.
"""

import logging
import os

from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers import testing_utils, utils

from plato.config import Config
from plato.datasources import base


class DataSource(base.DataSource):
    """XNLI data source with per-client language assignment and personalized token IDs."""

    def __init__(self, **kwargs):
        super().__init__()

        self.client_id = kwargs["client_id"]

        dataset_name = Config().data.dataset_name
        dataset_config = (
            Config().data.dataset_config
            if hasattr(Config.data, "dataset_config")
            else None
        )
        self.dataset_config = dataset_config

        saved_data_path = (
            f"{Config().params['data_path']}/{dataset_name}_{dataset_config}"
        )

        if os.path.exists(saved_data_path):
            self.dataset = load_from_disk(saved_data_path)
        else:
            self.dataset = load_dataset(dataset_name, dataset_config)
            self.dataset.save_to_disk(saved_data_path)

        model_name = self._resolve_model_name(Config().trainer.model_name)
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

        self.config = AutoConfig.from_pretrained(model_name, **config_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, config=self.config, **tokenizer_kwargs
        )
        self.tok_logger = utils.logging.get_logger(
            "transformers.tokenization_utils_base"
        )

        parser = HfArgumentParser(TrainingArguments)
        (self.training_args,) = parser.parse_args_into_dataclasses(
            args=["--output_dir=/tmp", "--report_to=none"]
        )

        if dataset_config == "xnli":
            self.column_names = ["premise", "hypothesis", "label"]
        else:
            self.column_names = ["text"]
            self.text_column_name = "text"

        self.lang_list = Config().data.lang_list.split(", ")
        self.total_clients = Config().clients.total_clients
        self.num_groups = (
            Config().server.num_groups
            if hasattr(Config().server, "num_groups")
            else self.total_clients
        )
        self.group_size = self.total_clients // self.num_groups

        self.lang_id = (self.client_id - 1) // self.group_size
        self.lang = self.lang_list[self.lang_id]

        self.trainset = self.preprocess_data(self.dataset[f"validation.{self.lang}"])
        self.testset = self.preprocess_data(self.dataset[f"test.{self.lang}"])

        train_ids = sum([item["input_ids"] for item in self.trainset], [])
        test_ids = sum([item["input_ids"] for item in self.testset], [])
        self.token_ids = list(sorted(set(train_ids + test_ids)))

        logging.info(
            "[Client #%s] Language: %s | Vocab size: %d",
            self.client_id,
            self.lang,
            len(self.token_ids),
        )

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        """Resolve HuggingFace model hub prefix from short model name."""
        if "l12" in model_name.lower():
            return "microsoft/" + model_name
        if "l6" in model_name.lower():
            return "MoritzLaurer/" + model_name
        return model_name

    def num_train_examples(self):
        return len(self.trainset)

    def num_test_examples(self):
        return len(self.testset)

    def get_train_set(self):
        return self.trainset

    def get_test_set(self):
        return self.testset

    @staticmethod
    def input_shape():
        raise NotImplementedError("Input shape is variable-length for transformer models.")

    def tokenize_function(self, examples):
        """Tokenize premise-hypothesis pairs (XNLI) or raw text."""
        with testing_utils.CaptureLogger(self.tok_logger) as cl:
            if self.dataset_config == "xnli":
                output = self.tokenizer(
                    examples[self.column_names[0]],
                    examples[self.column_names[1]],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_seq_len,
                )
            else:
                output = self.tokenizer(examples[self.text_column_name])

        if "Token indices sequence length is longer than the" in cl.out:
            self.tok_logger.warning(
                "Input is longer than max_length (%d); it will be truncated.",
                self.max_seq_len,
            )
        return output

    def preprocess_data(self, dataset):
        """Tokenize the raw dataset using batched map."""
        with self.training_args.main_process_first(desc="dataset map tokenization"):
            tokenized = dataset.map(
                self.tokenize_function,
                batched=True,
                num_proc=4,
                load_from_cache_file=True,
                desc="Running tokenizer on dataset",
            )
        return tokenized
