# Data

## XNLI (Cross-lingual Natural Language Inference)

MlFD experiments use the **XNLI** task from the [XGLUE benchmark](https://arxiv.org/abs/2004.01401).

### Dataset Description

XNLI is a natural language inference (NLI) task: given a premise and hypothesis sentence pair, predict the relationship as one of three classes: **contradiction**, **neutral**, or **entailment**.

The dataset covers 15 languages: `ar, bg, de, el, es, fr, hi, ru, sw, th, tr, ur, vi, zh, en`.

### Federated Partitioning

To simulate a multi-lingual FL setting:
- The **per-language validation splits** serve as each language group's local training data.
- The **per-language test splits** are used for local evaluation.
- Each client stores data from a single language, randomly sampled from the corresponding split (`partition_size: 1000` examples per client by default).

### Automatic Download

The data is downloaded automatically via HuggingFace `datasets` when the experiment is first run:

```python
from datasets import load_dataset
dataset = load_dataset("xglue", "xnli")
```

No manual download is required. The dataset is cached to the path specified by `data_path` in the config (default: `~/.cache/huggingface/datasets/`).

### License

XNLI is released under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).
