"""
MlFD Algorithm — Personalized embedding extraction and loading.

Extends the base Plato algorithm with two key operations:
  - extract_embeddings: returns only the word embedding rows corresponding to
    the client's personalized token IDs (a subset of the full MLLM vocabulary).
  - load_embeddings: writes a set of (possibly personalized) embedding rows back
    into the full embedding matrix, then loads the result into the model.
  - _personalized_to_full: helper to scatter personalized rows into the full matrix.
"""

import logging
import torch
from plato.algorithms import base


class Algorithm(base.Algorithm):
    """Federated learning algorithm for MlFD with personalized embedding support."""

    def extract_weights(self, model=None):
        """Return the full state dict of the model (used in MlFD_full ablation)."""
        target = model if model is not None else self.model
        return target.cpu().state_dict()

    def load_weights(self, weights):
        """Load a full state dict into the model."""
        self.model.load_state_dict(weights, strict=True)

    def extract_embeddings(self, model=None, token_ids=None):
        """Extract the word embedding matrix, optionally restricting to `token_ids`.

        Args:
            model: Optional; if None, uses self.model.
            token_ids: Optional list of vocabulary indices. If provided, only the
                rows at those indices are returned (personalized embedding).

        Returns:
            A dict mapping the word-embedding parameter name to its tensor.
        """
        weights = self.extract_weights(model)
        emb_key, emb_tensor = next(
            (k, v) for k, v in weights.items() if "word_embedding" in k
        )

        if token_ids is not None:
            personalized = {emb_key: emb_tensor[token_ids]}
        else:
            personalized = {emb_key: emb_tensor}

        return personalized

    def load_embeddings(self, embeddings, token_ids=None):
        """Load embeddings back into the model.

        If `token_ids` is provided, the incoming tensor contains only the rows
        corresponding to those IDs and must be scattered into the full matrix.
        """
        if token_ids is not None:
            full_embeddings = self._personalized_to_full(embeddings, token_ids)
        else:
            full_embeddings = embeddings

        self.model.load_state_dict(full_embeddings, strict=False)

    def _personalized_to_full(self, p_embeddings, token_ids):
        """Scatter personalized embedding rows back into the full embedding matrix.

        Args:
            p_embeddings: Dict mapping emb_key -> tensor of shape (len(token_ids), dim).
            token_ids: List of global vocabulary indices that correspond to rows
                in p_embeddings.

        Returns:
            Dict mapping emb_key -> full embedding tensor with updated rows.
        """
        f_embeddings = self.extract_embeddings()

        for emb_key, p_tensor in p_embeddings.items():
            f_tensor = f_embeddings[emb_key]
            for local_idx, global_idx in enumerate(token_ids):
                try:
                    f_tensor[global_idx] = p_tensor[local_idx]
                except IndexError:
                    logging.warning(
                        "Embedding index out of range: global=%d, local=%d, "
                        "p_size=%s, f_size=%s",
                        global_idx,
                        local_idx,
                        tuple(p_tensor.shape),
                        tuple(f_tensor.shape),
                    )
                    raise
            f_embeddings[emb_key] = f_tensor

        return f_embeddings
