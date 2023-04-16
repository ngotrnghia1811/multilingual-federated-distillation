"""
MlFD Client — FL client with personalized embedding upload and KD logit reception.

Each round the client:
  1. Receives server payload: aggregated embeddings (or full weights) + sum_logits.
  2. Loads the aggregated embeddings into its local model.
  3. Converts sum_logits into per-client global KD targets via LogitTracker.server_avg().
  4. Trains locally for one epoch using the KD-augmented loss.
  5. Extracts and returns its personalized embeddings + logit tracker to the server.
"""

import logging
import time
from types import SimpleNamespace

from plato.clients import simple
from plato.config import Config
from plato.utils import fonts


class Client(simple.Client):
    """MlFD federated learning client."""

    def configure(self) -> None:
        """Initialize client-side flags for global model vs. embedding mode."""
        super().configure()

        self.global_model = (
            hasattr(Config().server, "global_model") and Config().server.global_model
        )
        self.global_embedding = (
            hasattr(Config().server, "global_embedding")
            and Config().server.global_embedding
        )
        assert not (self.global_model and self.global_embedding), (
            "global_model and global_embedding are mutually exclusive."
        )

    async def _train(self):
        """Local training workload for one FL round."""
        logging.info(
            fonts.colourize(
                f"[{self}] Started training in communication round #{self.current_round}."
            )
        )

        try:
            if hasattr(self.trainer, "current_round"):
                self.trainer.current_round = self.current_round
            training_time = self.trainer.train(self.trainset, self.sampler)
        except ValueError as exc:
            logging.info(
                fonts.colourize(f"[{self}] Training error: {exc}")
            )
            await self.sio.disconnect()
            return

        if self.global_model:
            payload = self.algorithm.extract_weights()
        elif self.global_embedding:
            payload = self.algorithm.extract_embeddings(
                token_ids=self.datasource.token_ids
            )
        else:
            payload = None

        if (
            hasattr(Config().clients, "do_test") and Config().clients.do_test
        ) and (
            not hasattr(Config().clients, "test_interval")
            or self.current_round % Config().clients.test_interval == 0
        ):
            accuracy = self.trainer.test(self.testset, self.testset_sampler)
            if accuracy == -1:
                await self.sio.disconnect()
                return
        else:
            accuracy = 0

        comm_time = time.time()

        report = SimpleNamespace(
            client_id=self.client_id,
            num_samples=self.sampler.num_samples(),
            accuracy=accuracy,
            training_time=training_time,
            comm_time=comm_time,
            update_response=False,
            logit_tracker=self.trainer.logit_tracker,
            token_ids=self.datasource.token_ids,
        )

        self._report = self.customize_report(report)
        return self._report, payload

    def _load_payload(self, server_payload) -> None:
        """Load the server payload: global embeddings and KD logits.

        Args:
            server_payload: Dict with optional keys:
                - 'sum_logits': server-aggregated per-label logit sum tensor.
                - 'agg_weights': full model weights (global_model mode).
                - 'agg_embeddings': aggregated embedding rows (global_embedding mode).
                - 'group_token_ids': shared token IDs of the tier.
        """
        sum_logits = server_payload.get("sum_logits")

        if sum_logits is not None:
            self.trainer.logit_tracker.server_avg(sum_logits)
            self.trainer.logit_tracker.skip_kd = False
        else:
            self.trainer.logit_tracker.skip_kd = True

        if len(server_payload) > 1:
            if self.global_model:
                self.algorithm.load_weights(server_payload["agg_weights"])
            elif self.global_embedding:
                agg_embeddings = server_payload["agg_embeddings"]
                group_token_ids = server_payload["group_token_ids"]["aggr"]
                self.algorithm.load_embeddings(agg_embeddings, group_token_ids)

    async def _obtain_model_update(self, client_id, requested_time):
        """Return a model update for wall-clock time simulation (urgent request)."""
        if self.global_model:
            model = self.trainer.obtain_model_update(client_id, requested_time)
            weights = self.algorithm.extract_weights(model)
            self._report.comm_time = time.time()
            self._report.client_id = client_id
            self._report.update_response = True
            return self._report, weights

        elif self.global_embedding:
            model = self.trainer.obtain_model_update(client_id, requested_time)
            embeddings = self.algorithm.extract_embeddings(
                model=model, token_ids=self.datasource.token_ids
            )
            self._report.comm_time = time.time()
            self._report.client_id = client_id
            self._report.update_response = True
            return self._report, embeddings

        return self._report, None
