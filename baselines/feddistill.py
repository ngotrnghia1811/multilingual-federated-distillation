"""
FedDistill (Federated Distillation) baseline.

Clients communicate only per-label mean logit vectors instead of model weights,
eliminating the per-round payload cost of full model transfer. The server
aggregates per-label logits across clients, and each client uses the aggregated
global logits as KD targets in the next round.

Unlike MlFD, FedDistill performs no embedding aggregation and no tier-based
asynchronous training. It is a purely synchronous logit-exchange method.

Reference: Jeong et al., "Communication-Efficient On-Device Machine Learning:
Federated Distillation and Augmentation under Non-IID Private Data." (2018)
"""

import asyncio
import logging

from plato.config import Config
from plato.servers import fedavg
from plato.utils import fonts


class Server(fedavg.Server):
    """FedDistill server: synchronous logit aggregation only (no weight exchange)."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.sum_logits = None

    def configure(self) -> None:
        super().configure()
        self.global_model = (
            Config().server.global_model
            if hasattr(Config().server, "global_model")
            else False
        )
        self.sum_logits = None

    async def aggregate_weights(self, weights_received):
        """No weight aggregation in FedDistill — logits are the only payload."""
        updates = self.updates
        total_samples = sum(u.report.num_samples for u in updates)

        avg = {k: self.trainer.zeros(v.shape) for k, v in weights_received[0].items()}
        for i, w in enumerate(weights_received):
            n = updates[i].report.num_samples
            for k, v in w.items():
                avg[k] += v * n / total_samples
            await asyncio.sleep(0)
        return avg

    async def aggregate_logits(self):
        """Sum per-label average logits from all reporting clients."""
        n_clients = len(self.updates)
        logit_sum = 0.0
        for update in self.updates:
            lt = update.report.logit_tracker
            lt.num_agg_clients = n_clients
            logit_sum += lt.avg()
            await asyncio.sleep(0)
        return logit_sum

    async def _process_reports(self):
        """Aggregate logits; optionally also aggregate full model weights."""
        if self.global_model:
            weights_received = [u.payload for u in self.updates]
            weights_received = self.weights_received(weights_received)
            self.callback_handler.call_event("on_weights_received", self, weights_received)
            updated_weights = await self.aggregate_weights(weights_received)
            self.algorithm.load_weights(updated_weights)
            self.weights_aggregated(self.updates)

        self.sum_logits = await self.aggregate_logits()

        if hasattr(Config().server, "do_test") and not Config().server.do_test:
            self.accuracy = self.accuracy_averaging(self.updates)
        else:
            self.accuracy = self.trainer.test(self.testset, self.testset_sampler)

        logging.info(
            fonts.colourize(f"[{self}] Global model accuracy: {100 * self.accuracy:.2f}%\n")
        )
        self.clients_processed()
        self.callback_handler.call_event("on_clients_processed", self)


class Algorithm(object):
    """FedDistill algorithm: no weight extraction/loading (logit-only exchange)."""

    def __init__(self, trainer=None):
        self.trainer = trainer
        if trainer:
            self.model = trainer.model

    def extract_weights(self, model=None):
        return {}

    def load_weights(self, weights):
        pass


class Client(object):
    """FedDistill client stub — use Plato's simple.Client with logit reporting."""
    pass
