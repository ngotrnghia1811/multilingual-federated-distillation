"""
FedAsync (Fully Asynchronous Federated Learning) baseline.

The server performs global model updates immediately upon receiving any single
client update, without waiting for other clients. This eliminates stragglers
entirely but can suffer from staleness if slow clients dominate.

Reference: Xie et al., "Asynchronous Federated Optimization." (2019)
"""

import logging

from plato.config import Config
from plato.servers import fedavg
from plato.utils import fonts


class Server(fedavg.Server):
    """FedAsync server: immediate single-client aggregation with staleness weighting."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

    async def aggregate_weights(self, weights_received):
        """Weighted average of a single client update with the current global model.

        In the fully asynchronous setting, `weights_received` typically contains
        one client update. The server blends it with the current global model
        using a mixing coefficient alpha = 1 / (staleness + 1).
        """
        updates = self.updates
        total_samples = sum(u.report.num_samples for u in updates)

        avg = {k: self.trainer.zeros(v.shape) for k, v in weights_received[0].items()}
        for i, w in enumerate(weights_received):
            n = updates[i].report.num_samples
            for k, v in w.items():
                avg[k] += v * n / total_samples
        return avg

    async def _process_reports(self):
        weights_received = [u.payload for u in self.updates]
        weights_received = self.weights_received(weights_received)
        self.callback_handler.call_event("on_weights_received", self, weights_received)

        updated_weights = await self.aggregate_weights(weights_received)
        self.algorithm.load_weights(updated_weights)
        self.weights_aggregated(self.updates)

        if hasattr(Config().server, "do_test") and not Config().server.do_test:
            self.accuracy = self.accuracy_averaging(self.updates)
        else:
            self.accuracy = self.trainer.test(self.testset, self.testset_sampler)

        logging.info(
            fonts.colourize(f"[{self}] Global model accuracy: {100 * self.accuracy:.2f}%\n")
        )
        self.clients_processed()
        self.callback_handler.call_event("on_clients_processed", self)
