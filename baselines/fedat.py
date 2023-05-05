"""
FedAT (Federated learning with Asynchronous Tiers) baseline.

FedAT partitions clients into M tiers by response latency. It combines:
  - Synchronous intra-tier aggregation (FedAvg within a tier)
  - Asynchronous cross-tier aggregation (FedAsync across tiers)

Unlike MlFD, FedAT transmits the full model weights (not personalized embeddings).
Tier membership is based on the same nationality/language grouping as MlFD.

Reference: Chai et al., "TiFL: A Tier-based Federated Learning System." (2020)
"""

import asyncio
import logging
import os
import random

import numpy as np

from plato.config import Config
from plato.servers import fedavg
from plato.utils import fonts


class Server(fedavg.Server):
    """FedAT server: tiered semi-asynchronous federated averaging."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )
        self.num_groups = Config().server.num_groups
        self.group_size = self.total_clients // self.num_groups
        self.group_models = {}
        self.agg_counter = {}
        self.client_to_group_id = {}
        self.current_aggregated_group_id = None
        self.current_aggregated_updates = None
        self.selected_clients = []

    def configure(self) -> None:
        super().configure()
        self.num_groups = Config().server.num_groups
        init_weights = self.algorithm.extract_weights()
        for i in range(self.num_groups):
            self.group_models[i] = init_weights
            self.agg_counter[i] = 0

    async def register_client(self, sid, client_process_id, client_id):
        self.clients[client_process_id] = {"sid": sid, "client_id": client_id}
        group_id = (client_id - 1) // self.group_size
        self.clients[client_process_id]["group_id"] = group_id
        self.client_to_group_id[client_id] = group_id

        if hasattr(Config().trainer, "max_concurrency") and not Config().is_central_server():
            required = min(
                Config().trainer.max_concurrency * max(1, Config().gpu_count()),
                self.clients_per_round,
            )
        else:
            required = self.clients_per_round

        if (self.current_round == 0 or self.resumed_session) and len(self.clients) >= required:
            self.resumed_session = False
            self.training_will_start()
            self.callback_handler.call_event("on_training_will_start", self)
            await self._select_clients()

    async def sync_aggregate_weights(self, weights_received):
        """Intra-tier FedAvg."""
        updates = self.current_aggregated_updates
        total_samples = sum(u.report.num_samples for u in updates)

        avg = {k: self.trainer.zeros(v.shape) for k, v in weights_received[0].items()}
        for i, w in enumerate(weights_received):
            n = updates[i].report.num_samples
            for k, v in w.items():
                avg[k] += v * n / total_samples
            await asyncio.sleep(0)
        return avg

    async def async_aggregate_weights(self, tier_weights):
        """Cross-tier FedAsync weighted by cumulative aggregation count."""
        group_id = self.current_aggregated_group_id
        self.agg_counter[group_id] += 1
        self.group_models[group_id] = tier_weights

        total_agg = sum(self.agg_counter.values())
        global_weights = {k: self.trainer.zeros(v.shape) for k, v in tier_weights.items()}
        for g_id in range(self.num_groups):
            w = self.agg_counter[self.num_groups - 1 - g_id] / total_agg
            for k, v in self.group_models[g_id].items():
                global_weights[k] += w * v
        return global_weights

    async def _periodic_task(self):
        await self.periodic_task()
        if self.asynchronous_mode and not self.simulate_wall_time:
            for __, cdata in self.training_clients.items():
                if self.current_round - cdata["starting_round"] > self.staleness_bound:
                    return
            await self._find_and_aggregate_group()

    async def _find_and_aggregate_group(self):
        ready = []
        for g_id in range(self.num_groups):
            group_updates = [
                (t, u) for t, u in enumerate(self.updates) if u.group_id == g_id
            ]
            updates = [u for _, u in group_updates]
            t_sum = sum(t for t, _ in group_updates)
            if len(updates) >= self.minimum_clients:
                ready.append((t_sum, updates, g_id))

        if not ready:
            return

        fastest = sorted(ready, key=lambda x: x[0])[0]
        self.current_aggregated_group_id = fastest[2]
        self.current_aggregated_updates = fastest[1]

        await self._process_reports()
        await self.wrap_up()
        await self._select_clients()

    async def _process_reports(self):
        weights_received = [u.payload for u in self.current_aggregated_updates]
        weights_received = self.weights_received(weights_received)
        self.callback_handler.call_event("on_weights_received", self, weights_received)

        tier_weights = await self.sync_aggregate_weights(weights_received)
        global_weights = await self.async_aggregate_weights(tier_weights)
        self.algorithm.load_weights(global_weights)
        self.weights_aggregated(self.current_aggregated_updates)

        if hasattr(Config().server, "do_test") and not Config().server.do_test:
            self.accuracy = self.accuracy_averaging(self.updates)
        else:
            self.accuracy = self.trainer.test(self.testset, self.testset_sampler)

        logging.info(
            fonts.colourize(f"[{self}] Global model accuracy: {100 * self.accuracy:.2f}%\n")
        )
        self.clients_processed()
        self.callback_handler.call_event("on_clients_processed", self)

    async def _select_clients(self, for_next_batch=False):
        group_id = self.current_aggregated_group_id

        if not for_next_batch:
            if group_id is None:
                self.updates = []
                if hasattr(Config().trainer, "max_concurrency"):
                    self.trained_clients = []
            else:
                self.updates = [
                    u for u in self.updates
                    if (u.client_id - 1) // self.group_size != group_id
                ]
                if hasattr(Config().trainer, "max_concurrency"):
                    self.trained_clients = [
                        c for c in self.trained_clients
                        if (c - 1) // self.group_size != group_id
                    ]
                self.current_aggregated_group_id = None
                self.current_aggregated_updates = None

            self.current_round += 1
            self.round_start_wall_time = self.wall_time
            logging.info(
                fonts.colourize(
                    f"\n[{self}] Starting round {self.current_round}/{Config().trainer.rounds}."
                )
            )

            self.clients_pool = list(range(1, 1 + self.total_clients))
            if group_id is not None:
                selectable = [
                    c for c in self.clients_pool if (c - 1) // self.group_size == group_id
                ]
            else:
                selectable = self.clients_pool

            n_select = (
                min(self.group_size, self.clients_per_round)
                if group_id is not None
                else self.clients_per_round
            )
            self.selected_clients = self._choose_clients(selectable, n_select)
            self.current_reported_clients = {}
            self.current_processed_clients = {}
            if not self.simulate_wall_time:
                self.reported_clients = []

        for client_id in self.selected_clients:
            self.selected_client_id = client_id
            for pid in self.clients:
                sid = self.clients[pid]["sid"]
                if sid not in self.training_sids:
                    client_process_id = pid
                    break

            sid = self.clients[client_process_id]["sid"]
            self.training_sids.append(sid)
            self.clients[client_process_id]["client_id"] = client_id
            self.training_clients[client_id] = {
                "id": client_id,
                "starting_round": self.current_round,
                "start_time": self.round_start_wall_time,
                "update_requested": False,
            }

            server_response = {"id": client_id, "current_round": self.current_round}
            payload = self.algorithm.extract_weights()
            payload = self.customize_server_payload(payload)
            await self.sio.emit("payload_to_arrive", {"response": server_response}, room=sid)
            if not self.comm_simulation:
                await self._send(sid, payload, client_id)

        self.clients_selected(self.selected_clients)
        self.callback_handler.call_event("on_clients_selected", self, self.selected_clients)

    def _choose_clients(self, pool, count):
        assert count <= len(pool)
        random.setstate(self.prng_state)
        selected = random.sample(pool, count)
        self.prng_state = random.getstate()
        return selected
