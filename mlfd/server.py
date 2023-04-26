"""
MlFD Server — Semi-asynchronous tier-based federated learning server.

The server implements two levels of aggregation:
  1. Intra-tier (synchronous FedAvg): within a language group (tier), client
     embeddings are weighted-averaged by sample count.
  2. Cross-tier (asynchronous FedAsync): once any tier has enough updates, the
     server updates the global embedding as a weighted average across all tiers,
     weighted by each tier's cumulative aggregation count.

Additionally, the server aggregates per-label logit vectors (LogitTracker.avg())
from clients within the fastest ready tier, and distributes the logit sum back to
the next selected clients as a KD teaching signal.

Key config parameters (server section):
  - num_groups: number of language tiers (one per language in the experiment)
  - global_embedding: if true, exchange only word embeddings (MlFD_emb mode)
  - global_model: if true, exchange full model weights (MlFD_full ablation)
  - minimum_clients_aggregated: minimum clients per tier before aggregation fires
"""

import asyncio
import logging
import os
import pickle
import random
import sys
from types import SimpleNamespace

import numpy as np

from plato.config import Config
from plato.servers import fedavg
from plato.utils import fonts


class Server(fedavg.Server):
    """MlFD federated learning server."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
    ):
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
        self.group_logits = {}
        self.agg_counter = {}
        self.client_to_group_id = {}

        self.current_aggregated_group_id = None
        self.current_aggregated_updates = None
        self.current_group_token_ids = {}
        self.sum_logits = None
        self.selected_clients = []

    def configure(self) -> None:
        """Initialize per-group model stores and aggregation counters."""
        super().configure()

        self.num_groups = Config().server.num_groups
        self.global_model = (
            Config().server.global_model
            if hasattr(Config().server, "global_model")
            else False
        )
        self.global_embedding = (
            Config().server.global_embedding
            if hasattr(Config().server, "global_embedding")
            else False
        )
        assert not (self.global_model and self.global_embedding), (
            "global_model and global_embedding are mutually exclusive."
        )

        for i in range(self.num_groups):
            if self.global_model:
                self.group_models[i] = self.algorithm.extract_weights()
            elif self.global_embedding:
                self.group_models[i] = self.algorithm.extract_embeddings()
            self.group_logits[i] = None
            self.agg_counter[i] = 0

    async def register_client(self, sid, client_process_id, client_id):
        """Register a new client and assign it to its language tier."""
        self.clients[client_process_id] = {"sid": sid, "client_id": client_id}

        group_id = (client_id - 1) // self.group_size
        self.clients[client_process_id]["group_id"] = group_id
        self.client_to_group_id[client_id] = group_id

        logging.info(
            "[%s] Client #%d assigned to group (language tier) #%d.",
            self,
            client_id,
            group_id,
        )

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

    async def _select_clients(self, for_next_batch=False):
        """Select clients for the next training round.

        On the initial round or after a cross-tier aggregation, selects one client
        per group (language). After intra-tier aggregation, re-selects only from
        the aggregated tier.
        """
        group_id = self.current_aggregated_group_id

        if not for_next_batch:
            if group_id is None:
                self.current_aggregated_group_id = None
                self.current_aggregated_updates = None
                self.current_group_token_ids = {}
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

            self.current_round += 1
            self.round_start_wall_time = self.wall_time
            logging.info(
                fonts.colourize(
                    f"\n[{self}] Starting round {self.current_round}/{Config().trainer.rounds}."
                )
            )

            if Config().is_central_server():
                self.clients_pool = list(self.clients)
            elif not Config().is_edge_server():
                self.clients_pool = list(range(1, 1 + self.total_clients))
                if group_id is not None:
                    selectable_clients = [
                        c for c in self.clients_pool
                        if (c - 1) // self.group_size == group_id
                    ]
                else:
                    selectable_clients = self.clients_pool

            if (
                self.asynchronous_mode
                and len(self.selected_clients) > 0
                and len(self.reported_clients) > 0
                and len(self.reported_clients) < self.clients_per_round
            ):
                training_ids = [
                    self.training_clients[c]["id"] for c in self.training_clients
                ]
                reporting_ids = [c[2]["client_id"] for c in self.reported_clients]
                selectable_clients = [
                    c for c in selectable_clients
                    if c not in training_ids and c not in reporting_ids
                ]

                n_select = (
                    len(self.current_processed_clients)
                    if self.simulate_wall_time
                    else len(self.reported_clients)
                )
                self.selected_clients = self.choose_clients(
                    selectable_clients, n_select, group_id=group_id
                )
            else:
                n_select = (
                    min(self.group_size, self.clients_per_round)
                    if group_id is not None
                    else self.clients_per_round
                )
                self.selected_clients = self.choose_clients(
                    selectable_clients, n_select, group_id=group_id
                )

            self.current_reported_clients = {}
            self.current_processed_clients = {}
            if not self.simulate_wall_time:
                self.reported_clients = []

        if len(self.selected_clients) > 0:
            self.selected_sids = []

            if (
                hasattr(Config().trainer, "max_concurrency")
                and not Config().is_central_server()
            ):
                selected_clients = self._batch_select_clients()
            else:
                selected_clients = self.selected_clients

            for client_id in selected_clients:
                self.selected_client_id = client_id

                if Config().is_central_server():
                    client_process_id = client_id
                else:
                    for pid in self.clients:
                        sid = self.clients[pid]["sid"]
                        if sid not in self.training_sids and sid not in self.selected_sids:
                            client_process_id = pid
                            break

                sid = self.clients[client_process_id]["sid"]
                self.training_sids.append(sid)
                self.selected_sids.append(sid)
                self.clients[client_process_id]["client_id"] = client_id
                self.training_clients[client_id] = {
                    "id": client_id,
                    "starting_round": self.current_round,
                    "start_time": self.round_start_wall_time,
                    "update_requested": False,
                }

                logging.info("[%s] Selecting client #%d for training.", self, client_id)

                server_response = {
                    "id": client_id,
                    "current_round": self.current_round,
                }
                server_response = self.customize_server_response(
                    server_response, client_id=client_id
                )

                payload = {"sum_logits": self.sum_logits}
                if self.current_aggregated_group_id is not None:
                    if self.global_model:
                        payload["agg_weights"] = self.algorithm.extract_weights()
                    elif self.global_embedding:
                        payload["agg_embeddings"] = self.algorithm.extract_embeddings(
                            token_ids=self.current_group_token_ids.get("aggr")
                        )
                        payload["group_token_ids"] = self.current_group_token_ids
                payload = self.customize_server_payload(payload)

                if self.comm_simulation:
                    payload = self.outbound_processor.process(payload)
                    model_name = getattr(Config().trainer, "model_name", "custom")
                    checkpoint_path = Config().params["checkpoint_path"]
                    payload_filename = f"{checkpoint_path}/{model_name}_{client_id}.pth"

                    with open(payload_filename, "wb") as pf:
                        pickle.dump(payload, pf)

                    server_response["payload_filename"] = payload_filename
                    payload_size = sys.getsizeof(pickle.dumps(payload)) / 1024**2
                    logging.info(
                        "[%s] Payload to client #%d: %.2f MB (simulated).",
                        self, client_id, payload_size,
                    )
                    self.comm_overhead += payload_size
                    self.downlink_comm_time[client_id] = payload_size / (
                        (self.downlink_bandwidth / 8) / len(self.selected_clients)
                    )

                await self.sio.emit(
                    "payload_to_arrive", {"response": server_response}, room=sid
                )
                if not self.comm_simulation:
                    await self._send(sid, payload, client_id)

            self.clients_selected(self.selected_clients)
            self.callback_handler.call_event("on_clients_selected", self, self.selected_clients)

    def _batch_select_clients(self):
        """Select a batch of clients respecting max_concurrency and GPU count."""
        if Config().gpu_count() > 1:
            selected = []
            untrained = list(set(self.selected_clients).difference(self.trained_clients))
            for cuda_id in range(Config().gpu_count()):
                for c in untrained:
                    if c % Config().gpu_count() == cuda_id:
                        selected.append(c)
                    if len(selected) >= min(
                        len(self.clients),
                        (cuda_id + 1) * Config().trainer.max_concurrency,
                        self.clients_per_round,
                    ):
                        break
                if len(selected) >= len(self.clients):
                    break
        else:
            start = len(self.trained_clients)
            end = min(start + len(self.clients), len(self.selected_clients))
            selected = self.selected_clients[start:end]

        self.trained_clients += selected
        return selected

    def _choose_clients(self, clients_pool, clients_count):
        """Randomly sample `clients_count` clients from `clients_pool`."""
        assert clients_count <= len(clients_pool)
        random.setstate(self.prng_state)
        selected = random.sample(clients_pool, clients_count)
        self.prng_state = random.getstate()
        return selected

    def choose_clients(self, clients_pool, clients_count, group_id=None):
        """Select clients. If group_id is None, sample equally from all tiers."""
        if group_id is not None:
            return self._choose_clients(clients_pool, clients_count)

        selected = []
        clients_per_group = self.group_size
        for g_id in range(self.num_groups):
            group_pool = [c for c in clients_pool if (c - 1) // self.group_size == g_id]
            group_selected = self._choose_clients(group_pool, clients_per_group)
            selected += group_selected
            if len(selected) == clients_count:
                break
        return selected

    async def _periodic_task(self):
        """Periodic check: if a tier has enough updates, fire aggregation."""
        await self.periodic_task()

        if self.asynchronous_mode and not self.simulate_wall_time:
            for __, client_data in self.training_clients.items():
                staleness = self.current_round - client_data["starting_round"]
                if staleness > self.staleness_bound:
                    logging.info(
                        "[%s] Client %s staleness %d > bound %d; skipping aggregation.",
                        self, client_data["id"], staleness, self.staleness_bound,
                    )
                    return

            await self._find_and_aggregate_group()

    async def _find_and_aggregate_group(self):
        """Find the tier with the most updates that meets the minimum threshold,
        then trigger aggregation and a new round for that tier."""
        ready_groups = []
        for g_id in range(self.num_groups):
            group_updates = [
                (t, u) for t, u in enumerate(self.updates) if u.group_id == g_id
            ]
            arrival_sum = sum(t for t, _ in group_updates)
            updates = [u for _, u in group_updates]
            if len(updates) >= self.minimum_clients:
                ready_groups.append((arrival_sum, updates, g_id))

        if not ready_groups:
            logging.info(
                "[%s] No tier has reached the minimum of %d clients. Waiting.",
                self, self.minimum_clients,
            )
            return

        fastest = sorted(ready_groups, key=lambda x: x[0])[0]
        self.current_aggregated_group_id = fastest[2]
        self.current_aggregated_updates = fastest[1]

        token_ids_union = list(set(sum(
            [u.report.token_ids for u in fastest[1]], []
        )))
        self.current_group_token_ids = {"aggr": token_ids_union}
        for u in fastest[1]:
            self.current_group_token_ids[u.report.client_id] = u.report.token_ids

        logging.info(
            "[%s] Tier %d ready: %d clients. Aggregating.",
            self, fastest[2], len(fastest[1]),
        )
        await self._process_reports()
        await self.wrap_up()
        await self._select_clients()

    async def sync_aggregate_weights(self, weights_received):
        """Intra-tier FedAvg: weighted average of client weights by sample count."""
        updates = self.current_aggregated_updates
        total_samples = sum(u.report.num_samples for u in updates)

        if self.global_embedding:
            expanded = []
            for u, w in zip(updates, weights_received):
                expanded.append(
                    self.algorithm._personalized_to_full(
                        w, self.current_group_token_ids[u.report.client_id]
                    )
                )
            weights_received = expanded

        avg = {k: self.trainer.zeros(v.shape) for k, v in weights_received[0].items()}
        for i, w in enumerate(weights_received):
            n = updates[i].report.num_samples
            for k, v in w.items():
                avg[k] += v * n / total_samples
            await asyncio.sleep(0)

        return avg

    async def async_aggregate_weights(self, tier_weights):
        """Cross-tier FedAsync: update global model as weighted average across tiers.

        Weights are proportional to each tier's cumulative aggregation count,
        giving more influence to tiers that have reported more frequently.
        """
        group_id = self.current_aggregated_group_id
        self.agg_counter[group_id] += 1
        self.group_models[group_id] = tier_weights

        total_agg = sum(self.agg_counter.values())
        global_weights = {k: self.trainer.zeros(v.shape) for k, v in tier_weights.items()}

        for g_id in range(self.num_groups):
            weight = self.agg_counter[self.num_groups - 1 - g_id] / total_agg
            for k, v in self.group_models[g_id].items():
                global_weights[k] += weight * v

        return global_weights

    async def aggregate_logits(self):
        """Sum per-label logit averages across all clients in the current tier."""
        updates = self.current_aggregated_updates
        n_clients = len(updates)
        logit_sum = 0.0

        for update in updates:
            lt = update.report.logit_tracker
            lt.num_agg_clients = n_clients
            logit_sum += lt.avg()
            await asyncio.sleep(0)

        return logit_sum

    async def _process_reports(self):
        """Aggregate weights (or embeddings) and logits from the current tier."""
        if self.global_embedding or self.global_model:
            weights_received = [u.payload for u in self.current_aggregated_updates]
            weights_received = self.weights_received(weights_received)
            self.callback_handler.call_event("on_weights_received", self, weights_received)

            logging.info(
                "[Server #%d] Intra-tier aggregation for group %d.",
                os.getpid(), self.current_aggregated_group_id,
            )
            tier_weights = await self.sync_aggregate_weights(weights_received)

            logging.info(
                "[Server #%d] Cross-tier async aggregation from group %d.",
                os.getpid(), self.current_aggregated_group_id,
            )
            global_weights = await self.async_aggregate_weights(tier_weights)

            if self.global_model:
                self.algorithm.load_weights(global_weights)
            elif self.global_embedding:
                self.algorithm.load_embeddings(global_weights)
            self.weights_aggregated(self.current_aggregated_updates)

        self.sum_logits = await self.aggregate_logits()

        if hasattr(Config().server, "do_test") and not Config().server.do_test:
            self.accuracy = self.accuracy_averaging(self.updates)
            logging.info(
                "[%s] Average client accuracy: %.2f%%.", self, 100 * self.accuracy
            )
        else:
            logging.info("[%s] Started model testing.", self)
            self.accuracy = self.trainer.test(self.testset, self.testset_sampler)

        logging.info(
            fonts.colourize(f"[{self}] Global model accuracy: {100 * self.accuracy:.2f}%\n")
        )
        self.clients_processed()
        self.callback_handler.call_event("on_clients_processed", self)
