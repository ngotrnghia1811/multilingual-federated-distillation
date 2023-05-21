"""
MlFD training entry point.

Usage:
    # MlFD Base setting (6 clients / 3 groups / XLMR)
    python train.py --config configs/mlfd_base.yaml

    # MlFD Mini setting (12 clients / 6 groups / mMiniLM)
    python train.py --config configs/mlfd_mini.yaml

    # FedAT baseline
    python train.py --config configs/fedat_base.yaml --method fedat

    # FedDistill baseline
    python train.py --config configs/feddistill_base.yaml --method feddistill

    # FedAsync baseline
    python train.py --config configs/fedasync_base.yaml --method fedasync
"""

import argparse
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def parse_args():
    parser = argparse.ArgumentParser(description="Run MlFD or baseline FL experiments.")
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--method",
        default="mlfd",
        choices=["mlfd", "fedat", "feddistill", "fedasync"],
        help="FL method to run (default: mlfd).",
    )
    return parser.parse_args()


def run_mlfd(config_path: str):
    """Launch the MlFD training session."""
    from plato.datasources import registry as ds_registry
    from mlfd.data import DataSource as MlFDDataSource

    ds_registry.registered_datasources["mlfd"] = MlFDDataSource

    from mlfd.algorithm import Algorithm
    from mlfd.client import Client
    from mlfd.server import Server
    from mlfd.trainer import Trainer

    trainer = Trainer
    algorithm = Algorithm
    client = Client(algorithm=algorithm, trainer=trainer)
    server = Server(algorithm=algorithm, trainer=trainer)
    server.run(client)


def run_fedat(config_path: str):
    """Launch the FedAT baseline session."""
    from plato.datasources import registry as ds_registry
    from mlfd.data import DataSource as MlFDDataSource

    ds_registry.registered_datasources["mlfd"] = MlFDDataSource

    from mlfd.algorithm import Algorithm
    from mlfd.trainer import Trainer
    from plato.clients import simple
    from baselines.fedat import Server

    trainer = Trainer
    algorithm = Algorithm
    client = simple.Client(algorithm=algorithm, trainer=trainer)
    server = Server(algorithm=algorithm, trainer=trainer)
    server.run(client)


def run_feddistill(config_path: str):
    """Launch the FedDistill baseline session."""
    from plato.datasources import registry as ds_registry
    from mlfd.data import DataSource as MlFDDataSource

    ds_registry.registered_datasources["mlfd"] = MlFDDataSource

    from mlfd.algorithm import Algorithm
    from mlfd.trainer import Trainer
    from mlfd.client import Client
    from baselines.feddistill import Server

    trainer = Trainer
    algorithm = Algorithm
    client = Client(algorithm=algorithm, trainer=trainer)
    server = Server(algorithm=algorithm, trainer=trainer)
    server.run(client)


def run_fedasync(config_path: str):
    """Launch the FedAsync baseline session."""
    from plato.datasources import registry as ds_registry
    from mlfd.data import DataSource as MlFDDataSource

    ds_registry.registered_datasources["mlfd"] = MlFDDataSource

    from mlfd.algorithm import Algorithm
    from mlfd.trainer import Trainer
    from plato.clients import simple
    from baselines.fedasync import Server

    trainer = Trainer
    algorithm = Algorithm
    client = simple.Client(algorithm=algorithm, trainer=trainer)
    server = Server(algorithm=algorithm, trainer=trainer)
    server.run(client)


RUNNERS = {
    "mlfd": run_mlfd,
    "fedat": run_fedat,
    "feddistill": run_feddistill,
    "fedasync": run_fedasync,
}


if __name__ == "__main__":
    args = parse_args()

    os.environ["config_file"] = args.config

    runner = RUNNERS[args.method]
    runner(args.config)
