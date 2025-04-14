"""Contains the settings class that gathers all the run-specific information."""

import argparse
import torch
import wandb


class Context:
    """Contains all the usefull informations that can change between runs."""

    def __init__(self, config: dict, args: argparse.Namespace):
        # Attributes that can't be given through configuration
        self.nb_interactions: int = None
        self.nb_users: int = None
        self.nb_items: int = None
        self.user_features: list[str] = []
        self.item_features: list[str] = []
        self.data: torch.utils.data.DataLoader
        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer
        self.run: wandb.sdk.wandb_run.Run

        forbidden_settings = dir(self)

        # Attributes with default value
        self.requested_features: set[str] = set()

        self.train_heat_up_length: int = 0
        self.test_heat_up_length: int = 0
        self.evaluate_every: int = 1

        # Attributes without default value
        self.dataset: str
        self.user_id_column: str
        self.item_id_column: str
        self.timestamp_column: str

        self.epochs: int
        self.train_ratio: int
        self.train_sequence_length: int
        self.train_sequence_stride: int
        self.train_batch_size: int
        self.test_sequence_length: int
        self.test_sequence_stride: int
        self.test_batch_size: int
        self.device: torch.DeviceObjType

        self.model_name: str
        self.model_attributes: dict[str:object]
        self.loss: dict[str:int]
        self.learning_rate: float
        self.l2: float
        self.metrics: list[str]

        # Context initialization
        for key, value in config.items():
            if key in forbidden_settings:
                continue
            setattr(self, key, value)
        self.device = "cpu" if args.gpu is None else f"cuda:{args.gpu}"

    def __str__(self) -> str:
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])
