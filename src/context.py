"""Contains the settings class that gathers all the run-specific information."""

import argparse

from datetime import datetime
import hashlib
import torch
import wandb
import yaml


def load_config(config_file: str) -> dict:
    """Load config from file."""
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    raise IOError("Config file Could not be read.")


class Context:
    """Contains all the usefull informations that can change between runs."""

    def __init__(
        self, config: dict, args: argparse.Namespace, checkpoint: bool = False
    ):
        # Attributes that can't be given through configuration
        self.id: str = config["id"] if checkpoint else None
        self.nb_interactions: int = config["nb_interactions"] if checkpoint else None
        self.nb_users: int = None
        self.nb_items: int = None
        self.user_features: list[str] = []
        self.item_features: list[str] = []
        self.data: tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer
        self.run: wandb.sdk.wandb_run.Run
        self.epoch: int = 0

        forbidden_settings = dir(self)

        # Attributes with default value
        self.requested_features: set[str] = set()

        self.train_heat_up_length: int = 0
        self.test_heat_up_length: int = 0
        self.evaluate_every: int = 1
        self.checkpoint_every: int = 1
        self.nb_checkpoint_to_keep: int = 3

        # Attributes without default value
        self.name: str
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

    def from_config(config_file: str, args: argparse.Namespace):
        """Create a context from a config file."""
        with open(config_file, "r", encoding="utf-8") as f:
            return Context(yaml.load(f, Loader=yaml.FullLoader), args)
        raise IOError("Config file Could not be read.")

    def from_checkpoint(checkpoint_file: str, args: argparse.Namespace):
        """Create a context from a checkpoint file."""
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        context = Context(checkpoint["context"], args, checkpoint=True)
        context.epoch = checkpoint["context"].get("epoch", 0)
        return context

    def __str__(self) -> str:
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])

    def get_id(self) -> str:
        """Returns the run's id."""
        if self.id is not None:
            return self.id
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sha = (
            hashlib.shake_256(self.__dict__.__str__().encode()).digest(1).hex()
        )  # create small hashcode
        self.id = f"{self.name}__{date_time}__{sha}"
        return self.id

    def to_save(self) -> object:
        """Returns the data that needs to be saved for checkpointing."""
        content = self.__dict__.copy()
        content.pop("data")
        content.pop("model")
        content.pop("optimizer")
        content.pop("device")
        content.pop("run", None)
        return content
