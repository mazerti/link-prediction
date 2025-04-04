"""Contains the settings class that gathers all the run-specific information."""

import argparse
import pandas as pd


class Settings:
    """Contains all the usefull informations that can change between runs."""

    def __init__(self, config: dict, args: argparse.Namespace):
        # Attributes list
        self.user_id_column: str
        self.item_id_column: str
        self.timestamp_column: str

        self.epochs: int
        self.train_ratio: int
        self.sequence_length: int
        self.sequence_stride: int
        self.train_batch_size: int
        self.test_batch_size: int

        self.model_name: str
        self.model_attributes: dict[str:object]
        self.loss: dict[str:int]
        self.metrics: list[str]

        # Required attributes
        self.device = "cpu" if args.gpu is None else f"cuda:{args.gpu}"
        self.dataset = config["dataset"]

        # Facultative attributes
        self.user_features: list[str] = []
        self.item_features: list[str] = []
        self.evaluate_every: int = 1
        self.heat_up_length: int = 0
        self.training_heat_up: bool = False
        self.l2: float = 0

        # Fill in attributes from config
        for key, value in config.items():
            setattr(self, key, value)

        # Attributes defined later
        self.nb_interactions = None
        self.nb_users = None
        self.nb_items = None

    def set_dataset_info(self, dataset: pd.DataFrame) -> None:
        """Add dataset-specific information extracted after loading the data."""
        self.nb_interactions = len(dataset)
        self.nb_users = len(dataset[self.user_id_column].unique())
        self.nb_items = len(dataset[self.item_id_column].unique())

    def __str__(self) -> str:
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])
