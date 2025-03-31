"""Contains the settings class that gathers all the run-specific information."""

import argparse
import pandas as pd


class Settings:
    """Contains all the usefull informations that can change between runs."""

    def _set_base_settings(self) -> None:
        self.user_id_column = "user_id"
        self.item_id_column = "item_id"
        self.timestamp_column = "timestamp"

        self.epochs = 10
        self.train_ratio = 0.8
        self.train_batch_size = 64
        self.test_batch_size = 64

        self.model_name = "TrainableEmbeddings"
        self.model_attributes = {"embedding_size": 16}
        self.loss = "MSE"

    def __init__(self, config: dict, args: argparse.Namespace):
        self._set_base_settings()

        # optionnal arguments
        for key, value in config.items():
            setattr(self, key, value)

        # Required arguments
        self.device = "cpu" if args.gpu is None else f"cuda:{args.gpu}"
        self.dataset = config["dataset"]

        # Arguments defined later
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
