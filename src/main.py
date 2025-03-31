"""Entry point of the framework."""

import yaml

import argparse
import pandas as pd
import torch
import torch.utils.data.dataloader
from tqdm import tqdm
import wandb

from settings import Settings
from trainable_embeddings import TrainableEmbeddings
from limnet import LiMNet
import metrics


def main():
    """Main function."""
    settings, run = get_config()
    data = get_dataset(settings)
    model, optimizer = build_model(settings)
    train_model(model, data, settings, optimizer, run)
    run.finish()


def get_config() -> tuple[Settings, wandb.sdk.wandb_run.Run]:
    """Returns the config given as a CLI argument."""
    args = parse_args()
    config_raw = load_config(args.config)
    settings = Settings(config_raw, args)
    run = set_up_wandb(settings)
    return settings, run


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Link Prediction for Temporal Interaction Network"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--gpu", type=str, required=False, default=None, help="Specify a gpu to use."
    )
    return parser.parse_args()


def load_config(config_file: str) -> dict:
    """Load config from file."""
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    raise IOError("Config file Could not be read.")


def set_up_wandb(settings: Settings) -> wandb.sdk.wandb_run.Run:
    """Set up W&B for logging evaluation."""
    return wandb.init(
        project="link-prediction",
        config=settings.__dict__,
    )


def get_dataset(settings: Settings) -> torch.utils.data.DataLoader:
    """Load and prepare the dataset."""
    df = pd.read_csv(settings.dataset)
    settings.set_dataset_info(df)
    return prepare_dataset(df, settings)


def prepare_dataset(
    data: pd.DataFrame, settings: Settings
) -> torch.utils.data.DataLoader:
    """Split the dataset, select the columns and turn the data into a torch DataLoader."""
    train_df, test_df = train_test_split(data, settings.train_ratio, settings)
    return torch.utils.data.DataLoader(
        TemporalInteractionNetworkDataset(
            df=train_df,
            user_ids=settings.user_id_column,
            item_ids=settings.item_id_column,
            timestamps=settings.timestamp_column,
            device=settings.device,
        ),
        batch_size=settings.train_batch_size,
        shuffle=False,
    ), torch.utils.data.DataLoader(
        TemporalInteractionNetworkDataset(
            df=test_df,
            user_ids=settings.user_id_column,
            item_ids=settings.item_id_column,
            timestamps=settings.timestamp_column,
            device=settings.device,
        ),
        batch_size=settings.test_batch_size,
        shuffle=False,
    )


def train_test_split(
    df: pd.DataFrame, train_size: float, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a temporal split of the data."""
    split_at = int(len(df) * train_size)
    df = df.sort_values(by=settings.timestamp_column).reset_index()
    return df[:split_at], df[split_at:].reset_index()


class TemporalInteractionNetworkDataset(torch.utils.data.Dataset):
    """
    Torch Dataset for temporal interaction network.
    Each row contain a user_id, item_id, timestamp triplet and eventually features.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user_ids: str,
        item_ids: str,
        timestamps: str,
        device: torch.DeviceObjType,
    ):
        super().__init__()
        self.users = df[user_ids]
        self.items = df[item_ids]
        self.timestamps = df[timestamps]
        self.device = device

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        user_id = self.users[index]
        item_id = self.items[index]
        return (
            torch.tensor(user_id, device=self.device),
            torch.tensor(item_id, device=self.device),
        )


def build_model(settings: Settings) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Creates the model."""
    model = pick_model(settings.model_name)(**settings.model_attributes)
    model.build(settings)
    optimizer = torch.optim.Adam(model.parameters())
    return model, optimizer


def pick_model(model_name: str) -> torch.nn.Module:
    """Return the implementation of the model matching the given model name."""
    match model_name:
        case "TrainableEmbeddings":
            return TrainableEmbeddings
        case "LiMNet":
            return LiMNet
    raise ValueError("Provided model name does not match an implementation.")


def train_model(
    model: torch.nn.Module,
    dataset: tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
    settings: Settings,
    optimizer: torch.optim.Optimizer,
    run: wandb.sdk.wandb_run.Run,
) -> None:
    """Process to the model training on given dataset."""
    train_data, test_data = dataset
    for epoch in tqdm(range(1, settings.epochs + 1), desc="epochs"):
        train_epoch(model, train_data, settings, optimizer, run, epoch)
        evaluate(model, test_data, settings, run, epoch)


def train_epoch(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    settings: Settings,
    optimizer: torch.optim.Optimizer,
    run: wandb.sdk.wandb_run.Run,
    epoch: int,
):
    """Train one epoch."""
    loss_fn = assemble_loss_fn(settings.loss)
    num_batches = len(data)
    train_loss = 0
    model.train()
    for batch, X in tqdm(
        enumerate(data), desc="Training", leave=False, total=num_batches
    ):
        user_embeddings, item_embeddings = model(data=X)
        loss = loss_fn(user_embeddings, item_embeddings)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
    train_loss /= num_batches
    run.log({"epoch": epoch, "Training loss": train_loss})


def assemble_loss_fn(losses: dict[str:float]) -> callable:
    """Generate a loss function as a weighted sum of the listed losses."""

    def loss_fn(
        user_embeddings: torch.Tensor, item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Apply all losses and sum their results with matching weights."""
        return sum(
            (
                pick_metric(loss)(user_embeddings, item_embeddings) * weight
                for loss, weight in losses.items()
            )
        )

    return loss_fn


def pick_metric(metric_name: str) -> callable:
    """Select the implementation of the metric function matching the given function name."""
    return getattr(metrics, metric_name)


def evaluate(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    settings: Settings,
    run: wandb.sdk.wandb_run.Run,
    epoch: int,
):
    """Perform an evaluation of the model."""
    model.eval()
    num_batches = len(data)
    loss_fn = assemble_loss_fn(settings.loss)
    test_loss = 0
    measures = {}
    with torch.no_grad():
        for X in tqdm(data, desc="validation", leave=False):
            user_id, item_id = X
            user_embedding = model(users=user_id)
            item_embeddings = model(
                items=torch.arange(settings.nb_items, device=settings.device)
            )
            expected_item_embedding = item_embeddings[item_id]
            test_loss += loss_fn(user_embedding, expected_item_embedding)
            for metric in settings.metrics:
                measures[metric] = measures.get(metric, 0) + pick_metric(metric)(
                    user_embedding, item_embeddings, item_id
                )
    test_loss /= num_batches
    for measure in measures.keys():
        measures[measure] /= num_batches
    run.log({"epoch": epoch, "Testing loss": test_loss} | measures)


def score_all(
    user_embedding: torch.Tensor, item_embeddings: torch.Tensor
) -> torch.Tensor:
    """Computes the dot product of given embedding and all items."""
    return torch.einsum("be,ie->bi", user_embedding, item_embeddings)


if __name__ == "__main__":
    main()
