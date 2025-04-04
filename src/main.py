"""Entry point of the framework."""

# standard imports
import argparse
import traceback

# third-party imports
import yaml

# first-party imports
import pandas as pd
import torch
from tqdm import tqdm
import wandb


from settings import Settings
from trainable_embeddings import TrainableEmbeddings
from limnet import LiMNet
import metrics

torch.autograd.set_detect_anomaly(True)


def main():
    """Main function."""
    args = parse_args()
    for config in args.config:
        try:
            settings, run = get_config(args, config_name=config)
            data = get_dataset(settings)
            model, optimizer = build_model(settings)
            train_model(model, data, settings, optimizer, run)
        # pylint: disable=locally-disabled, broad-exception-caught
        except Exception:
            print(traceback.format_exc())

        run.finish()


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Link Prediction for Temporal Interaction Network"
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="+",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--gpu", type=str, required=False, default=None, help="Specify a gpu to use."
    )
    return parser.parse_args()


def get_config(
    args: argparse.Namespace, config_name: str
) -> tuple[Settings, wandb.sdk.wandb_run.Run]:
    """Returns the config given as a CLI argument."""
    config_raw = load_config(config_name)
    settings = Settings(config_raw, args)
    run = set_up_wandb(settings)
    return settings, run


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
    data = reindex_ids(data, settings)
    data = compute_features(data, settings)
    train_df, test_df = train_test_split(data, settings.train_ratio, settings)
    return torch.utils.data.DataLoader(
        TemporalInteractionNetworkDataset(
            df=train_df,
            user_ids=settings.user_id_column,
            user_features=settings.user_features,
            item_ids=settings.item_id_column,
            item_features=settings.item_features,
            timestamps=settings.timestamp_column,
            sequence_length=settings.sequence_length,
            sequence_stride=settings.sequence_stride,
            device=settings.device,
        ),
        batch_size=settings.train_batch_size,
        shuffle=False,
    ), torch.utils.data.DataLoader(
        TemporalInteractionNetworkDataset(
            df=test_df,
            user_ids=settings.user_id_column,
            user_features=settings.user_features,
            item_ids=settings.item_id_column,
            item_features=settings.item_features,
            timestamps=settings.timestamp_column,
            sequence_length=settings.sequence_length,
            sequence_stride=settings.sequence_stride,
            device=settings.device,
        ),
        batch_size=settings.test_batch_size,
        shuffle=False,
    )


def reindex_ids(data: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Rename the ids in the data so that all users and items ids are indexed from 0."""
    data[settings.user_id_column] = reindex_serie(data[settings.user_id_column])
    data[settings.item_id_column] = reindex_serie(data[settings.item_id_column])
    return data


def reindex_serie(serie: pd.Series):
    """Return the serie of indices from 0 corresponding to the input serie."""
    return serie.map({item: idx for idx, item in enumerate(serie.unique())})


def compute_features(data: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Compute the features to input to the models."""
    for feature in set(settings.user_features) | set(settings.item_features):
        match feature:
            case "delta_users":
                data = compute_deltas(
                    data,
                    id_column=settings.user_id_column,
                    timestamp_column=settings.timestamp_column,
                    column_name="delta_users",
                )
            case "delta_items":
                data = compute_deltas(
                    data,
                    id_column=settings.item_id_column,
                    timestamp_column=settings.timestamp_column,
                    column_name="delta_items",
                )
            case _:
                raise NotImplementedError("Requested features are not implemented yet.")
    return data


def compute_deltas(
    data: pd.DataFrame, id_column: str, timestamp_column: str, column_name: str
) -> pd.DataFrame:
    """Compute for each interaction the time since the last time the user/item interacted.

    Arguments:
    id_column: name of the column containing the user/item ids, depending on which one to compute.
    timestamp_column: name of the column containing interaction timestamps.
    column_name: name of the column containing the computed deltas.
    """
    data = data.sort_values(by=[id_column, timestamp_column])
    data[column_name] = (
        data[timestamp_column].diff().apply(lambda x: 0 if pd.isna(x) or x < 0 else x)
    )
    return data.sort_index()


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

    Each row contain a (users, items), pair.
        users: (batch_size, sequence_length, 1 + nb_user_features).
        items: (batch_size, sequence_length, 1 + nb_item_features).
    Where users[:,:,0] and items[:,:,0] are the users/items ids
    and users[:,:,1:] and items[:,:,1:] are the users/items features (there can be none).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user_ids: str,
        user_features: list[str],
        item_ids: str,
        item_features: list[str],
        timestamps: str,
        sequence_length: int,
        sequence_stride: int,
        device: torch.DeviceObjType,
    ):
        super().__init__()
        self.users = df[user_ids]
        self.user_features = df[user_features]
        self.items = df[item_ids]
        self.item_features = df[item_features]
        self.timestamps = df[timestamps]
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.device = device

    def __len__(self) -> int:
        return int((len(self.users) - self.sequence_length) / self.sequence_stride)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.sequence_stride * index
        stop = start + self.sequence_length
        user_ids = self.users[start:stop].to_numpy()
        user_features = self.user_features[start:stop].to_numpy()
        item_ids = self.items[start:stop].to_numpy()
        item_features = self.item_features[start:stop].to_numpy()
        return (
            torch.hstack(
                (
                    torch.tensor(user_ids, device=self.device).unsqueeze(1),
                    torch.tensor(user_features, device=self.device),
                )
            ),
            torch.hstack(
                (
                    torch.tensor(item_ids, device=self.device).unsqueeze(1),
                    torch.tensor(item_features, device=self.device),
                )
            ),
        )


def build_model(settings: Settings) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Creates the model."""
    model = pick_model(settings.model_name)(**settings.model_attributes)
    model.build(settings)
    learning_rate = 0.001 * settings.train_batch_size / 64
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=settings.l2
    )
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
        training_loss = train_epoch(model, train_data, settings, optimizer)
        results = {"epoch": epoch, "Training loss": training_loss}
        if epoch % settings.evaluate_every == 0:
            results = results | evaluate(model, test_data, settings)
        run.log(results)


def train_epoch(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    settings: Settings,
    optimizer: torch.optim.Optimizer,
):
    """Train one epoch."""
    loss_fn = assemble_loss_fn(settings.loss)
    num_batches = len(data)
    train_loss = 0
    for user_batch, item_batch in tqdm(
        data, desc="Training", leave=False, total=num_batches
    ):
        train_loss += training_step(
            model, settings, optimizer, loss_fn, user_batch, item_batch
        )
    return train_loss / num_batches


def training_step(
    model: torch.nn.Module,
    settings: Settings,
    optimizer: torch.optim.Optimizer,
    loss_fn: callable,
    user_sequences: torch.Tensor,
    item_sequences: torch.Tensor,
):
    """Run training on one batch of sequences.

    Arguments:
    loss_fn: receive (user_embeddings, item_embeddings) as input and return a float.
        user_embeddings: (batch_size, embedding_size) tensor
        item_embeddings: (batch_size, embedding_size) tensor
    user_sequences: (batch_size, sequence_length, 1 + nb_user_features) tensor.
    item_sequences: (batch_size, sequence_length, 1 + nb_item_features) tensor.
    """
    batch_size, sequence_size, _ = user_sequences.shape
    model.initialize_batch_run(batch_size=batch_size)
    loss = 0
    if settings.training_heat_up:
        heat_up_model(
            model, user_sequences, item_sequences, length=settings.heat_up_length
        )
    model.train()
    for i in range(
        settings.heat_up_length if settings.training_heat_up else 0, sequence_size
    ):
        users, items = user_sequences[:, i], item_sequences[:, i]
        user_ids, item_ids = users[:, 0].to(torch.int32), items[:, 0].to(torch.int32)
        user_features, item_features = users[:, 1:], items[:, 1:]
        user_embeddings, item_embeddings = model(
            user_ids=user_ids,
            user_features=user_features,
            item_ids=item_ids,
            item_features=item_features,
        )
        loss += loss_fn(user_embeddings, item_embeddings)
    loss = loss / sequence_size
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


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
):
    """Perform an evaluation of the model.

    Arguments:
    data: a contains batches of (users, items) tuples where
        users: (batch_size, sequence_length, 1 + nb_user_features).
        items: (batch_size, sequence_length, 1 + nb_item_features).
    """
    model.eval()
    num_batches = len(data)
    loss_fn = assemble_loss_fn(settings.loss)
    test_loss = 0
    measures = {}
    with torch.no_grad():
        # pylint: disable:locally-disabled, invalid-name
        for X in tqdm(data, desc="validation", leave=False):
            user_sequences, item_sequences = X
            batch_size = user_sequences.shape[0]
            model.initialize_batch_run(batch_size=batch_size)
            heat_up_model(
                model, user_sequences, item_sequences, length=settings.heat_up_length
            )
            for i in range(settings.heat_up_length, settings.sequence_length):
                users, items = user_sequences[:, i], item_sequences[:, i]
                test_loss += evaluate_step(
                    model,
                    settings,
                    loss_fn,
                    measures,
                    users,
                    items,
                )
    test_loss /= num_batches * (settings.sequence_length - settings.heat_up_length)
    for measure in measures.keys():
        measures[measure] /= num_batches * (
            settings.sequence_length - settings.heat_up_length
        )
    return {"Testing loss": test_loss} | measures


def heat_up_model(
    model: torch.nn.Module,
    user_sequences: torch.Tensor,
    item_sequences: torch.Tensor,
    length: int,
) -> None:
    """
    Run the model on the start of the sequence without evaluating.

    Arguments:
    users: (batch_size, sequence_length, 1 + nb_user_features) tensor.
    items: (batch_size, sequence_length, 1 + nb_item_features) tensor.
    length: must be less than sequence_length.
    """
    model.eval()
    for i in range(length):
        users, items = user_sequences[:, i], item_sequences[:, i]
        user_ids, item_ids = users[:, 0].to(torch.int32), items[:, 0].to(torch.int32)
        user_features, item_features = users[:, 1:], items[:, 1:]
        model(
            user_ids=user_ids,
            user_features=user_features,
            item_ids=item_ids,
            item_features=item_features,
        )


def evaluate_step(
    model: torch.nn.Module,
    settings: Settings,
    loss_fn: callable,
    measures: dict[str, float],
    users: torch.Tensor,
    items: torch.Tensor,
):
    """Run evaluation of one step in a sequence.

    Arguments:
    loss_fn: receive (user_embeddings, item_embeddings) as input and return a float.
        user_embeddings: (batch_size, embedding_size) tensor
        item_embeddings: (batch_size, embedding_size) tensor
    measures: Dict where values are the sum of the metrics over all steps.
    users: (batch_size, 1 + nb_user_features) tensor.
    items: (batch_size, 1 + nb_item_features) tensor.
    """
    batch_size = users.shape[0]
    user_ids, item_ids = users[:, 0].to(torch.int32), items[:, 0].to(torch.int32)
    user_features, item_features = users[:, 1:], items[:, 1:]
    user_embedding = model(user_ids=user_ids.unsqueeze(1)).squeeze()
    item_embeddings = model(
        item_ids=torch.arange(settings.nb_items, device=settings.device).repeat(
            batch_size, 1
        )
    )
    expected_item_embeddings = item_embeddings[torch.arange(batch_size), item_ids]
    test_loss = loss_fn(user_embedding, expected_item_embeddings)
    compute_metrics(
        settings.metrics, measures, item_ids, user_embedding, item_embeddings
    )
    # Communicate the interaction to the model for memory updates.
    model(
        user_ids=user_ids,
        user_features=user_features,
        item_ids=item_ids,
        item_features=item_features,
    )
    return test_loss


def compute_metrics(
    metrics_list: list[str],
    measures: dict[str:float],
    item_id: torch.Tensor,
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
):
    """
    Evaluates the metrics on the given embeddings and adds them to the past measures.

    Arguments:
    metrics: the name of the metrics function as defined in the metric module.
    measures: the sums of the evaluations of the metrics over past data.
    item_id: (batch_size) tensor. Contains the id of the expected items.
    user_embeddings: (batch_size, embedding size) tensor.
    item_embeddings: (nb_items, embedding size) tensor.
        Contains the embeddings for every single item.
    """
    for metric in metrics_list:
        measures[metric] = measures.get(metric, 0) + pick_metric(metric)(
            user_embedding, item_embeddings, item_id
        )


def score_all(
    user_embedding: torch.Tensor, item_embeddings: torch.Tensor
) -> torch.Tensor:
    """Computes the dot product of given embedding and all items."""
    return torch.einsum("be,ie->bi", user_embedding, item_embeddings)


if __name__ == "__main__":
    main()
