"""Entry point of the framework."""

# standard imports
import argparse
import traceback

# third-party imports
import pandas as pd
import torch
from tqdm import tqdm
import wandb
import yaml

# first-party imports
from context import Context
import data_loader
import metrics
from models.deepred import DeePRed
from models.limnet import LiMNet
from models.trainable_embeddings import TrainableEmbeddings

torch.autograd.set_detect_anomaly(True)


def main():
    """Main function."""
    args = parse_args()
    for config in args.config:
        try:
            run_training(args, config)
        # pylint: disable=locally-disabled, broad-exception-caught
        except Exception:
            print(traceback.format_exc())


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


def run_training(args: argparse.Namespace, config_name: str) -> wandb.sdk.wandb_run.Run:
    """Run the whole training for the requested config and arguments."""
    context = initialize_run(args, config_name=config_name)
    train_model(context)


def initialize_run(
    args: argparse.Namespace, config_name: str
) -> tuple[
    torch.nn.Module, torch.utils.data.DataLoader, Context, wandb.sdk.wandb_run.Run
]:
    """Initialize the context for the training."""
    context = Context(load_config(config_name), args)

    context.model = pick_model(context.model_name)(**context.model_attributes)
    if hasattr(context.model, "requested_features"):
        context.requested_features |= context.model.requested_features

    df: pd.DataFrame = pd.read_csv(context.dataset)
    context.data = data_loader.prepare_dataset(df, context)
    context.nb_interactions = len(df)
    context.nb_users = len(df[context.user_id_column].unique())
    context.nb_items = len(df[context.item_id_column].unique())

    context = build_model(context)

    context.run = set_up_wandb(context)
    return context


def load_config(config_file: str) -> dict:
    """Load config from file."""
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    raise IOError("Config file Could not be read.")


def pick_model(model_name: str) -> torch.nn.Module:
    """Return the implementation of the model matching the given model name."""
    match model_name:
        case "TrainableEmbeddings":
            return TrainableEmbeddings
        case "LiMNet":
            return LiMNet
        case "DeePRed":
            return DeePRed
    raise ValueError("Provided model name does not match an implementation.")


def build_model(context: Context) -> Context:
    """Build the model and optimizer."""
    context.model.build(context)
    # learning_rate = 0.001 * context.train_batch_size / 64
    context.optimizer = torch.optim.Adam(
        context.model.parameters(), lr=context.learning_rate, weight_decay=context.l2
    )
    return context


def set_up_wandb(context: Context) -> wandb.sdk.wandb_run.Run:
    """Set up W&B for logging evaluation."""
    return wandb.init(
        project="link-prediction",
        config=context.__dict__,
    )


def train_model(
    context: Context,
) -> None:
    """Process to the model training on given dataset."""
    train_data, test_data = context.data
    for epoch in tqdm(range(1, context.epochs + 1), desc="epochs"):
        training_loss = train_epoch(context.model, train_data, context)
        results = {"epoch": epoch, "Training loss": training_loss}
        if epoch % context.evaluate_every == 0:
            results = results | evaluate(context.model, test_data, context)
        context.run.log(results)


def train_epoch(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    context: Context,
):
    """Train one epoch."""
    loss_fn = assemble_loss_fn(context.loss)
    num_batches = len(data)
    train_loss = 0
    for user_batch, item_batch in tqdm(
        data, desc="Training", leave=False, total=num_batches
    ):
        train_loss += training_step(
            model, context, context.optimizer, loss_fn, user_batch, item_batch
        )
    return train_loss / num_batches


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


def training_step(
    model: torch.nn.Module,
    context: Context,
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
    heat_up_model(
        model, user_sequences, item_sequences, length=context.train_heat_up_length
    )
    model.train()
    for i in range(context.train_heat_up_length, sequence_size):
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


def evaluate(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    context: Context,
):
    """Perform an evaluation of the model.

    Arguments:
    data: a contains batches of (users, items) tuples where
        users: (batch_size, sequence_length, 1 + nb_user_features).
        items: (batch_size, sequence_length, 1 + nb_item_features).
    """
    model.eval()
    num_batches = len(data)
    loss_fn = assemble_loss_fn(context.loss)
    test_loss = 0
    measures = {}
    with torch.no_grad():
        # pylint: disable:locally-disabled, invalid-name
        for X in tqdm(data, desc="validation", leave=False):
            user_sequences, item_sequences = X
            batch_size = user_sequences.shape[0]
            model.initialize_batch_run(batch_size=batch_size)
            heat_up_model(
                model,
                user_sequences,
                item_sequences,
                length=context.test_heat_up_length,
            )
            for i in range(context.test_heat_up_length, context.test_sequence_length):
                users, items = user_sequences[:, i], item_sequences[:, i]
                test_loss += evaluate_step(
                    model,
                    context,
                    loss_fn,
                    measures,
                    users,
                    items,
                )
    test_loss /= num_batches * (
        context.test_sequence_length - context.test_heat_up_length
    )
    for measure in measures.keys():
        measures[measure] /= num_batches * (
            context.test_sequence_length - context.test_heat_up_length
        )
    return {"Testing loss": test_loss} | measures


def evaluate_step(
    model: torch.nn.Module,
    context: Context,
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
    user_embedding = model(user_ids=user_ids.unsqueeze(1)).squeeze(dim=1)
    item_embeddings = model(
        item_ids=torch.arange(context.nb_items, device=context.device).repeat(
            batch_size, 1
        )
    )
    expected_item_embeddings = item_embeddings[torch.arange(batch_size), item_ids]
    test_loss = loss_fn(user_embedding, expected_item_embeddings)
    compute_metrics(
        context.metrics, measures, item_ids, user_embedding, item_embeddings
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


if __name__ == "__main__":
    main()
