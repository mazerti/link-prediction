"""Entry point of the framework."""

# standard imports
import argparse
from collections.abc import Callable
import os
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
from metrics import pick_metric
from models.deepred import DeePRed
from models.jodie import Jodie
from models.limnet import LiMNet
from models.trainable_embeddings import TrainableEmbeddings

torch.autograd.set_detect_anomaly(True)


def main():
    """Main function."""
    args = parse_args()
    for config in args.config:
        try:
            run_training(args, config)
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
        help="Path to the checkpoint PTH or to the configuration YAML file.",
    )
    parser.add_argument(
        "--gpu", type=str, required=False, default=None, help="Specify a gpu to use."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=False,
        default=None,
        help="If you want to change the number of epochs from the loaded configs.",
    )
    return parser.parse_args()


def run_training(args: argparse.Namespace, config_name: str) -> wandb.sdk.wandb_run.Run:
    """Run the whole training for the requested config and arguments."""
    context = initialize_run(
        args, config_name=config_name, wand_project="link-prediction"
    )
    train_model(context)


def initialize_run(
    args: argparse.Namespace, config_name: str, wand_project: str
) -> Context:
    """Initialize the context for the training."""
    config_type = os.path.splitext(config_name)[-1]
    if config_type == ".yaml":
        context = Context(load_config(config_name), args)
    elif config_type == ".pth":
        context = Context.from_checkpoint(config_name, args)

    context.model = pick_model(context.model_name)(**context.model_attributes)
    if hasattr(context.model, "requested_features"):
        context.requested_features |= context.model.requested_features

    df: pd.DataFrame = pd.read_csv(context.dataset)
    context.data = data_loader.prepare_dataset(df, context)
    context.nb_interactions = len(df)
    context.nb_users = len(df[context.user_id_column].unique())
    context.nb_items = len(df[context.item_id_column].unique())

    context = build_model(context)

    context.run = set_up_wandb(context, wand_project)
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
        case "Jodie":
            return Jodie
    raise ValueError("Provided model name does not match an implementation.")


def build_model(context: Context) -> Context:
    """Build the model and optimizer."""
    context.model.build(context)

    context.optimizer = torch.optim.Adam(
        context.model.parameters(), lr=context.learning_rate, weight_decay=context.l2
    )

    if context.checkpoint:
        context.model.load_state_dict(context.checkpoint["model_state_dict"])
        context.optimizer.load_state_dict(context.checkpoint["optimizer_state_dict"])

    if context.lr_scheduler:
        create_lr_scheduler(context.optimizer, context)

    return context


def create_lr_scheduler(optimizer: torch.optim.Optimizer, context: Context) -> None:
    """Create a learning rate scheduler based on the description given in the context."""

    def unfold_scheduler_entry(scheduler_entry: dict[str : dict[str:float]]):
        name, payload = next(iter(scheduler_entry.items()))
        milestone = payload.pop("milestone")

        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer=context.optimizer, **payload
        )
        return scheduler, milestone

    lr_schedulers, milestones = zip(
        *map(
            unfold_scheduler_entry,
            context.lr_scheduler,
        )
    )
    context._lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        context.optimizer, lr_schedulers, milestones=milestones[1:]
    )


def set_up_wandb(context: Context, project: str) -> wandb.sdk.wandb_run.Run:
    """Set up W&B for logging evaluation."""
    return wandb.init(
        project=project,
        config=context.to_save(),
        id=context.get_id(),
        resume="allow",
    )


def train_model(
    context: Context,
) -> None:
    """Process to the model training on given dataset."""
    train_data, test_data = context.data
    if context.epoch > context.epochs:
        results = {"epoch": context.epoch} | evaluate(context.model, test_data, context)
        context.run.log(results)
        return
    for epoch in tqdm(
        range(context.epoch, context.epochs + 1),
        initial=context.epoch,
        total=context.epochs,
        desc="epochs",
    ):
        training_loss = train_epoch(context.model, train_data, context)
        results = {
            "epoch": epoch,
            "Training loss": training_loss,
            "learning-rate": context.optimizer.param_groups[0]["lr"],
        }
        if epoch % context.evaluate_every == 0:
            results = results | evaluate(context.model, test_data, context)
        if epoch % context.checkpoint_every == 0:
            save_progress(context, epoch)
        context.run.log(results)
        if context._lr_scheduler:
            context._lr_scheduler.step()
        context.epoch = epoch
    save_progress(context, context.epochs)


def train_epoch(
    model: torch.nn.Module,
    data: torch.utils.data.DataLoader,
    context: Context,
) -> torch.Tensor:
    """Train one epoch."""
    loss_fn = assemble_loss_fn(context.loss)
    num_batches = len(data)
    train_loss = 0
    for user_batch, item_batch in tqdm(
        data, desc="Training", leave=False, total=num_batches
    ):
        train_loss += model.training_sequence(
            context, context.optimizer, loss_fn, user_batch, item_batch
        )
    return train_loss / num_batches


def assemble_loss_fn(
    losses: dict[str, float],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Generate a loss function as a weighted sum of the listed losses."""

    def loss_fn(
        context: Context, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Apply all losses and sum their results with matching weights."""
        return sum(
            (
                pick_metric(loss)(context, user_embeddings, item_embeddings) * weight
                for loss, weight in losses.items()
            )
        )

    return loss_fn


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
        for X in tqdm(data, desc="validation", leave=False):
            user_sequences, item_sequences = X
            test_loss += model.evaluate_sequence(
                context, loss_fn, measures, user_sequences, item_sequences
            )
    test_loss /= num_batches * (
        context.test_sequence_length - context.test_heat_up_length
    )
    for measure in measures.keys():
        measures[measure] /= num_batches * (
            context.test_sequence_length - context.test_heat_up_length
        )
    return {"Testing loss": test_loss} | measures


def save_progress(context: Context, epoch: int):
    """Save the progress of the training to a file."""
    parent_folder = os.path.join("checkpoints", context.get_id())
    os.makedirs(parent_folder, exist_ok=True)
    write_checkpoint(context, epoch, parent_folder)
    delete_old_checkpoint(context, epoch, parent_folder)


def write_checkpoint(context: Context, epoch: int, parent_folder: str):
    """Write the content of the context in a new checkpoint file."""
    checkpoint_file = os.path.join(
        parent_folder, f"checkpoint-{str(epoch).zfill(len(str(context.epochs)))}.pth"
    )  # clever code adding leading zeros to the epoch number, appropriate to the run max.
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": context.model.state_dict(),
            "optimizer_state_dict": context.optimizer.state_dict(),
            "context": context.to_save(),
        },
        checkpoint_file,
    )


def delete_old_checkpoint(context: Context, epoch: int, parent_folder: str):
    """Delete old checkpoint to limit the disk use."""
    file_to_delete = os.path.join(
        parent_folder,
        "checkpoint-"
        + str(epoch - context.nb_checkpoints_to_keep * context.checkpoint_every).zfill(
            len(str(context.epochs))
        )
        + ".pth",
    )
    try:
        os.remove(file_to_delete)
    except OSError:
        pass


if __name__ == "__main__":
    main()
