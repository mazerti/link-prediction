"""Simple straightforward model consisting of one layer of trainable embeddings."""

from collections.abc import Callable

import torch
from tqdm import tqdm

from context import Context
from metrics import compute_metrics


class TrainableEmbeddings(torch.nn.Module):
    """Simple straightforward model consisting of one layer of trainable embeddings."""

    def __init__(self, embedding_size: int, normalize=True, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size: int = embedding_size
        self.normalize = normalize

        self.user_embeddings: torch.nn.Embedding = None
        self.item_embeddings: torch.nn.Embedding = None

        # Arguments defined later
        self.nb_users: int
        self.nb_items: int

    def build(self, settings: Context) -> None:
        """Builds the model."""
        self.complete_arguments(settings)
        self.user_embeddings = torch.nn.Embedding(
            self.nb_users, self.embedding_size, device=settings.device
        )
        self.item_embeddings = torch.nn.Embedding(
            self.nb_items, self.embedding_size, device=settings.device
        )

    def complete_arguments(self, settings: Context):
        """Complete the known arguments with up to date settings."""
        self.nb_users = settings.nb_users
        self.nb_items = settings.nb_items

    def initialize_batch_run(self, batch_size: int) -> None:
        """Prepare the model before processing a batch.

        There is nothing to do for that model."""

    # pylint: disable=locally-disabled, unused-argument
    def forward(
        self,
        user_ids: None | tuple[torch.Tensor, torch.Tensor] = None,
        user_features: any = None,
        item_ids: None | tuple[torch.Tensor, torch.Tensor] = None,
        item_features: any = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Return the embeddings for the requested users/items.

        This function can be called either on interactions, on a set of users or on a set of items

        When called on interaction, it must receive as input:
            user_ids: (batch_size) tensor.
            item_ids: (batch_size) tensor.
            user_features: not supported by this model but required by the framework.
            item_features: not supported by this model but required by the framework.
        The function will return the embeeddings for the given users and items.

        When called with only users or only items, the expected input are
            users: (batch_size, nb_users_to_extract) tensor.
            or
            items: (batch_size, nb_items_to_extract) tensor.
        The function will return the embeeddings for the given users and items.
        """
        if user_ids is not None and item_ids is not None:
            user_embeddings = self.user_embeddings(user_ids)
            item_embeddings = self.item_embeddings(item_ids)
            if self.normalize:
                return (
                    torch.nn.functional.normalize(user_embeddings, dim=-1),
                    torch.nn.functional.normalize(item_embeddings, dim=-1),
                )
            return (
                user_embeddings(user_ids),
                item_embeddings(item_ids),
            )
        if user_ids is not None:
            return torch.nn.functional.normalize(self.user_embeddings(user_ids), dim=-1)
        if item_ids is not None:
            return torch.nn.functional.normalize(self.item_embeddings(item_ids), dim=-1)
        raise ValueError("Invalid input, provide user, item, or both.")

    def training_sequence(
        self: torch.nn.Module,
        context: Context,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
        self.initialize_batch_run(batch_size=batch_size)
        loss = 0
        self.train()
        for i in range(sequence_size):
            users, items = user_sequences[:, i], item_sequences[:, i]
            user_ids, item_ids = users[:, 0].to(torch.int32), items[:, 0].to(
                torch.int32
            )
            user_features, item_features = users[:, 1:], items[:, 1:]
            user_embeddings, item_embeddings = self.forward(
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

    def evaluate_sequence(
        self: torch.nn.Module,
        context: Context,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        measures: dict[str, float],
        user_sequences: torch.Tensor,
        item_sequences: torch.Tensor,
    ):
        """Run evaluation on one batch of sequences.

        Arguments:
        loss_fn: receive (user_embeddings, item_embeddings) as input and return a float.
            user_embeddings: (batch_size, embedding_size) tensor
            item_embeddings: (batch_size, embedding_size) tensor
        user_sequences: (batch_size, sequence_length, 1 + nb_user_features) tensor.
        item_sequences: (batch_size, sequence_length, 1 + nb_item_features) tensor.
        """
        batch_size = user_sequences.shape[0]
        self.initialize_batch_run(batch_size=batch_size)
        test_loss = 0
        for i in tqdm(
            range(context.test_sequence_length),
            desc="sequence",
            leave=False,
        ):
            users, items = user_sequences[:, i], item_sequences[:, i]
            test_loss += self.evaluate_step(
                context,
                loss_fn,
                measures,
                users,
                items,
            )
        return test_loss

    def evaluate_step(
        self: torch.nn.Module,
        context: Context,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
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
        user_embedding = self.forward(user_ids=user_ids.unsqueeze(1)).squeeze(dim=1)
        item_embeddings = self.forward(
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
        self.forward(
            user_ids=user_ids,
            user_features=user_features,
            item_ids=item_ids,
            item_features=item_features,
        )
        return test_loss
