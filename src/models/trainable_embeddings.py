"""Simple straightforward model consisting of one layer of trainable embeddings."""

import torch

from context import Context


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
