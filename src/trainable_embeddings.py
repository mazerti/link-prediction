"""Simple straightforward model consisting of one layer of trainable embeddings."""

import torch

from settings import Settings


class TrainableEmbeddings(torch.nn.Module):
    """Simple straightforward model consisting of one layer of trainable embeddings."""

    def __init__(self, embedding_size: int, **kwargs):
        super().__init__(**kwargs)
        self.embedding_size: int = embedding_size

        self.user_embeddings: torch.nn.Embedding = None
        self.item_embeddings: torch.nn.Embedding = None

        # Arguments defined later
        self.nb_users: int
        self.nb_items: int

    def build(self, settings: Settings) -> None:
        """Builds the model."""
        self.complete_arguments(settings)
        self.user_embeddings = torch.nn.Embedding(
            self.nb_users, self.embedding_size, device=settings.device
        )
        self.item_embeddings = torch.nn.Embedding(
            self.nb_items, self.embedding_size, device=settings.device
        )

    def complete_arguments(self, settings: Settings):
        """Complete the known arguments with up to date settings."""
        self.nb_users = settings.nb_users
        self.nb_items = settings.nb_items

    def initialize_batch_run(self, batch_size: int) -> None:
        """Prepare the model before processing a batch.

        There is nothing to do for that model."""

    # pylint: disable=locally-disabled, invalid-name, not-callable
    def forward(
        self,
        data: None | tuple[torch.Tensor, torch.Tensor] = None,
        users: None | tuple[torch.Tensor, torch.Tensor] = None,
        items: None | tuple[torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass

        Arguments:
        data: passing data=(users, items) is equivalent to passing users=users, items=items
        users: (batch_size) tensor.
        items: (batch_size) tensor.
        """
        if data is not None or (users is not None and items is not None):
            user_ids, item_ids = data or (users, items)
            return (
                torch.nn.functional.normalize(self.user_embeddings(user_ids), dim=-1),
                torch.nn.functional.normalize(self.item_embeddings(item_ids), dim=-1),
            )
        if users is not None:
            return torch.nn.functional.normalize(self.user_embeddings(users), dim=-1)
        if items is not None:
            return torch.nn.functional.normalize(self.item_embeddings(items), dim=-1)
        raise ValueError("Invalid input, provide user, item, or both.")
