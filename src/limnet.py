"""Cross RNN embeddings described in 'LiMNet: Early-Stage Detection of IoT Botnets
with Lightweight Memory Networks' by L.Giaretta et al."""

import torch

from settings import Settings


class LiMNet(torch.nn.Module):
    """Own implementation of the Lightweight Memory Network."""

    def __init__(
        self, embedding_size: int, initialization: callable, *args, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_size: int = embedding_size
        self.initialization: callable = getattr(torch.nn.init, initialization)

        self.user_memory: torch.Tensor = None
        self.item_memory: torch.Tensor = None
        self.cell: torch.nn.GRUCell = None

        # Arguments defined later
        self.nb_users: int
        self.nb_items: int

    def build(self, settings: Settings) -> None:
        """Builds the model."""
        self.complete_arguments(settings)
        self.initialize_memory(settings)
        self.cell = torch.nn.GRUCell(
            2 * self.embedding_size, self.embedding_size, device=settings.device
        )

    def complete_arguments(self, settings: Settings) -> None:
        """Complete the known arguments with up to date settings."""
        self.nb_users = settings.nb_users
        self.nb_items = settings.nb_items

    def initialize_memory(
        self, settings: Settings
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize the model's memory.

        Returns:
        user_memory: (nb_users, embedding_size) tensor.
        item_memory: (nb_items, embedding_size) tensor.
        """
        self.user_memory = torch.zeros(
            (settings.nb_users, self.embedding_size), device=settings.device
        )
        self.item_memory = torch.zeros(
            (settings.nb_items, self.embedding_size), device=settings.device
        )
        self.initialization(self.user_memory)
        self.initialization(self.item_memory)
        return self.user_memory, self.item_memory

    # pylint: disable=locally-disabled, invalid-name, not-callable
    def forward(
        self,
        data: None | tuple[torch.Tensor, torch.Tensor] = None,
        users: None | tuple[torch.Tensor, torch.Tensor] = None,
        items: None | tuple[torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass, when called for an edge (user and item data) update memory. Otherwise just return the current state of the memory.

        Arguments:
        data: passing data=(users, items) is equivalent to passing users=users, items=items
        users: (batch_size) tensor.
        items: (batch_size) tensor.
        """
        if data is not None or (users is not None and items is not None):
            user_ids, item_ids = data or (users, items)
            return self.forward_edge(user_ids, item_ids)
        if users is not None:
            return self.extract_user_embeddings(users)
        if items is not None:
            return self.extract_item_embeddings(items)
        raise ValueError("Invalid input, provide user, item, or both.")

    def forward_edge(
        self,
        users: tuple[torch.Tensor, torch.Tensor],
        items: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process an incomming edge, returning the updating memory (i.e. embedding) of the interacting nodes.

        Arguments:
        users: (batch_size) tensor.
        items: (batch_size) tensor.

        Returns:
        user_embeddings: (batch_size, embedding_size) tensor.
        item_embeddings: (batch_size, embedding_size) tensor.
        """
        user_embeddings = self.extract_user_embeddings(users)
        item_embeddings = self.extract_item_embeddings(items)
        user_input = torch.hstack((user_embeddings, item_embeddings))
        item_input = torch.hstack((item_embeddings, user_embeddings))

        updated_user_embeddings = self.cell(user_input)
        updated_item_embeddings = self.cell(item_input)

        self.user_memory[users, :] = updated_user_embeddings
        self.item_memory[items, :] = updated_item_embeddings
        return updated_user_embeddings, updated_item_embeddings

    def extract_user_embeddings(self, users: torch.Tensor) -> torch.Tensor:
        """Return the current state of the memory for the given users.

        Arguments:
        users: (batch_size) tensor.
        Returns: (batch_size, embedding_size) tensor.
        """
        return self.user_memory[users, :]

    def extract_item_embeddings(self, items: torch.Tensor) -> torch.Tensor:
        """Return the current state of the memory for the given items.

        Arguments:
        items: (batch_size) tensor.

        Returns: (batch_size, embedding_size) tensor.
        """
        return self.item_memory[items, :]
