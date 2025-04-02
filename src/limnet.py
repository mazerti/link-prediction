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
        self.cell = torch.nn.GRUCell(
            2 * self.embedding_size, self.embedding_size, device=settings.device
        )

    def complete_arguments(self, settings: Settings) -> None:
        """Complete the known arguments with up to date settings."""
        self.nb_users = settings.nb_users
        self.nb_items = settings.nb_items

    # pylint: disable=locally-disabled, invalid-name, not-callable
    def forward(
        self,
        data: None | tuple[torch.Tensor, torch.Tensor] = None,
        users: None | tuple[torch.Tensor, torch.Tensor] = None,
        items: None | tuple[torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass, when called for an edge (user and item data) update memory. Otherwise just return the current state of the memory.

        When feeding only user/items for inference, make sure to initialize the memory by feeding at least 1 interaction beforehand.

        Arguments:
        data: passing data=(users, items) is equivalent to passing users=users, items=items
        users: (batch_size, sequence_size) tensor.
        items: (batch_size, sequence_size) tensor.
        """
        if data is not None or (users is not None and items is not None):
            user_ids, item_ids = data or (users, items)
            return self.forward_edge_sequence(user_ids, item_ids)
        if users is not None:
            return self.extract_user_embeddings(users)
        if items is not None:
            return self.extract_item_embeddings(items)
        raise ValueError("Invalid input, provide user, item, or both.")

    def forward_edge_sequence(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence of incomming edges, updating the memory (i.e. embedding) for the interacting nodes.

        Arguments:
        users: (batch_size, sequence_size) tensor.
        items: (batch_size, sequence_size) tensor.

        Returns:
        user_embeddings: (batch_size, sequence_size, embedding_size) tensor.
        item_embeddings: (batch_size, sequence_size, embedding_size) tensor.
        """
        self.initialize_memory(
            batch_size=users.shape[0],
            nb_users=self.nb_users,
            nb_items=self.nb_items,
            device=users.device,
        )
        batch_size, sequence_size = users.shape
        updated_user_embeddings = []
        updated_item_embeddings = []
        for i in range(sequence_size):
            batch_users, batch_items = users[:, i], items[:, i]
            user_embeddings = self.user_memory[torch.arange(batch_size), batch_users, :]
            item_embeddings = self.item_memory[torch.arange(batch_size), batch_items, :]
            user_input = torch.hstack((user_embeddings, item_embeddings))
            item_input = torch.hstack((item_embeddings, user_embeddings))

            new_user_embeddings = self.cell(user_input)
            new_item_embeddings = self.cell(item_input)

            self.user_memory[torch.arange(batch_size), batch_users, :] = (
                new_user_embeddings
            )
            self.item_memory[torch.arange(batch_size), batch_items, :] = (
                new_item_embeddings
            )

            updated_user_embeddings.append(new_user_embeddings)
            updated_item_embeddings.append(new_item_embeddings)
        return (
            torch.stack(updated_user_embeddings, dim=1),
            torch.stack(updated_item_embeddings, dim=1),
        )

    def initialize_memory(
        self, batch_size: int, nb_users: int, nb_items: int, device: torch.DeviceObjType
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize the model's memory.

        Returns:
        user_memory: (batch_size, nb_users, embedding_size) tensor.
        item_memory: (batch_size, nb_items, embedding_size) tensor.
        """
        self.user_memory = torch.zeros(
            (batch_size, nb_users, self.embedding_size), device=device
        )
        self.item_memory = torch.zeros(
            (batch_size, nb_items, self.embedding_size), device=device
        )
        self.initialization(self.user_memory)
        self.initialization(self.item_memory)
        return self.user_memory, self.item_memory

    def extract_user_embeddings(self, users: torch.Tensor) -> torch.Tensor:
        """Return the current state of the memory for the given users.

        Arguments:
        users: (batch_size, users_to_extract) tensor.
        Returns: (batch_size, users_to_extract, embedding_size) tensor.
        """
        batch_size, user_to_extract = users.shape
        assert (
            batch_size == self.user_memory.shape[0]
        ), f"shape mismatch: {batch_size} as input while {self.user_memory.shape[0]} for memory."
        return self.user_memory[torch.arange(users.shape[0]).unsqueeze(1), users, :]

    def extract_item_embeddings(self, items: torch.Tensor) -> torch.Tensor:
        """Return the current state of the memory for the given items.

        Arguments:
        items: (batch_size, items_to_extract) tensor.

        Returns: (batch_size, items_to_extract, embedding_size) tensor.
        """
        batch_size, item_to_extract = items.shape
        assert (
            batch_size == self.item_memory.shape[0]
        ), f"shape mismatch: {batch_size} as input while {self.item_memory.shape[0]} for memory."
        return self.item_memory[torch.arange(items.shape[0]).unsqueeze(1), items, :]
