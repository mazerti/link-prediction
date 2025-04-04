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
        self.user_cell: torch.nn.GRUCell = None
        self.item_cell: torch.nn.GRUCell = None

        # Arguments defined later
        self.nb_users: int
        self.nb_items: int
        self.device: torch.DeviceObjType

    def build(self, settings: Settings) -> None:
        """Builds the model."""
        self.nb_users = settings.nb_users
        self.nb_items = settings.nb_items
        self.device = settings.device
        input_size = (
            self.embedding_size
            + len(settings.user_features)
            + self.embedding_size
            + len(settings.item_features)
        )
        self.user_cell = torch.nn.GRUCell(
            input_size, self.embedding_size, device=settings.device
        )
        self.item_cell = torch.nn.GRUCell(
            input_size, self.embedding_size, device=settings.device
        )

    def initialize_batch_run(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize the model's memory.

        Returns:
        user_memory: (batch_size, nb_users, embedding_size) tensor.
        item_memory: (batch_size, nb_items, embedding_size) tensor.
        """
        self.user_memory = torch.zeros(
            (batch_size, self.nb_users, self.embedding_size), device=self.device
        )
        self.item_memory = torch.zeros(
            (batch_size, self.nb_items, self.embedding_size), device=self.device
        )
        self.initialization(self.user_memory)
        self.initialization(self.item_memory)
        return self.user_memory, self.item_memory

    # pylint: disable=locally-disabled, invalid-name, not-callable
    def forward(
        self,
        user_ids: None | torch.Tensor = None,
        user_features: None | torch.Tensor = None,
        item_ids: None | torch.Tensor = None,
        item_features: None | torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Return the embeddings for the requested users/items.

        This function can be called either to make the model memorize an interaction or to return
        the current embeddings.

        When watching an interaction, it must receive as input:
            user_ids: (batch_size) tensor.
            item_ids: (batch_size) tensor.
            user_features: (batch_size, nb_user_features) tensor. nb_user_features can be 0.
            item_features: (batch_size, nb_item_features) tensor. nb_item_features can be 0.
        The function will compute the new embeeddings for the given users and items, and update
        the current memory with these new embeddings.

        When called with only users or only items, the expected input are
            users: (batch_size, nb_users_to_extract) tensor.
            or
            items: (batch_size, nb_items_to_extract) tensor.
        This will simply return the current state of the memory for the requested users/items.
        If features are given as input they will be disregarded since no computation is performed.
        Make sure to always initialize the memory by feeding at least 1 interaction before using the
        model this way.
        """
        if user_ids is not None and item_ids is not None:
            return self.forward_edge_sequence(
                user_ids, user_features, item_ids, item_features
            )
        if user_ids is not None:
            return self.extract_user_embeddings(user_ids)
        if item_ids is not None:
            return self.extract_item_embeddings(item_ids)
        raise ValueError("Invalid input, provide user, item, or both.")

    def forward_edge_sequence(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor,
        item_ids: torch.Tensor,
        item_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence of incomming edges and updates the memory for the interacting nodes.

        Arguments:
        user_ids: (batch_size) tensor.
        item_ids: (batch_size) tensor.
        user_features: (batch_size, nb_user_features) tensor. nb_user_features can be 0.
        item_features: (batch_size, nb_item_features) tensor. nb_item_features can be 0.

        Returns:
        user_embeddings: (batch_size, embedding_size) tensor.
        item_embeddings: (batch_size, embedding_size) tensor.
        """
        batch_size = user_ids.shape[0]
        assert list(item_ids.shape) == [
            batch_size
        ], f"{item_ids.shape=}, {self.item_memory.shape=}"
        user_embeddings = self.user_memory[torch.arange(batch_size), user_ids, :]
        item_embeddings = self.item_memory[torch.arange(batch_size), item_ids, :]
        user_input = torch.hstack(
            (user_embeddings, user_features, item_embeddings, item_features)
        ).to(torch.float32)
        item_input = torch.hstack(
            (item_embeddings, item_features, user_embeddings, user_features)
        ).to(torch.float32)

        new_user_embeddings = torch.nn.functional.normalize(self.user_cell(user_input), dim=1)
        new_item_embeddings = torch.nn.functional.normalize(self.item_cell(item_input), dim=1)

        self.user_memory[torch.arange(batch_size), user_ids, :] = new_user_embeddings
        self.item_memory[torch.arange(batch_size), item_ids, :] = new_item_embeddings
        return new_user_embeddings, new_item_embeddings

    def extract_user_embeddings(self, users: torch.Tensor) -> torch.Tensor:
        """Return the current state of the memory for the given users.

        Arguments:
        users: (batch_size, nb_users_to_extract) tensor.
        Returns: (batch_size, nb_users_to_extract, embedding_size) tensor.
        """
        batch_size, _ = users.shape
        assert (
            batch_size == self.user_memory.shape[0]
        ), f"shape mismatch: {batch_size} as input while {self.user_memory.shape[0]} for memory."
        return self.user_memory[torch.arange(users.shape[0]).unsqueeze(1), users, :]

    def extract_item_embeddings(self, items: torch.Tensor) -> torch.Tensor:
        """Return the current state of the memory for the given items.

        Arguments:
        items: (batch_size, nb_items_to_extract) tensor.

        Returns: (batch_size, nb_items_to_extract, embedding_size) tensor.
        """
        batch_size, _ = items.shape
        assert (
            batch_size == self.item_memory.shape[0]
        ), f"shape mismatch: {batch_size} as input while {self.item_memory.shape[0]} for memory."
        return self.item_memory[torch.arange(items.shape[0]).unsqueeze(1), items, :]
