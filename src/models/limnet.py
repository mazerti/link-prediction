"""Cross RNN embeddings described in 'LiMNet: Early-Stage Detection of IoT Botnets
with Lightweight Memory Networks' by L.Giaretta et al."""

from collections.abc import Callable

import torch
from tqdm import tqdm

from context import Context
from metrics import compute_metrics


class LiMNet(torch.nn.Module):
    """Own implementation of the Lightweight Memory Network."""

    def __init__(
        self,
        embedding_size: int,
        initialization: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        normalize=True,
        dropout_rate=0,
        nb_layers: int = 1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_size: int = embedding_size
        self.nb_layers = nb_layers
        self.initialization: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
            getattr(torch.nn.init, initialization)
        )
        self.normalize = normalize
        self.dropout_rate = dropout_rate

        self.user_memory: torch.Tensor = None
        self.item_memory: torch.Tensor = None
        self.cross_rnn: torch.nn.Sequential = None

        # Arguments defined later
        self.nb_users: int
        self.nb_items: int
        self.device: torch.DeviceObjType

    def build(self, context: Context) -> None:
        """Builds the model."""
        self.nb_users = context.nb_users
        self.nb_items = context.nb_items
        self.device = context.device

        self.cross_rnn = torch.nn.Sequential(
            LimnetLayer(
                embedding_size=self.embedding_size,
                user_feature_size=len(context.user_features),
                item_feature_size=len(context.item_features),
                device=context.device,
                dropout_rate=self.dropout_rate,
                normalize=self.normalize,
            )
        )
        for _ in range(self.nb_layers - 1):
            self.cross_rnn.append(
                LimnetLayer(
                    embedding_size=self.embedding_size,
                    user_feature_size=self.embedding_size,
                    item_feature_size=self.embedding_size,
                    cell=torch.nn.GRUCell,
                    activation= torch.nn.LeakyReLU,
                    device=context.device,
                    dropout_rate=self.dropout_rate,
                    normalize=self.normalize,
                )
            )

    def initialize_batch_run(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize the model's memory.

        Returns:
        user_memory: (batch_size, nb_users, embedding_size) tensor.
        item_memory: (batch_size, nb_items, embedding_size) tensor.
        """
        for layer in self.cross_rnn.modules():
            if isinstance(layer, LimnetLayer):
                layer.user_memory = self.initialize_memory(batch_size, self.nb_users)
                user_memory = layer.user_memory
                layer.item_memory = self.initialize_memory(batch_size, self.nb_items)
                item_memory = layer.item_memory
        self.user_memory, self.item_memory = user_memory, item_memory
        return user_memory, item_memory

    def initialize_memory(self, batch_size: int, memory_size: int):
        """Return a resetted memory with requested size.

        :returns memory: (batch_size, memory_size, embedding_size) tensor."""
        memory = torch.zeros(
            (batch_size, memory_size, self.embedding_size), device=self.device
        )
        self.initialization(memory)
        return memory

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
        inputs = torch.hstack(
            (
                user_ids.unsqueeze(1),
                item_ids.unsqueeze(1),
                user_features,
                item_features,
            )
        ).to(torch.float32)

        new_embeddings = self.cross_rnn(inputs)
        new_user_embeddings = new_embeddings[:, 2: 2 + self.embedding_size]
        new_item_embeddings = new_embeddings[:, 2 + self.embedding_size :]

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

    def heat_up(
        self: torch.nn.Module,
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
        self.eval()
        for i in tqdm(range(length), desc="Heat up", leave=False):
            users, items = user_sequences[:, i], item_sequences[:, i]
            user_ids, item_ids = users[:, 0].to(torch.int32), items[:, 0].to(
                torch.int32
            )
            user_features, item_features = users[:, 1:], items[:, 1:]
            self.forward(
                user_ids=user_ids,
                user_features=user_features,
                item_ids=item_ids,
                item_features=item_features,
            )

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
        self.heat_up(
            user_sequences, item_sequences, length=context.train_heat_up_length
        )
        self.train()
        for i in range(context.train_heat_up_length, sequence_size):
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
            loss += loss_fn(context, user_embeddings, item_embeddings)
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
        self.heat_up(user_sequences, item_sequences, length=context.test_heat_up_length)
        test_loss = 0
        for i in tqdm(
            range(context.test_heat_up_length, context.test_sequence_length),
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
        test_loss = loss_fn(context, user_embedding, expected_item_embeddings)
        compute_metrics(
            context,
            context.metrics,
            measures,
            item_ids,
            user_embedding,
            item_embeddings,
        )
        # Communicate the interaction to the model for memory updates.
        self.forward(
            user_ids=user_ids,
            user_features=user_features,
            item_ids=item_ids,
            item_features=item_features,
        )
        return test_loss


class LimnetLayer(torch.nn.Module):
    """One layer of cross-GRU"""

    def __init__(
        self,
        embedding_size: int,
        cell: torch.nn.Module = torch.nn.GRUCell,
        user_feature_size: int = 0,
        item_feature_size: int = 0,
        device: torch.DeviceObjType = "cpu",
        dropout_rate: int = 0,
        normalize=True,
        activation: torch.nn.Module = torch.nn.Identity,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embedding_size: int = embedding_size
        self.user_feature_size: int = user_feature_size
        self.item_feature_size: int = item_feature_size
        self.normalize: bool = normalize

        self.user_memory: torch.Tensor = None
        self.item_memory: torch.Tensor = None

        input_size: int = (
            self.embedding_size
            + self.user_feature_size
            + self.embedding_size
            + self.item_feature_size
        )
        self.user_cell: torch.nn.Module = torch.nn.Sequential(
            activation(),
            cell(input_size, self.embedding_size, device=device),
            torch.nn.Dropout(dropout_rate),
        )
        self.item_cell: torch.nn.Module = torch.nn.Sequential(
            activation(),
            cell(input_size, self.embedding_size, device=device),
            torch.nn.Dropout(dropout_rate),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """

        :param inputs:
            (batch_size, embedding_size + user_feature + embedding_size + item_features)
            tensor. First half is user embedding and features, second half is for item embeddings
            and features.

        :return:
            (batch_size, embedding_size + embedding_size) tensor, first half is user embedding,
            second half is item.
        """
        batch_size = inputs.size(0)
        user_ids, item_ids, user_features, item_features = self.extract_inputs(inputs)
        user_memory = self.user_memory[torch.arange(batch_size), user_ids]
        item_memory = self.item_memory[torch.arange(batch_size), item_ids]

        user_inputs = torch.hstack(
            (user_memory, user_features, item_memory, item_features)
        )
        item_inputs = torch.hstack(
            (item_memory, item_features, user_memory, user_features)
        )

        new_user_embeddings = self.user_cell(user_inputs)
        new_item_embeddings = self.item_cell(item_inputs)
        if self.normalize:
            new_user_embeddings = torch.nn.functional.normalize(
                new_user_embeddings, dim=1
            )
            new_item_embeddings = torch.nn.functional.normalize(
                new_item_embeddings, dim=1
            )

        self.user_memory[torch.arange(batch_size), user_ids, :] = new_user_embeddings
        self.item_memory[torch.arange(batch_size), item_ids, :] = new_item_embeddings

        return torch.hstack(
            (
                user_ids.unsqueeze(1),
                item_ids.unsqueeze(1),
                new_user_embeddings,
                new_item_embeddings,
            )
        )

    def extract_inputs(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features and ids from the inputs.

        :param inputs: (batch_size, 1 + 1 user_features_size + item_features_size) tensor.

        :return user_ids: (batch_size) tensor.
        :return item_ids: (batch_size) tensor.
        :return user_features: (batch_size, user_features_size) tensor.
        :return item_features: (batch_size, item_features_size) tensor."""
        user_ids = inputs[:, 0].to(dtype=torch.long)
        item_ids = inputs[:, 1].to(dtype=torch.long)
        user_features = inputs[:, 2 : 2 + self.user_feature_size]
        item_features = inputs[:, 2 + self.user_feature_size :]
        return user_ids, item_ids, user_features, item_features
