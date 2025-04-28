"""Cross RNN architecture described in 'Predicting Dynamic Embedding Trajectory in Temporal
Interaction Networks' by S.Kumar et al."""

from collections.abc import Callable

import torch
from tqdm import tqdm

from context import Context
from metrics import compute_metrics


class Jodie(torch.nn.Module):
    """Own implementation of the Jodie model."""

    def __init__(self, embedding_size: int, **kwargs):
        super().__init__(**kwargs)

        self.embedding_size: int = embedding_size

        self.requested_features: set[str] = set(("delta_users", "delta_items"))

        # Must be initialized for each batch
        self.dynamic_user_embeddings: torch.Tensor
        self.dynamic_item_embeddings: torch.Tensor

    def build(self, settings: Context) -> None:
        """Builds the model, using latest knowledge about the dataset."""
        self.nb_users: int = settings.nb_users
        self.nb_items: int = settings.nb_items
        self.device: torch.DeviceObjType = settings.device

        self.user_long_term_embeddings: torch.nn.Embedding = (
            lambda x: torch.nn.functional.one_hot(x, num_classes=settings.nb_users)
        )
        self.item_long_term_embeddings: torch.nn.Embedding = (
            lambda x: torch.nn.functional.one_hot(x, num_classes=settings.nb_items)
        )
        rnn_input_size: int = (
            self.embedding_size
            + len(settings.user_features)
            + self.embedding_size
            + len(settings.item_features)
        )
        self.user_rnn: torch.nn.Linear = torch.nn.Linear(
            rnn_input_size, self.embedding_size, device=settings.device
        )
        self.item_rnn: torch.nn.Linear = torch.nn.Linear(
            rnn_input_size, self.embedding_size, device=settings.device
        )
        self.activation: torch.nn.Sigmoid = torch.nn.Sigmoid()

        self.time_context_layer: torch.nn.Linear = torch.nn.Linear(
            1, self.embedding_size, device=settings.device
        )
        prediction_input_size: int = (
            self.embedding_size + self.nb_users + self.embedding_size + self.nb_items
        )
        self.prediction_layer: torch.nn.Linear = torch.nn.Linear(
            prediction_input_size, self.embedding_size, device=settings.device
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
        This function will return the embeeddings for the given users and items and update the
        memory.

        When called with only users or only items, the expected input are
            users: (batch_size, nb_users_to_extract) tensor.
            or
            items: (batch_size, nb_items_to_extract) tensor.
        This will simply return the current state of the memory for the requested users/items.
        Make sure to always initialize the memory by feeding at least 1 interaction before using the
        model this way.
        """
        if user_ids is not None and item_ids is not None:
            return self.forward_edge_sequence(
                user_ids, user_features, item_ids, item_features
            )
        if user_ids is not None:
            return self.extract_user_embeddings(user_ids, user_features)
        if item_ids is not None:
            return self.extract_item_embeddings(item_ids, item_features)
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
        new_user_embeddings: (batch_size, embedding_size) tensor.
        previous_user_embeddings: (batch_size, embedding_size) tensor.
        new_item_embeddings: (batch_size, embedding_size) tensor.
        predicted_item_embeddings: (batch_size, embedding_size) tensor.
        previous_item_embeddings: (batch_size, embedding_size) tensor.
        """
        batch_size = user_ids.shape[0]

        previous_user_embeddings = self.user_memory[
            torch.arange(batch_size), user_ids, :
        ]
        previous_item_embeddings = self.item_memory[
            torch.arange(batch_size), item_ids, :
        ]
        user_projections = self.embedding_projection(
            previous_user_embeddings, user_features
        )
        predicted_item_embeddings = self.prediction_layer(
            torch.hstack(
                (
                    user_projections,
                    self.user_long_term_embeddings(user_ids),
                    previous_item_embeddings,
                    self.item_long_term_embeddings(item_ids),
                )
            )
        )

        new_user_embeddings, new_item_embeddings = self.embedding_update(
            user_ids, user_features, item_ids, item_features
        )
        return (
            new_user_embeddings,
            previous_user_embeddings,
            new_item_embeddings,
            predicted_item_embeddings,
            previous_item_embeddings,
        )

    def embedding_projection(
        self,
        dynamic_embedding: torch.Tensor,
        time_delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Projects the embedding on its trajectory after a time delta.

        :Argument dynamic_embedding: (batch_size, embedding_size) tensor.
        :Argument time_delta: (batch_size) tensor.

        :Returns projected_embedding: (batch_size, embedding_size) tensor.
        """
        time_context = self.time_context_layer(time_delta)
        return (1 + time_context) * dynamic_embedding

    def embedding_update(
        self,
        user_ids: torch.Tensor,
        user_features: torch.Tensor,
        item_ids: torch.Tensor,
        item_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the dynamic embeddings of the interacting user and item.

        :Argument user_ids: (batch_size) tensor.
        :Argument item_ids: (batch_size) tensor.
        :Argument user_features: (batch_size, nb_user_features) tensor. nb_user_features can be 0.
        :Argument item_features: (batch_size, nb_item_features) tensor. nb_item_features can be 0.

        :Returns user_embeddings: (batch_size, embedding_size) tensor.
        :Returns item_embeddings: (batch_size, embedding_size) tensor.
        """
        batch_size = user_ids.shape[0]
        user_dynamic_embeddings = self.user_memory[
            torch.arange(batch_size), user_ids, :
        ]
        item_dynamic_embeddings = self.item_memory[
            torch.arange(batch_size), item_ids, :
        ]
        user_delta = user_features[:, 0]
        item_delta = item_features[:, 0]
        interaction_features = user_features[:, 1:]
        user_input = torch.hstack(
            (
                user_dynamic_embeddings,
                item_dynamic_embeddings,
                interaction_features,
                user_delta,
            )
        ).to(torch.float32)
        item_input = torch.hstack(
            (
                item_dynamic_embeddings,
                user_dynamic_embeddings,
                interaction_features,
                item_delta,
            )
        ).to(torch.float32)

        new_user_embeddings = self.activation(self.user_rnn(user_input))
        new_item_embeddings = self.activation(self.item_rnn(item_input))

        self.user_memory[torch.arange(batch_size), user_ids, :] = new_user_embeddings
        self.item_memory[torch.arange(batch_size), item_ids, :] = new_item_embeddings
        return new_user_embeddings, new_item_embeddings

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
