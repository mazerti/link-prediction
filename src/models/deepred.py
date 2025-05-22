"""Cross RNN embeddings described in 'Dynamic Embeddings for Interaction Prediction' by Z. Kefato et
al."""

from collections.abc import Callable

import torch
from tqdm import tqdm

from context import Context
from metrics import compute_metrics


class DeePRed(torch.nn.Module):
    """Own implementation of the Dynamic Embeddings for Interaction Prediction."""

    def __init__(
        self,
        embedding_size: int,
        history_size: int,
        pooling: str,
        dropout_rate: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_size: int = embedding_size
        self.history_size = history_size
        self.pooling = pooling
        self.dropout_rate = dropout_rate

        self.requested_features = set(
            (f"user_last_{history_size}", f"item_last_{history_size}")
        )

        # Arguments defined later
        self.nb_users: int
        self.nb_items: int
        self.device: torch.DeviceObjType
        self.user_long_term_embeddings: torch.nn.Embedding
        self.item_long_term_embeddings: torch.nn.Embedding
        self.user_short_term_embeddings: torch.Tensor
        self.item_short_term_embeddings: torch.Tensor
        self.encoder: Encoder

    def build(self, context: Context) -> None:
        """Builds the model."""
        self.nb_users = context.nb_users
        self.nb_items = context.nb_items
        self.device = context.device

        self.user_long_term_embeddings = torch.nn.Embedding(
            self.nb_users + 1, self.embedding_size, device=self.device
        )  # + 1 to account for NaNs
        self.item_long_term_embeddings = torch.nn.Embedding(
            self.nb_items + 1, self.embedding_size, device=self.device
        )  # + 1 to account for NaNs

        self.encoder = Encoder(
            self.embedding_size,
            self.history_size,
            self.device,
            self.dropout_rate,
        )

    # pylint: disable=locally-disabled, unused-argument
    def initialize_batch_run(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize the model's memory."""
        self.user_short_term_embeddings = torch.zeros(
            self.nb_users, 2 * self.embedding_size, device=self.device
        )
        self.item_short_term_embeddings = torch.zeros(
            self.nb_items, 2 * self.embedding_size, device=self.device
        )

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
        user_history_embedding: torch.Tensor = self.encoder(
            user_features, self.item_long_term_embeddings
        )  # (batch_size, history_size, embedding_size)
        item_history_embedding: torch.Tensor = self.encoder(
            item_features, self.user_long_term_embeddings
        )  # (batch_size, history_size, embedding_size)

        alignment_scores: torch.Tensor = torch.tanh(
            user_history_embedding.matmul(item_history_embedding.transpose(-1, -2))
        )  # (batch_size, history_size, history_size)

        user_attention, item_attention = self.pool(alignment_scores)

        user_attention = torch.softmax(user_attention, dim=1)
        item_attention = torch.softmax(item_attention, dim=1)

        user_short_term_embeddings = (
            user_history_embedding * user_attention.unsqueeze(-1)
        ).sum(dim=1)
        item_short_term_embeddings = (
            item_history_embedding * item_attention.unsqueeze(-1)
        ).sum(dim=1)

        self.user_short_term_embeddings[user_ids] = user_short_term_embeddings
        self.item_short_term_embeddings[item_ids] = item_short_term_embeddings
        return user_short_term_embeddings, item_short_term_embeddings

    def pool(self, alignment_scores: torch.Tensor) -> torch.Tensor:
        """Apply pooling on the attention (alignment) scores.

        :Arguments:
            alignment_scores: (batch_size, history_size, history_size), the first history_size
            dimension corresponds to the users, the second to the items.
        :Returns:
            user_attention: (batch_size, history_size) tensor.
            item_attention: (batch_size, history_size) tensor.
        """
        if self.pooling == "mean":
            return (
                alignment_scores.mean(dim=-1),
                alignment_scores.mean(dim=-2),
            )
        elif self.pooling == "max":
            return (
                alignment_scores.max(dim=-1),
                alignment_scores.max(dim=-2),
            )
        raise NotImplementedError("Requested pooling strategy is not supported.")

    def extract_user_embeddings(self, users: torch.Tensor) -> torch.Tensor:
        """Return the current state of the memory for the given users.

        Arguments:
        users: (batch_size, nb_users_to_extract) tensor.
        Returns: (batch_size, nb_users_to_extract, embedding_size) tensor.
        """
        return self.user_short_term_embeddings[users]

    def extract_item_embeddings(self, items: torch.Tensor) -> torch.Tensor:
        """Return the current state of the memory for the given items.

        Arguments:
        items: (batch_size, nb_items_to_extract) tensor.

        Returns: (batch_size, nb_items_to_extract, embedding_size) tensor.
        """
        return self.item_short_term_embeddings[items]

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
        self.train()
        users, items = user_sequences.flatten(0, 1), item_sequences.flatten(0, 1)
        user_ids, item_ids = users[:, 0].to(torch.int32), items[:, 0].to(torch.int32)
        user_features, item_features = users[:, 1:], items[:, 1:]
        user_embeddings = self.forward(user_ids=user_ids, user_features=user_features)
        item_embeddings = self.forward(item_ids=item_ids, item_features=item_features)
        loss = loss_fn(context, user_embeddings, item_embeddings) / sequence_size
        loss.backward()
        self.forward(
            user_ids=user_ids,
            user_features=user_features,
            item_ids=item_ids,
            item_features=item_features,
        )
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


class Encoder(torch.nn.Module):
    """Load the context/history of each interaction and feed it into a GRU."""

    def __init__(
        self,
        embedding_size: int,
        history_size: int,
        device: torch.DeviceObjType,
        dropout_rate: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.history_size = history_size
        self.device = device
        self.dropout = torch.nn.Dropout(dropout_rate)

        input_size = embedding_size + 1
        self.gru = torch.nn.GRU(
            input_size,
            embedding_size,
            batch_first=True,
            device=device,
            bidirectional=True,
        )

    def forward(
        self, raw_features: torch.Tensor, embeddings: torch.nn.Embedding
    ) -> torch.Tensor:
        """Return high level representation of the history context of the input user/item.

        :Arguments:
            features: (batch_size, 2 * history_size) tensor. For each batch, it contains a list of
                (node_id, time_delta) for each history step.
            embeddings: Long terms embeddings of the matching type of nodes.
        :Returns:
            (batch_size, history_size, embedding_size) tensor.
        """
        history_sequence = self.extract_history_features(raw_features, embeddings)
        output_features, _ = self.gru(history_sequence)
        return output_features

    def extract_history_features(
        self, raw_features: torch.Tensor, embeddings: torch.nn.Embedding
    ) -> torch.Tensor:
        """Process the raw features (ids) into a sequence of high level features.

        :Arguments:
            features: (batch_size, 2 * history_size) tensor. For each batch, it contains a list of
                (node_id, time_delta) for each history step.
            embeddings: Long terms embeddings of the matching type of nodes.
        :Returns:
            (batch_size, history_size, embedding_size + 1)
        """
        batch_size, k2 = raw_features.shape
        assert (
            k2 % 2 == 0
        ), f"Features don't match the expected shape of k * [node_id, delta]\n{raw_features}"
        raw_features = raw_features.reshape(batch_size, k2 // 2, 2)
        node_ids = (
            raw_features[:, :, 0].to(torch.long) + 1
        )  # shift all indices by 1 so that nan end up on 0
        deltas = raw_features[:, :, 1]

        node_embeddings = embeddings(node_ids)
        node_embeddings = self.dropout(node_embeddings)

        return torch.cat((node_embeddings, deltas.unsqueeze(-1)), dim=-1).to(
            torch.float32
        )
