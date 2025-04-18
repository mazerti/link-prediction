"""Metrics and loss functions."""

import torch


def l2_mrr(
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    expected_item_ids: torch.Tensor,
) -> float:
    """Computes the mean reciprocal rank metric.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, nb_items, embedding_size) tensor
    expected_item_ids: (batch_size) tensor
    """
    assert len(user_embedding.shape) == 2, f"{user_embedding.shape=}"
    assert len(item_embeddings.shape) == 3, f"{item_embeddings.shape=}"
    l2_distance_fn = torch.nn.PairwiseDistance(2)
    scores = l2_distance_fn(user_embedding.unsqueeze(1), item_embeddings)
    return _mean_reciprocal_rank(expected_item_ids, scores)


def dot_product_mrr(
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    expected_item_ids: torch.Tensor,
) -> float:
    """Computes the mean reciprocal rank metric.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, nb_items, embedding_size) tensor
    expected_item_ids: (batch_size) tensor
    """
    assert len(user_embedding.shape) == 2, f"{user_embedding.shape=}"
    assert len(item_embeddings.shape) == 3, f"{item_embeddings.shape=}"
    scores = torch.einsum("bd,bid->bi", user_embedding, item_embeddings)
    return _mean_reciprocal_rank(expected_item_ids, scores)


def _mean_reciprocal_rank(
    expected_item_ids: torch.Tensor, scores: torch.Tensor
) -> float:
    """Computes the mean reciprocal rank metric.

    Can not be directly used as a metric, need to be combined with a scoring function (see l2_mrr or
    dot_product_mrr).

    Arguments:
    expected_item_ids: (batch_size) tensor.
    scores: (batch_size, nb_items) tensor.
    """
    _, ranked_items = scores.sort(dim=-1, descending=True)
    true_item_rank = (ranked_items == expected_item_ids.unsqueeze(1)).nonzero(
        as_tuple=True
    )[-1] + 1
    return (1.0 / true_item_rank).float().mean().item()


def dot_product_mse(user_embeddings: torch.Tensor, item_embeddings: torch.Tensor):
    """Loss function taking normalized embeddings and trying to maximise their dot product.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, embedding_size) tensor
    """
    loss_fn = torch.nn.MSELoss()
    pred = torch.einsum("bd,bd->b", user_embeddings, item_embeddings)
    loss = loss_fn(pred, torch.ones_like(pred))
    return loss


def expressivity_loss(user_embeddings: torch.Tensor, item_embeddings: torch.Tensor):
    """Loss rewarding a bigger distance inbetween two items or two users embeddings.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, embedding_size) tensor
    """
    user_embeddings_matrix = user_embeddings
    item_embeddings_matrix = item_embeddings
    return (
        torch.matmul(
            user_embeddings_matrix, user_embeddings_matrix.transpose(dim0=0, dim1=1)
        ).mean()
        + torch.matmul(
            item_embeddings_matrix, item_embeddings_matrix.transpose(dim0=0, dim1=1)
        ).mean()
    )


def frobenius_regularization(
    user_embeddings: torch.Tensor, item_embeddings: torch.Tensor
):
    """Loss rewarding a bigger distance inbetween two embeddings.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, embedding_size) tensor
    """
    embeddings = torch.cat((user_embeddings, item_embeddings), dim=-1).unsqueeze(-2)
    regularization_matrix = torch.matmul(
        embeddings.transpose(-1, -2), embeddings
    ) - torch.eye(embeddings.shape[-1])
    return regularization_matrix.norm()


def mean_squared_error(user_embeddings: torch.Tensor, item_embeddings: torch.Tensor):
    """Loss function taking embeddings and trying to minimise their l2 distance.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, embedding_size) tensor
    """
    loss_fn = torch.nn.MSELoss()
    return loss_fn(user_embeddings, item_embeddings)
