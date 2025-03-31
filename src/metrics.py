"""Metrics and loss functions."""

import torch


def mean_reciprocal_rank(
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    expected_item_ids: torch.Tensor,
) -> float:
    """Computes the mean reciprocal rank metric."""
    scores = torch.einsum("be,ie->bi", user_embedding, item_embeddings)
    _, ranked_items = scores.sort(dim=-1, descending=True)
    true_item_rank = (ranked_items == expected_item_ids.unsqueeze(1)).nonzero(
        as_tuple=False
    )[:, 1] + 1
    return 1.0 / true_item_rank.float().mean().item()


def dot_product_mse(user_embeddings: torch.Tensor, item_embeddings: torch.Tensor):
    """Loss function taking normalized embeddings and trying to maximise their dot product."""
    loss_fn = torch.nn.MSELoss()
    pred = torch.einsum("ij,ij->i", user_embeddings, item_embeddings)
    loss = loss_fn(pred, torch.ones_like(pred))
    return loss


def expressivity_loss(user_embeddings: torch.Tensor, item_embeddings: torch.Tensor):
    """Loss rewarding a bigger distance inbetween two items or two users embeddings."""
    return (
        torch.matmul(user_embeddings, user_embeddings.transpose(dim0=0, dim1=1)).mean()
        + torch.matmul(
            item_embeddings, item_embeddings.transpose(dim0=0, dim1=1)
        ).mean()
    )
