"""Metrics and loss functions."""

import pandas as pd
import torch
import wandb

from context import Context


def compute_metrics(
    context: Context,
    metrics_list: list[str],
    measures: dict[str:float],
    item_id: torch.Tensor,
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
):
    """
    Evaluates the metrics on the given embeddings and adds them to the past measures.

    :param metrics: the name of the metrics function as defined in the metric module.
    :param measures: the sums of the evaluations of the metrics over past data.
    :param item_id: (batch_size) tensor. Contains the id of the expected items.
    :param user_embeddings: (batch_size, embedding size) tensor.
    :param item_embeddings: (batch_size, nb_items, embedding size) tensor.
        Contains the embeddings for every single item.
    """
    for metric in metrics_list:
        measures[metric] = measures.get(metric, 0) + pick_metric(metric)(
            context, user_embedding, item_embeddings, item_id
        )


def _l2_scores(
    context: Context,
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Computes the mean reciprocal rank metric.

    :param: user_embeddings: (batch_size, embedding_size) tensor
    :param item_embeddings: (batch_size, nb_items, embedding_size) tensor

    :returns scores: (batch_size, nb_items) tensor.
    """
    l2_distance_fn = torch.nn.PairwiseDistance(2)
    return l2_distance_fn(user_embedding.unsqueeze(1), item_embeddings)


def _dot_product_scores(
    context: Context,
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Computes the mean reciprocal rank metric.

    :param user_embeddings: (batch_size, embedding_size) tensor
    :param item_embeddings: (batch_size, nb_items, embedding_size) tensor

    :returns scores: (batch_size, nb_items) tensor.
    """
    return torch.einsum("bd,bid->bi", user_embedding, item_embeddings)


def l2_mrr(
    context: Context,
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
    scores = _l2_scores(context, user_embedding, item_embeddings)
    return _mean_reciprocal_rank(context, expected_item_ids, scores, descending=False)


def dot_product_mrr(
    context: Context,
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
    scores = _dot_product_scores(context, user_embedding, item_embeddings)
    return _mean_reciprocal_rank(context, expected_item_ids, scores, descending=True)


def _mean_reciprocal_rank(
    context: Context,
    expected_item_ids: torch.Tensor,
    scores: torch.Tensor,
    descending: bool,
) -> float:
    """Computes the mean reciprocal rank metric.

    Can not be directly used as a metric, need to be combined with a scoring function (see l2_mrr or
    dot_product_mrr).

    :param expected_item_ids: (batch_size) tensor.
    :param scores: (batch_size, nb_items) tensor.
    :param descending: true if higher score are better.
    """
    true_item_rank = _compute_rank(context, expected_item_ids, scores, descending)
    with open(f"ranks{expected_item_ids.device}.txt", "+a") as f:
        torch.set_printoptions(profile="full")
        print(true_item_rank, file=f)
        torch.set_printoptions(profile="default")
    return (1.0 / true_item_rank).float().mean().item()


def l2_recall_at_1(
    context: Context,
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    expected_item_ids: torch.Tensor,
) -> float:
    """Computes the recall at k using l2 scoring.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, nb_items, embedding_size) tensor
    expected_item_ids: (batch_size) tensor
    """
    return _l2_recall_at_k(
        context, 1, user_embedding, item_embeddings, expected_item_ids
    )


def l2_recall_at_10(
    context: Context,
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    expected_item_ids: torch.Tensor,
) -> float:
    """Computes the recall at k using l2 scoring.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, nb_items, embedding_size) tensor
    expected_item_ids: (batch_size) tensor
    """
    return _l2_recall_at_k(
        context, 10, user_embedding, item_embeddings, expected_item_ids
    )


def _l2_recall_at_k(
    context: Context,
    k: int,
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    expected_item_ids: torch.Tensor,
) -> float:
    """Computes the recall at k using l2 scoring.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, nb_items, embedding_size) tensor
    expected_item_ids: (batch_size) tensor
    """
    l2_distance_fn = torch.nn.PairwiseDistance(2)
    scores = l2_distance_fn(user_embedding.unsqueeze(1), item_embeddings)
    return _recall_at_k(context, k, expected_item_ids, scores, descending=False)


def dot_product_recall_at_1(
    context: Context,
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    expected_item_ids: torch.Tensor,
) -> float:
    """Computes the recall at k using dot product scoring.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, nb_items, embedding_size) tensor
    expected_item_ids: (batch_size) tensor
    """
    return _dot_product_recall_at_k(
        context, 1, user_embedding, item_embeddings, expected_item_ids
    )


def dot_product_recall_at_10(
    context: Context,
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    expected_item_ids: torch.Tensor,
) -> float:
    """Computes the recall at k using dot product scoring.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, nb_items, embedding_size) tensor
    expected_item_ids: (batch_size) tensor
    """
    return _dot_product_recall_at_k(
        context, 10, user_embedding, item_embeddings, expected_item_ids
    )


def _dot_product_recall_at_k(
    context: Context,
    k: int,
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    expected_item_ids: torch.Tensor,
) -> float:
    """Computes the recall at k using dot product scoring.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, nb_items, embedding_size) tensor
    expected_item_ids: (batch_size) tensor
    """
    scores = torch.einsum("bd,bid->bi", user_embedding, item_embeddings)
    return _recall_at_k(context, k, expected_item_ids, scores, descending=True)


def _recall_at_k(
    context: Context,
    k: int,
    expected_item_ids: torch.Tensor,
    scores: torch.Tensor,
    descending: bool,
) -> float:
    """Computes the recall at k.

    Can not be directly used as a metric, need to be combined with a scoring function (see
    l2_recall_at_k or dot_product_recall_at_k).

    :param expected_item_ids: (batch_size) tensor.
    :param scores: (batch_size, nb_items) tensor.
    :param descending: true if higher score are better.
    """
    true_item_rank = _compute_rank(context, expected_item_ids, scores, descending)
    return (true_item_rank <= k).sum().item() / true_item_rank.shape[0]


def _compute_rank(
    context: Context,
    expected_item_ids: torch.Tensor,
    scores: torch.Tensor,
    descending: bool,
) -> torch.Tensor:
    """
    Computes the rank of the true items according to given scoring function.

    Can not be directly used as a metric.

    :param expected_item_ids: (batch_size) tensor.
    :param scores: (batch_size, nb_items) tensor.
    :param descending: true if higher score are better.

    :returns: (batch_size) tensor.
    """
    _, ranked_items = scores.sort(dim=-1, descending=descending)
    true_item_rank = (ranked_items == expected_item_ids.unsqueeze(1)).nonzero(
        as_tuple=True
    )[-1] + 1

    return true_item_rank


def dot_product_mse(
    context: Context, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor
):
    """Loss function taking normalized embeddings and trying to maximise their dot product.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, embedding_size) tensor
    """
    loss_fn = torch.nn.MSELoss()
    pred = torch.einsum("bd,bd->b", user_embeddings, item_embeddings)
    loss = loss_fn(pred, torch.ones_like(pred))
    return loss


def expressivity_loss(
    context: Context, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor
):
    """Loss rewarding a bigger distance inbetween two items or two users embeddings.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, embedding_size) tensor
    """
    return (
        torch.matmul(user_embeddings, user_embeddings.transpose(dim0=0, dim1=1)).mean()
        + torch.matmul(
            item_embeddings, item_embeddings.transpose(dim0=0, dim1=1)
        ).mean()
    )


def frobenius_regularization(
    context: Context, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor
):
    """Loss rewarding a bigger distance inbetween two embeddings.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, embedding_size) tensor
    """
    embeddings = torch.cat((user_embeddings, item_embeddings), dim=-1).unsqueeze(-2)
    regularization_matrix = torch.matmul(
        embeddings.transpose(-1, -2), embeddings
    ) - torch.eye(embeddings.shape[-1], device=embeddings.device)
    return regularization_matrix.norm(dim=(-1, -2)).mean()


def mean_squared_error(
    context: Context, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor
):
    """Loss function taking embeddings and trying to minimise their l2 distance.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, embedding_size) tensor
    """
    loss_fn = torch.nn.MSELoss()
    return loss_fn(user_embeddings, item_embeddings)


def plot_predictions(
    context: Context,
    user_embedding: torch.Tensor,
    item_embeddings: torch.Tensor,
    expected_item_id: torch.Tensor,
):
    """Plot as many information as possible on the prediction.

    This function will directly log the distances and the ranks to the W&B run.
    It is only meant to be used with an appropriate run.

    Arguments:
    user_embeddings: (batch_size, embedding_size) tensor
    item_embeddings: (batch_size, nb_items, embedding_size) tensor
    expected_item_ids: (batch_size) tensor
    """
    nb_batches, nb_items, _ = item_embeddings.shape

    l2_scores = _l2_scores(context, user_embedding, item_embeddings)
    dot_product_scores = _dot_product_scores(context, user_embedding, item_embeddings)

    store_results(context, expected_item_id, nb_items, l2_scores, dot_product_scores)

    return 0


def store_results(
    context: Context,
    expected_item_id: torch.Tensor,
    nb_items: int,
    l2_scores: torch.Tensor,
    dot_product_scores: torch.Tensor,
):
    """Store results in context for further analysis."""
    results = torch.stack(
        (
            l2_scores.flatten(),
            dot_product_scores.flatten(),
            _compute_ranks(l2_scores, descending=False).flatten(),
            _compute_ranks(dot_product_scores, descending=True).flatten(),
            torch.eye(nb_items, device=expected_item_id.device)[
                expected_item_id
            ].flatten(),
        ),
        dim=1,
    )
    context.results = pd.concat(
        (
            context.results,
            pd.DataFrame(
                results.detach().cpu(),
                columns=[
                    "l2-score",
                    "dot-product-score",
                    "l2-rank",
                    "dot-product-rank",
                    "label",
                ],
            ),
        )
    )


def _compute_ranks(scores: torch.Tensor, descending: bool) -> torch.Tensor:
    """Computes for each given score it's rank in it's row.

    :param scores: (batch_size, nb_items) tensor.
    :param descending: true if higher score are better.

    :returns: (batch_size, nb_items) tensor."""
    batch_size, nb_items = scores.shape
    _, sorted_indices = torch.sort(scores, descending=descending)
    ranks = torch.zeros_like(sorted_indices)
    ranks[
        torch.arange(batch_size, device=scores.device).unsqueeze(1), sorted_indices
    ] = torch.arange(1, nb_items + 1, device=scores.device)
    return ranks


def _plot_rank_histogram(ranks: torch.Tensor, name: str):
    """Plot the histogram from a single dimension tensor."""
    table = wandb.Table(data=ranks.unsqueeze(1).cpu().numpy(), columns=[name])
    return {f"{name}-hist": wandb.plot.histogram(table, name, title=name)}


def _plot_distribution(serie: torch.Tensor, name: str):
    """Plot the distribution of a serie."""
    sorted, _ = torch.sort(serie)
    sorted = sorted.mean(dim=0).unsqueeze(0).cpu()
    xs = torch.arange(1, sorted.shape[-1] + 1).expand_as(sorted)
    return {
        f"{name}-distribution": wandb.plot.line_series(
            xs=xs.numpy(), ys=sorted.detach().numpy(), title=name, xname="rank"
        )
    }


def pick_metric(metric_name: str) -> callable:
    """Select the implementation of the metric function matching the given function name."""
    return globals()[metric_name]
