"""List all the function regarding data extraction and preparation."""

# standard imports
import re

# third party imports
import pandas as pd

# first party imports
import torch

from context import Context


def prepare_dataset(df: pd.DataFrame, context: Context) -> torch.utils.data.DataLoader:
    """Prepare the raw data for the training.

    Include, reingexing of the ids, features computation, split into train/test sets, and loading
    the data into a torch DataLoader.
    """
    df = reindex_ids(df, context)
    df = compute_features(df, context)
    train_df, test_df = train_test_split(df, context.train_ratio, context)
    return torch.utils.data.DataLoader(
        TemporalInteractionNetworkDataset(
            df=train_df,
            user_ids=context.user_id_column,
            user_features=context.user_features,
            item_ids=context.item_id_column,
            item_features=context.item_features,
            timestamps=context.timestamp_column,
            sequence_length=context.train_sequence_length,
            sequence_stride=context.train_sequence_stride,
            device=context.device,
        ),
        batch_size=context.train_batch_size,
        shuffle=False,
    ), torch.utils.data.DataLoader(
        TemporalInteractionNetworkDataset(
            df=test_df,
            user_ids=context.user_id_column,
            user_features=context.user_features,
            item_ids=context.item_id_column,
            item_features=context.item_features,
            timestamps=context.timestamp_column,
            sequence_length=context.test_sequence_length,
            sequence_stride=context.test_sequence_stride,
            device=context.device,
        ),
        batch_size=context.test_batch_size,
        shuffle=False,
    )


def reindex_ids(data: pd.DataFrame, context: Context) -> pd.DataFrame:
    """Rename the ids in the data so that all users and items ids are indexed from 0."""
    data[context.user_id_column] = reindex_serie(data[context.user_id_column])
    data[context.item_id_column] = reindex_serie(data[context.item_id_column])
    return data


def reindex_serie(serie: pd.Series):
    """Return the serie of indices from 0 corresponding to the input serie."""
    return serie.map({item: idx for idx, item in enumerate(serie.unique())})


def compute_features(data: pd.DataFrame, context: Context) -> pd.DataFrame:
    """Compute the features to input to the models."""
    for feature in context.requested_features:
        if feature == "delta_users":
            data = compute_deltas(
                data,
                id_column=context.user_id_column,
                timestamp_column=context.timestamp_column,
                column_name="delta_users",
            )
            context.user_features.append("delta_users")
        elif feature == "delta_items":
            data = compute_deltas(
                data,
                id_column=context.item_id_column,
                timestamp_column=context.timestamp_column,
                column_name="delta_items",
            )
            context.item_features.append("delta_items")
        elif re.match(r"user_last_(\d+)", feature):
            k = int(re.findall(r"user_last_(\d+)", feature)[0])
            data, columns = add_last_k(
                data,
                id_column=context.user_id_column,
                columns_prefix=[f"user_last_{i}" for i in range(1, k + 1)],
                feature=context.item_id_column,
                context=context,
            )
            context.user_features = context.user_features + columns
        elif re.match(r"item_last_(\d+)", feature):
            k = int(re.findall(r"item_last_(\d+)", feature)[0])
            data, columns = add_last_k(
                data,
                id_column=context.item_id_column,
                columns_prefix=[f"item_last_{i}" for i in range(1, k + 1)],
                feature=context.user_id_column,
                context=context,
            )
            context.item_features = context.item_features + columns
        else:
            raise NotImplementedError("Requested features are not implemented yet.")
    data.sort_index(inplace=True)
    return data


def compute_deltas(
    data: pd.DataFrame, id_column: str, timestamp_column: str, column_name: str
) -> pd.DataFrame:
    """Compute for each interaction the time since the last time the user/item interacted.

    Arguments:
    id_column: name of the column containing the user/item ids, depending on which one to compute.
    timestamp_column: name of the column containing interaction timestamps.
    column_name: name of the column containing the computed deltas.
    """
    data = data.sort_values(by=[id_column, timestamp_column])
    data[column_name] = (
        data[timestamp_column].diff().apply(lambda x: 0 if pd.isna(x) or x < 0 else x)
    )
    return data.sort_index()


def add_last_k(
    data: pd.DataFrame,
    id_column: str,
    columns_prefix: list[str],
    feature: str,
    context: Context,
) -> tuple[pd.DataFrame, list[str]]:
    """Find the last k interactions for the user/item at each interaction and add them.

    k is given as the length of the columns_prefix list.

    Arguments:
    id_column: name of the column containing the user/item ids, depending on which one to compute.
    columns_prefix: prefix of the history columns.
    interaction_features: list of features to copy as history features.
    """
    data = data.sort_values(by=[id_column, context.timestamp_column])
    feature_columns = []
    dfs = [data]
    for i, prefix in enumerate(columns_prefix):
        dfs.append(
            data.groupby(id_column)[feature]
            .shift(i + 1)
            .fillna(-1)
            .rename(f"{prefix}_{feature}")
            .to_frame()
        )
        feature_columns.append(f"{prefix}_{feature}")

        dfs.append(
            (
                data[context.timestamp_column]
                - data.groupby(id_column)[context.timestamp_column].shift(i + 1)
            )
            .fillna(0)
            .rename(f"{prefix}_{feature}_delta")
            .to_frame()
        )
        feature_columns.append(f"{prefix}_{feature}_delta")
    data = pd.concat(dfs, axis=1)
    return data, feature_columns


def train_test_split(
    df: pd.DataFrame, train_size: float, settings: Context
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a temporal split of the data."""
    split_at = int(len(df) * train_size)
    df = df.sort_values(by=settings.timestamp_column).reset_index()
    return df[:split_at], df[split_at:].reset_index()


class TemporalInteractionNetworkDataset(torch.utils.data.Dataset):
    """
    Torch Dataset for temporal interaction network.

    Each row contain a (users, items), pair.
        users: (batch_size, sequence_length, 1 + nb_user_features).
        items: (batch_size, sequence_length, 1 + nb_item_features).
    Where users[:,:,0] and items[:,:,0] are the users/items ids
    and users[:,:,1:] and items[:,:,1:] are the users/items features (there can be none).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        user_ids: str,
        user_features: list[str],
        item_ids: str,
        item_features: list[str],
        timestamps: str,
        sequence_length: int,
        sequence_stride: int,
        device: torch.DeviceObjType,
    ):
        super().__init__()
        self.users = df[user_ids]
        self.user_features = df[user_features]
        self.items = df[item_ids]
        self.item_features = df[item_features]
        self.timestamps = df[timestamps]
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.device = device

    def __len__(self) -> int:
        return int((len(self.users) - self.sequence_length) / self.sequence_stride)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.sequence_stride * index
        stop = start + self.sequence_length
        user_ids = self.users[start:stop].to_numpy()
        user_features = self.user_features[start:stop].to_numpy()
        item_ids = self.items[start:stop].to_numpy()
        item_features = self.item_features[start:stop].to_numpy()
        return (
            torch.hstack(
                (
                    torch.tensor(user_ids, device=self.device).unsqueeze(1),
                    torch.tensor(user_features, device=self.device),
                )
            ),
            torch.hstack(
                (
                    torch.tensor(item_ids, device=self.device).unsqueeze(1),
                    torch.tensor(item_features, device=self.device),
                )
            ),
        )
