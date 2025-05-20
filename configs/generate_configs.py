"""Generate the config files for the experiments."""

import os

# --- Constants ----
# Important! make sure that all the following constants are up to date with
# the configuration you want to use for the experiments.

DATA_FOLDER = "/data/gen-limnet"
DEFAULT_SEQUENCE_LENGTH = 1024

MEMORY_SIZE = 16_000_000_000


def config_common(
    config_name, dataset, embedding_size, sequence_length=DEFAULT_SEQUENCE_LENGTH
):
    return f"""name: "{config_name}"

dataset: {os.path.join(DATA_FOLDER, f"{dataset}.csv")}
user_id_column: user_id
item_id_column: item_id
timestamp_column: timestamp

epochs: 100
train_ratio: 0.8
train_sequence_length: {sequence_length}
train_sequence_stride: {sequence_length/3.5}
train_batch_size: {int(MEMORY_SIZE / (500 * sequence_length * embedding_size))}
test_sequence_length: {sequence_length}
test_sequence_stride: {sequence_length/3.5}
test_batch_size: {int(MEMORY_SIZE / (250 * sequence_length * embedding_size))}


metrics:
  - l2_mrr
  - dot_product_mrr
  - dot_product_recall_at_1
  - dot_product_recall_at_10
  - l2_recall_at_1
  - l2_recall_at_10
    """


def config_model(
    model,
    embedding_size,
    nb_layers=1,
    normalize=True,
    time_features=None,
    sequence_length=DEFAULT_SEQUENCE_LENGTH,
):
    if model == "LiMNet":
        return f"""
model_name: LiMNet
model_attributes:
  embedding_size: {embedding_size}
  initialization: xavier_normal_
  nb_layers: {nb_layers}
  normalize: {normalize}

train_heat_up_length: 0
test_heat_up_length: {int(0.5 * sequence_length)}

l2: 0.00001
learning_rate: 0.016

loss:
  dot_product_mse: {1 if normalize else 0.2}
  mean_squared_error: {0.2 if normalize else 1}
  expressivity_loss: 1

{"requested_features:\n" if time_features else ""}{"  - time_of_day\n" if time_features in ("day", "both") else ""}{"  - time_of_week\n" if time_features in ("week", "both") else ""}
lr_scheduler:
  - ExponentialLR:
      gamma: 1.25
      milestone: 0
  - ExponentialLR:
      gamma: 0.8
      milestone: 15
        """
    if model == "Jodie":
        return f"""
model_name: Jodie
model_attributes:
  embedding_size: {embedding_size}
loss:

l2: 0.00001
learning_rate: 0.01
    """
    if model == "StaticEmbeddings":
        return f"""
model_name: TrainableEmbeddings
model_attributes:
  embedding_size: {embedding_size}
  normalize: {normalize}
loss:
  {"dot_product_mse" if normalize else "mean_squared_error"}: 1
  expressivity_loss: 1

l2: 0
learning_rate: 0.1
"""


counter = 1


def create_config(
    config_name,
    folder,
    model,
    dataset,
    embedding_size=32,
    normalize=True,
    time_features=None,
    nb_layers=1,
    sequence_length=DEFAULT_SEQUENCE_LENGTH,
):
    global counter
    file = os.path.join(folder, f"{str(counter).zfill(2)}-{config_name}.yaml")
    counter += 1
    with open(file, "w", encoding="utf-8") as f:
        f.write(config_common(config_name, dataset, embedding_size))
        f.write(
            config_model(
                model,
                embedding_size,
                nb_layers,
                normalize,
                time_features,
                sequence_length,
            )
        )


def create_configs(
    config_name,
    folder,
    model,
    embedding_size=32,
    normalize=True,
    time_features=None,
    nb_layers=1,
    sequence_length=DEFAULT_SEQUENCE_LENGTH,
):
    create_config(
        config_name,
        folder,
        model,
        "wikipedia",
        embedding_size,
        normalize,
        time_features,
        nb_layers,
        sequence_length,
    )
    create_config(
        config_name,
        folder,
        model,
        "reddit",
        embedding_size,
        normalize,
        time_features,
        nb_layers,
        sequence_length,
    )
    create_config(
        config_name,
        folder,
        model,
        "lastfm",
        embedding_size,
        normalize,
        time_features,
        nb_layers,
        sequence_length,
    )


def main():
    parent_folder = os.path.join("configs", "experiment")
    os.makedirs(parent_folder, exist_ok=True)
    create_configs("limnet-best", parent_folder, "LiMNet")
    create_configs("jodie-best", parent_folder, "Jodie")

    create_configs("limnet-normalize-without", parent_folder, "LiMNet", normalize=False)

    create_configs(
        "limnet-time-feature-both", parent_folder, "LiMNet", time_features="both"
    )
    create_configs(
        "limnet-time-feature-day", parent_folder, "LiMNet", time_features="day"
    )
    create_configs(
        "limnet-time-feature-week", parent_folder, "LiMNet", time_features="week"
    )

    create_configs("limnet-layers-3", parent_folder, "LiMNet", nb_layers=3)
    create_configs("limnet-layers-5", parent_folder, "LiMNet", nb_layers=5)
    create_configs("limnet-layers-2", parent_folder, "LiMNet", nb_layers=2)

    create_configs("jodie-embeddings-16", parent_folder, "Jodie", embedding_size=16)
    create_configs("jodie-embeddings-64", parent_folder, "Jodie", embedding_size=64)
    create_configs("jodie-embeddings-48", parent_folder, "Jodie", embedding_size=48)
    create_configs("jodie-embeddings-128", parent_folder, "Jodie", embedding_size=128)

    create_configs("limnet-low-seq-length", parent_folder, "LiMNet", sequence_length=64)
    create_configs("jodie-low-seq-length", parent_folder, "Jodie", sequence_length=64)
    create_configs(
        "limnet-low-seq-length", parent_folder, "LiMNet", sequence_length=256
    )
    create_configs("jodie-low-seq-length", parent_folder, "Jodie", sequence_length=256)
    create_configs("limnet-low-seq-length", parent_folder, "LiMNet", sequence_length=16)
    create_configs("jodie-low-seq-length", parent_folder, "Jodie", sequence_length=16)

    create_configs("static-embeddings-best", parent_folder, "StaticEmbeddings")
    create_configs(
        "static-embeddings-normalize-without",
        parent_folder,
        "StaticEmbeddings",
        normalize=False,
    )


if __name__ == "__main__":
    main()
