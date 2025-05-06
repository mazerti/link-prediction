import argparse
import traceback

from tqdm import tqdm

from context import Context
from main import initialize_run


def main():
    """Main function."""
    args = parse_args()
    for config in args.checkpoint:
        try:
            run_evaluation(args, config)
        # pylint: disable=locally-disabled, broad-exception-caught
        except Exception:
            print(traceback.format_exc())


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Link Prediction for Temporal Interaction Network"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="+",
        help="Path to the checkpoint PTH file.",
    )
    parser.add_argument(
        "--gpu", type=str, required=False, default=None, help="Specify a gpu to use."
    )
    return parser.parse_args()


def run_evaluation(args: argparse.Namespace, checkpoint: str):
    """Run a thorough evaluation for the requested checkpoint and arguments."""
    context = initialize_run(
        args, config_name=checkpoint, wand_project="link-prediction-insights"
    )
    context.metrics = ["plot_predictions"]
    _, eval_data = context.data
    model = context.model
    model.eval()
    measures = {}
    for X in tqdm(eval_data, desc="validation", leave=False):
        user_sequences, item_sequences = X
        model.evaluate_sequence(
            context,
            loss_fn=lambda _, __, ___: 0,
            measures=measures,
            user_sequences=user_sequences,
            item_sequences=item_sequences,
        )


if __name__ == "__main__":
    main()
