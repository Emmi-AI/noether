#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from noether.core.factory.dataset import DatasetFactory
from noether.core.schemas.dataset import DatasetBaseConfig
from noether.data.base.dataset import Dataset
from noether.data.base.wrappers import PropertySubsetWrapper
from noether.data.stats import RunningMoments


def parse_args() -> dict[str, Any]:
    """
    Parse command line arguments for dataset statistics calculation.

    Returns:
        dict[str, Any]: Dictionary containing all parsed arguments
    """
    parser = argparse.ArgumentParser("Calculate statistics of DrivAerML/AhmedML.")
    parser.add_argument(
        "--dataset_kind",
        type=str,
        required=True,
        help="Class path of the dataset to use, e.g., noether.data.datasets.AhmedMLDataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use, e.g., 'train' or 'test'",
    )
    parser.add_argument(
        "--log_scale",
        type=lambda s: set(s.split(",")) if s else set(),
        default=set(),
        help="Comma-separated list of attributes to calculate statistics for in log scale, e.g., 'pressure,vorticity'.",
    )

    parser.add_argument(
        "--exclude_attributes",
        type=lambda s: set(s.split(",")) if s else set(),
        default=set(),
        help="Comma-separated list of properties from the dataset class to exclude from statistics calculation. By default, all properties are included.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save statistics as JSON file (optional)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )
    # Parse known args first
    known_args, unknown_args = parser.parse_known_args()

    # Process unknown args for dataset constructor
    dataset_constructor_args = parse_dataset_args(unknown_args)

    return {**vars(known_args), **dataset_constructor_args}


def parse_dataset_args(args: list) -> dict[str, Any]:
    """
    Parse additional arguments for dataset constructor.

    Args:
        args: List of unparsed command-line arguments

    Returns:
        Dict[str, Any]: Dictionary of parsed dataset constructor arguments
    """
    dataset_args: dict[str, Any] = {}
    i = 0

    while i < len(args):
        arg = args[i]
        if not arg.startswith("-"):
            i += 1
            continue

        # Clean the argument name
        key = arg.lstrip("-")
        value: Any = True  # Default for flags

        # Handle different argument formats
        if "=" in key:
            key, value = key.split("=", 1)
        elif i + 1 < len(args) and not args[i + 1].startswith("-"):
            value = args[i + 1]
            i += 1  # Skip the next item

        # Try to convert string values to appropriate types
        try:
            # Try to convert to int or float if possible
            if isinstance(value, str):
                if value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit() and value.count(".") < 2:
                    value = float(value)
                elif value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
        except (ValueError, TypeError):
            pass  # Keep as string if conversion fails

        dataset_args[key] = value
        print(f"  -> Added '{key}' with value '{value}'")
        i += 1

    return dataset_args


def get_dataset_attributes(dataset) -> set[str]:
    """
    Extract all available attributes from the dataset that have getitem methods.

    Args:
        dataset: The dataset object

    Returns:
        Set[str]: Set of attribute names
    """
    return {attribute[len("getitem_") :] for attribute in dataset.get_all_getitem_names()}


def calculate_statistics(
    dataset,
    dataset_attributes: set[str],
    log_scale: set[str],
    num_workers: int = 0,
) -> dict[str, RunningMoments]:
    """
    Calculate statistics for all dataset attributes.

    Args:
        dataset: The dataset object
        dataset_attributes: Set of attribute names to process
        log_scale: Set of attributes to process in log scale
        num_workers: Number of workers for data loading

    Returns:
        Dict[str, RunningMoments]: Dictionary mapping attribute names to their statistics
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_stats = {
        attr: RunningMoments(log_scale=(attr in log_scale)).to(device) for attr in dataset_attributes if attr != "index"
    }

    dataloader = DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    # Use tqdm for progress tracking
    with tqdm(total=len(dataset), desc="Processing dataset") as pbar:
        for batch in dataloader:
            for sample in batch:
                try:
                    for key, value in sample.items():
                        if key not in dataset_attributes and key != "index":
                            raise ValueError(f"Sample has unexpected key: '{key}'")
                        if key != "index":
                            running_stats[key].push_tensor(value)
                except Exception as e:
                    print(f"Error processing sample: {str(e)}")
                pbar.update(1)

    return running_stats


def print_statistics(running_stats: dict[str, RunningMoments], log_scale: set[str]) -> None:
    """
    Print calculated statistics for each attribute.

    Args:
        running_stats: Dictionary mapping attribute names to their statistics
        log_scale: Set of attributes processed in log scale
    """
    torch.set_printoptions(precision=5, sci_mode=True)

    for key, stats in sorted(running_stats.items()):
        scale_label = " (log scale)" if key in log_scale else ""
        print(f"\nStatistics for '{key}'{scale_label}:")
        stats.print()


def save_statistics_to_json(
    running_stats: dict[str, RunningMoments], output_path: str | Path, log_scale: set[str]
) -> None:
    """
    Save calculated statistics to a JSON file.

    Args:
        running_stats: Dictionary mapping attribute names to their statistics
        output_path: Path where the JSON file will be saved
        log_scale: Set of attributes processed in log scale
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats_dict = {}
    for key, stats in sorted(running_stats.items()):
        stats_dict[key] = {
            "mean": stats.mean.tolist() if isinstance(stats.mean, torch.Tensor) else stats.mean,
            "std": stats.std.tolist() if isinstance(stats.std, torch.Tensor) else stats.std,
            "min": stats.min.tolist() if isinstance(stats.min, torch.Tensor) else stats.min,
            "max": stats.max.tolist() if isinstance(stats.max, torch.Tensor) else stats.max,
            "count": int(stats.count),
            "log_scale": key in log_scale,
        }

    with open(output_path, "w") as f:
        json.dump(stats_dict, f, indent=2)

    print(f"\nStatistics saved to: {output_path}")


def calculate_dataset_statistics(
    dataset_kind: str,
    log_scale: set[str],
    exclude_attributes: set[str],
    output_json: str | None = None,
    num_workers: int = 0,
    **dataset_constructor_args,
) -> None:
    """
    Main function to calculate and display dataset statistics.

    Args:
        dataset_kind: Class path of the dataset
        log_scale: Set of attributes to process in log scale
        exclude_attributes: Set of attributes to exclude from calculation
        output_json: Optional path to save statistics as JSON
        num_workers: Number of workers for data loading
        dataset_constructor_args: Additional arguments for dataset constructor
    """
    try:
        # Instantiate dataset
        config = DatasetBaseConfig(kind=dataset_kind, **dataset_constructor_args)
        dataset: Dataset | PropertySubsetWrapper = DatasetFactory().instantiate(config)
        dataset.compute_statistics = True  # type: ignore[union-attr]

        # Get dataset attributes
        dataset_attributes = get_dataset_attributes(dataset)
        print(f"Dataset has the following getitem attributes: {dataset_attributes}")
        if exclude_attributes:
            print(f"Excluding attributes: {exclude_attributes}")
            # Check if any excluded attribute is not in dataset_attributes
            invalid_attributes = exclude_attributes - dataset_attributes
            if invalid_attributes:
                raise ValueError(f"Cannot exclude non-existent attributes: {invalid_attributes}")

            dataset_attributes = dataset_attributes - exclude_attributes

            print(f"Dataset used for the statistics: {dataset_attributes}")
            dataset = PropertySubsetWrapper(dataset, dataset_attributes)  # type: ignore[arg-type]

        # Calculate statistics
        running_stats = calculate_statistics(
            dataset,
            dataset_attributes,
            log_scale,
            num_workers=num_workers,
        )

        # Print results
        print_statistics(running_stats, log_scale)

        # Save to JSON if output path is provided
        if output_json:
            save_statistics_to_json(running_stats, output_json, log_scale)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


def main() -> None:
    """CLI entry point for calculating dataset statistics."""
    calculate_dataset_statistics(**parse_args())


if __name__ == "__main__":
    main()
