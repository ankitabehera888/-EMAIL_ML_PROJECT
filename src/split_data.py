"""Split processed email pairs into train, validation, and test sets."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_PROCESSED,
    PROCESSED_FILENAME,
    TEST_FILENAME,
    TRAIN_FILENAME,
    VAL_FILENAME,
)


def split_pairs(
    input_path: Path,
    output_dir: Path,
    test_size: float = 0.2,
    val_size: float = 0.5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_path).dropna()

    train, temp = train_test_split(df, test_size=test_size, random_state=random_state)
    if len(temp) < 2:
        val = temp
        test = temp.iloc[0:0].copy()
    else:
        val, test = train_test_split(temp, test_size=val_size, random_state=random_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / TRAIN_FILENAME, index=False)
    val.to_csv(output_dir / VAL_FILENAME, index=False)
    test.to_csv(output_dir / TEST_FILENAME, index=False)

    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Split email pairs into train/val/test CSVs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_PROCESSED / PROCESSED_FILENAME,
    )
    parser.add_argument("--output-dir", type=Path, default=DATA_PROCESSED)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Processed data not found at {args.input}.")

    train, val, test = split_pairs(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")


if __name__ == "__main__":
    main()
