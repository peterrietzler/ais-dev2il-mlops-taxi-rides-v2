from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

INPUT_DIR = Path("data")
OUTPUT_FILE = Path("data/all.taxi-rides.parquet")
OVERWRITE = True


def find_input_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    filename_pattern = re.compile(r"\d{4}-\d{2}-\d{2}\.taxi-rides\.parquet$")
    matches = [
        path for path in input_dir.iterdir() if path.is_file() and filename_pattern.fullmatch(path.name)
    ]
    return sorted(matches)


def combine_parquet_files(input_files: list[Path]) -> pd.DataFrame:
    if not input_files:
        raise ValueError("No input parquet files matched the expected filename pattern.")

    frames = [pd.read_parquet(path) for path in input_files]
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    if OUTPUT_FILE.exists() and not OVERWRITE:
        raise FileExistsError(
            f"Output file already exists: {OUTPUT_FILE}."
        )

    input_files = find_input_files(INPUT_DIR)
    combined = combine_parquet_files(input_files)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(str(OUTPUT_FILE))

    print(f"Input files: {len(input_files)}")
    print(f"Rows written: {len(combined)}")
    print(f"Output: {OUTPUT_FILE}")

    # `DataFrame.describe()` defaults to numeric columns only.
    # We want a summary for *all* columns.
    print(combined.describe(include="all"))


if __name__ == "__main__":
    main()
