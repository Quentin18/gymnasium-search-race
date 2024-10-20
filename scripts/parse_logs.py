import argparse
import json
from pathlib import Path


def parse_logs(input_path: Path, output_path: Path) -> None:
    stdin = []
    stdout = []

    with open(input_path, "r", encoding="utf-8") as file:
        for line in file:
            if line[0].isdigit():
                stdin.append(line.strip())
            elif "EXPERT" in line:
                stdout.append(line.split(" - ")[1].strip())

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump({"stdin": stdin, "stdout": stdout}, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse logs from CGSearchRace referee",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="path to input log file",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        required=True,
        help="path to output json file",
    )
    args = parser.parse_args()
    parse_logs(input_path=args.input_path, output_path=args.output_path)
