import argparse
import json

from gymnasium_search_race.envs.search_race import MAPS_PATH, get_test_ids


def merge_maps() -> dict[str, list[list[int]]]:
    test_maps = {}

    for test_id in get_test_ids():
        path = MAPS_PATH / f"test{test_id}.json"
        test_map = json.loads(path.read_text(encoding="UTF-8"))
        test_maps[str(test_id)] = [
            [int(i) for i in checkpoint.split()]
            for checkpoint in test_map["testIn"].split(";")
        ]

    return test_maps


def write_merged_maps(
    path: str,
    maps: dict[str, list[list[int]]],
) -> None:
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(maps, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge Search Race maps in one JSON file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-path",
        help="path to output JSON file",
    )
    args = parser.parse_args()
    merged_maps = merge_maps()

    if args.output_path:
        write_merged_maps(path=args.output_path, maps=merged_maps)
