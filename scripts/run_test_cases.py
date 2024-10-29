import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from gymnasium_search_race.envs.search_race import MAPS_PATH


def get_test_ids() -> list[int]:
    return sorted(
        int(re.match(r"test(\d+)\.json", path.name).group(1))
        for path in MAPS_PATH.iterdir()
    )


def get_test_case_length(env: gym.Env, model: PPO, test_id: int) -> int:
    observation, info = env.reset(options={"test_id": test_id})
    terminated = truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(observation=observation, deterministic=True)
        observation, _reward, terminated, truncated, info = env.step(action)

    return int(info["episode"]["l"][0])


def write_metrics(
    metrics_folder: str,
    env_id: str,
    episode_lengths: dict[int, int],
) -> None:
    with open(Path(metrics_folder) / "metrics.csv", "a", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["env", "date", "total", *sorted(episode_lengths.keys())],
        )
        writer.writerow(
            {
                "env": env_id,
                "date": datetime.now().isoformat(timespec="seconds"),
                "total": sum(episode_lengths.values()),
                **episode_lengths,
            }
        )


def run_test_cases(
    model_path: str,
    env_id: str,
    record_video: bool = False,
    video_folder: str = "videos",
) -> dict[int, int]:
    env = gym.make(env_id, render_mode="rgb_array" if record_video else None)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    if record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda _: True,
            disable_logger=True,
        )

    model = PPO.load(path=model_path, env=env)

    test_ids = get_test_ids()
    episode_lengths = {}
    total_length = 0

    for test_id in test_ids:
        episode_length = get_test_case_length(env=env, model=model, test_id=test_id)
        print(f"Test {test_id:03}: {episode_length}")
        episode_lengths[test_id] = episode_length
        total_length += episode_length

    env.close()
    print("Total:", total_length)

    return episode_lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Search Race model on CodinGame test cases",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="path to model file",
    )
    parser.add_argument(
        "--env",
        default="gymnasium_search_race:gymnasium_search_race/SearchRace-v1",
        help="environment id",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="flag to record videos of episodes",
    )
    parser.add_argument(
        "--video-folder",
        default="videos",
        help="path to videos folder",
    )
    parser.add_argument(
        "--record-metrics",
        action="store_true",
        help="flag to record metrics",
    )
    parser.add_argument(
        "--metrics-folder",
        default="metrics",
        help="path to metrics folder",
    )
    args = parser.parse_args()

    lengths = run_test_cases(
        model_path=args.path,
        env_id=args.env,
        record_video=args.record_video,
        video_folder=args.video_folder,
    )

    if args.record_metrics:
        write_metrics(
            metrics_folder=args.metrics_folder,
            env_id=args.env,
            episode_lengths=lengths,
        )
