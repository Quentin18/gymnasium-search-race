import argparse
import gzip
import json

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

from gymnasium_search_race.envs.search_race import get_test_ids
from gymnasium_search_race.wrappers import RecordBestEpisodeStatistics


def search_best_actions_on_test_id(
    test_id: int,
    model_path: str,
    env_id: str,
    total_timesteps: int = 200_000,
) -> list[list[int]]:
    env = RecordBestEpisodeStatistics(gym.make(env_id, test_id=test_id))
    model = PPO.load(
        model_path,
        env=DummyVecEnv([lambda: env]),
        verbose=0,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=total_timesteps)
    return [
        [
            round(action[0] * env.get_wrapper_attr("max_rotation_per_turn")),
            round(action[1] * env.get_wrapper_attr("car_max_thrust")),
        ]
        for action in env.best_episode_actions
    ]


def search_best_actions(
    model_path: str,
    env_id: str,
    total_timesteps: int = 200_000,
) -> dict[int, list[list[int]]]:
    total_length = 0
    actions_per_test_id = {}

    progress_bar = tqdm(get_test_ids(), desc="Search best actions")
    for test_id in progress_bar:
        actions = search_best_actions_on_test_id(
            test_id=test_id,
            model_path=model_path,
            env_id=env_id,
            total_timesteps=total_timesteps,
        )
        length = len(actions)
        progress_bar.set_postfix({f"test_{test_id}": length})
        total_length += length
        actions_per_test_id[test_id] = actions

    print("Total:", total_length)

    return actions_per_test_id


def write_best_actions(
    path: str,
    actions: dict[int, list[list[int]]],
) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as json_file:
        json.dump(actions, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search best actions for Search Race",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="path to model file",
    )
    parser.add_argument(
        "--env",
        default="gymnasium_search_race:gymnasium_search_race/SearchRace-v1",
        help="environment id",
    )
    parser.add_argument(
        "--total-timesteps",
        default=200_000,
        type=int,
        help="total timesteps to train",
    )
    parser.add_argument(
        "--output-path",
        help="path to output GZIP compressed JSON file",
    )
    args = parser.parse_args()
    best_actions = search_best_actions(
        model_path=args.model_path,
        env_id=args.env,
        total_timesteps=args.total_timesteps,
    )

    if args.output_path:
        write_best_actions(
            path=args.output_path,
            actions=best_actions,
        )
