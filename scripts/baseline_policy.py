import argparse
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType


def get_next_action(observation: ObsType, info: dict[str, Any]) -> ActType:
    next_checkpoint_pos = observation[1:3] * [info["width"], info["height"]]
    car_pos = observation[5:7] * [info["width"], info["height"]]
    car_angle = observation[9] * info["car_angle_upper_bound"]

    diff = next_checkpoint_pos - car_pos
    next_checkpoint_angle = np.rad2deg(np.atan2(diff[1], diff[0]))
    right_diff_angle = (next_checkpoint_angle - car_angle) % 360
    left_diff_angle = (car_angle - next_checkpoint_angle) % 360

    angle = np.clip(
        (right_diff_angle if right_diff_angle < left_diff_angle else -left_diff_angle),
        -info["max_rotation_per_turn"],
        info["max_rotation_per_turn"],
    )

    thrust = 50 if np.linalg.norm(diff) < 3000 else 100

    return np.array([angle, thrust]) / [
        info["max_rotation_per_turn"],
        info["car_max_thrust"],
    ]


def baseline_policy(
    env_id: str,
    test_id: int | None = None,
    seed: int | None = None,
    n_timesteps: int = 600,
) -> None:
    env = gym.make(env_id, render_mode="human")

    observation, info = env.reset(seed=seed, options={"test_id": test_id})

    for _ in range(n_timesteps):
        action = get_next_action(observation=observation, info=info)
        observation, _reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--env",
        default="gymnasium_search_race:gymnasium_search_race/SearchRace-v1",
        help="environment id",
    )
    parser.add_argument(
        "--test-id",
        type=int,
        help="test id",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random generator seed",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=600,
        help="number of timesteps",
    )
    args = parser.parse_args()
    baseline_policy(
        env_id=args.env,
        test_id=args.test_id,
        seed=args.seed,
        n_timesteps=args.n_timesteps,
    )
