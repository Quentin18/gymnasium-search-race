import argparse
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType


def get_next_action(observation: ObsType, info: dict[str, Any]) -> ActType:
    car_angle = info["angle"]
    next_cp_diff = observation[:2] * info["distance_upper_bound"]
    next_checkpoint_angle = np.rad2deg(np.arctan2(next_cp_diff[1], next_cp_diff[0]))
    right_diff_angle = (next_checkpoint_angle - car_angle) % 360
    left_diff_angle = (car_angle - next_checkpoint_angle) % 360

    angle = np.clip(
        (right_diff_angle if right_diff_angle < left_diff_angle else -left_diff_angle),
        -info["max_rotation_per_turn"],
        info["max_rotation_per_turn"],
    )

    thrust = 50 if np.linalg.norm(next_cp_diff) < 3000 else 100

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
        default="gymnasium_search_race:gymnasium_search_race/SearchRace-v2",
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
