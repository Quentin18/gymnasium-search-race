import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

RESOURCES_PATH = Path(__file__).resolve().parent / "resources"


def test_check_env():
    env = gym.make(
        "gymnasium_search_race:gymnasium_search_race/SearchRace-v1",
        test_id=1,
    )
    check_env(env=env.unwrapped)


@pytest.mark.parametrize(
    "test_id,expected",
    (
        (
            1,
            np.array(
                [
                    0.1723125,
                    0.51766667,
                    0.209875,
                    0.31533333,
                    0.6470625,
                    0.22066667,
                    1.0,
                    1.0,
                    1.0,
                    0.6470625,
                    0.22066667,
                    0.0,
                    0.0,
                    0.44722222,
                ]
            ),
        ),
        (
            2,
            np.array(
                [
                    0.2144375,
                    0.7031111111111111,
                    0.26775,
                    0.31122222222222223,
                    0.6963125,
                    0.51,
                    1.0,
                    1.0,
                    1.0,
                    0.6963125,
                    0.51,
                    0.0,
                    0.0,
                    0.4638888888888889,
                ]
            ),
        ),
        (
            700,
            np.array(
                [
                    0.75,
                    0.1111111111111111,
                    0.78125,
                    0.2777777777777778,
                    0.8125,
                    0.4444444444444444,
                    1.0,
                    1.0,
                    1.0,
                    0.0625,
                    0.1111111111111111,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
    ),
)
def test_search_race_reset(test_id: int, expected: np.ndarray):
    env = gym.make(
        "gymnasium_search_race:gymnasium_search_race/SearchRace-v1",
        test_id=test_id,
    )

    observation, _info = env.reset()

    np.testing.assert_allclose(
        observation,
        expected,
        err_msg="observation is wrong",
    )


@pytest.mark.parametrize(
    "test_id",
    (1, 2, 700),
)
def test_search_race_step(test_id: int):
    env = gym.make(
        "gymnasium_search_race:gymnasium_search_race/SearchRace-v1",
        test_id=test_id,
    )

    game = json.loads(
        (RESOURCES_PATH / f"game{test_id}.json").read_text(encoding="UTF-8")
    )
    nb_checkpoints = int(game["stdin"][0])

    observation, info = env.reset()

    for i, (stdin, stdout) in enumerate(
        zip(game["stdin"][nb_checkpoints + 1 :], game["stdout"])
    ):
        expected = [int(i) for i in stdin.split()]
        expected[5] = expected[5] % 360
        np.testing.assert_allclose(
            [
                info["current_checkpoint"],
                *observation[-5:]
                * [
                    info["width"],
                    info["height"],
                    info["car_thrust_upper_bound"],
                    info["car_thrust_upper_bound"],
                    info["car_angle_upper_bound"],
                ],
            ],
            expected,
            err_msg=f"observation is wrong at step {i}",
        )

        action = np.array([int(i) for i in stdout.split()[1:3]]) / [
            info["max_rotation_per_turn"],
            info["car_max_thrust"],
        ]
        observation, _reward, _terminated, _truncated, info = env.step(action)

        if info["current_checkpoint"] == nb_checkpoints - 2:
            visit_checkpoints = [1, 1, 0]
        elif info["current_checkpoint"] == nb_checkpoints - 1:
            visit_checkpoints = [1, 0, 0]
        elif info["current_checkpoint"] == nb_checkpoints:
            visit_checkpoints = [0, 0, 0]
        else:
            visit_checkpoints = [1, 1, 1]

        np.testing.assert_array_equal(
            observation[6:9],
            visit_checkpoints,
            err_msg=f"visit checkpoints is wrong at step {i}",
        )
