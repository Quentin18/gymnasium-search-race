import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

RESOURCES_PATH = Path(__file__).resolve().parent / "resources"


@pytest.mark.parametrize(
    "env_id",
    (
        "gymnasium_search_race:gymnasium_search_race/SearchRace-v3",
        "gymnasium_search_race:gymnasium_search_race/SearchRaceDiscrete-v3",
        "gymnasium_search_race:gymnasium_search_race/MadPodRacing-v2",
        "gymnasium_search_race:gymnasium_search_race/MadPodRacingDiscrete-v2",
    ),
)
def test_check_env(env_id: str):
    env = gym.make(env_id)
    check_env(env=env.unwrapped)


@pytest.mark.parametrize(
    "env_id,test_id,test_index",
    (
        ("gymnasium_search_race:gymnasium_search_race/SearchRace-v3", 700, 46),
        ("gymnasium_search_race:gymnasium_search_race/SearchRaceDiscrete-v3", 700, 46),
        ("gymnasium_search_race:gymnasium_search_race/MadPodRacing-v2", 8, 8),
        ("gymnasium_search_race:gymnasium_search_race/MadPodRacingDiscrete-v2", 8, 8),
    ),
)
def test_env_reset_when_test_id_is_not_none(
    env_id: str,
    test_id: int,
    test_index: int,
):
    env = gym.make(env_id, test_id=test_id)

    for _ in range(100):
        _observation, _info = env.reset()
        assert getattr(env.unwrapped, "test_index") == test_index


@pytest.mark.parametrize(
    "env_id,test_size",
    (
        ("gymnasium_search_race:gymnasium_search_race/SearchRace-v3", 100),
        ("gymnasium_search_race:gymnasium_search_race/SearchRaceDiscrete-v3", 100),
        ("gymnasium_search_race:gymnasium_search_race/MadPodRacing-v2", 26),
        ("gymnasium_search_race:gymnasium_search_race/MadPodRacingDiscrete-v2", 26),
    ),
)
def test_env_reset_when_sequential_maps_is_false(env_id: str, test_size: int):
    env = gym.make(env_id, sequential_maps=False)

    test_indexes = []

    for _ in range(test_size):
        _observation, _info = env.reset()
        test_indexes.append(getattr(env.unwrapped, "test_index"))

    assert test_indexes != sorted(test_indexes)


@pytest.mark.parametrize(
    "env_id,test_size",
    (
        ("gymnasium_search_race:gymnasium_search_race/SearchRace-v3", 100),
        ("gymnasium_search_race:gymnasium_search_race/SearchRaceDiscrete-v3", 100),
        ("gymnasium_search_race:gymnasium_search_race/MadPodRacing-v2", 26),
        ("gymnasium_search_race:gymnasium_search_race/MadPodRacingDiscrete-v2", 26),
    ),
)
def test_env_reset_when_sequential_maps_is_true(env_id: str, test_size: int):
    env = gym.make(env_id, sequential_maps=True)

    test_indexes = []

    for _ in range(test_size):
        _observation, _info = env.reset()
        test_indexes.append(getattr(env.unwrapped, "test_index"))

    assert (
        test_indexes[: test_size // 2]
        == test_indexes[test_size // 2 :]
        == list(range(test_size // 2))
    )


@pytest.mark.parametrize(
    "test_id",
    (1, 2, 700),
)
def test_search_race_step(test_id: int):
    env = gym.make(
        "gymnasium_search_race:gymnasium_search_race/SearchRace-v3",
        test_id=test_id,
    )

    game = json.loads(
        (RESOURCES_PATH / f"game{test_id}.json").read_text(encoding="UTF-8")
    )
    nb_checkpoints = int(game["stdin"][0])

    _observation, info = env.reset()

    for i, (stdin, stdout) in enumerate(
        zip(game["stdin"][nb_checkpoints + 1 :], game["stdout"])
    ):
        actual = [
            info["current_checkpoint"],
            info["x"],
            info["y"],
            info["vx"],
            info["vy"],
            info["angle"],
        ]
        expected = [int(i) for i in stdin.split()]
        expected[5] = expected[5] % 360
        np.testing.assert_allclose(
            actual,
            expected,
            err_msg=f"game state is wrong at step {i}",
        )

        action = np.array([int(i) for i in stdout.split()[1:3]]) / [
            info["max_rotation_per_turn"],
            info["car_max_thrust"],
        ]
        _observation, _reward, _terminated, _truncated, info = env.step(action)
