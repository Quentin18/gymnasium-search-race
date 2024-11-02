from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from gymnasium_search_race.envs.car import Car
from gymnasium_search_race.envs.search_race import SearchRaceEnv

ROOT_PATH = Path(__file__).resolve().parent
ASSETS_PATH = ROOT_PATH / "assets" / "mad_pod_racing"

MAPS = [
    [
        [12460, 1350],
        [10540, 5980],
        [3580, 5180],
        [13580, 7600],
    ],
    [
        [3600, 5280],
        [13840, 5080],
        [10680, 2280],
        [8700, 7460],
        [7200, 2160],
    ],
    [
        [4560, 2180],
        [7350, 4940],
        [3320, 7230],
        [14580, 7700],
        [10560, 5060],
        [13100, 2320],
    ],
    [
        [5010, 5260],
        [11480, 6080],
        [9100, 1840],
    ],
    [
        [14660, 1410],
        [3450, 7220],
        [9420, 7240],
        [5970, 4240],
    ],
    [
        [3640, 4420],
        [8000, 7900],
        [13300, 5540],
        [9560, 1400],
    ],
    [
        [4100, 7420],
        [13500, 2340],
        [12940, 7220],
        [5640, 2580],
    ],
    [
        [14520, 7780],
        [6320, 4290],
        [7800, 860],
        [7660, 5970],
        [3140, 7540],
        [9520, 4380],
    ],
    [
        [10040, 5970],
        [13920, 1940],
        [8020, 3260],
        [2670, 7020],
    ],
    [
        [7500, 6940],
        [6000, 5360],
        [11300, 2820],
    ],
    [
        [4060, 4660],
        [13040, 1900],
        [6560, 7840],
        [7480, 1360],
        [12700, 7100],
    ],
    [
        [3020, 5190],
        [6280, 7760],
        [14100, 7760],
        [13880, 1220],
        [10240, 4920],
        [6100, 2200],
    ],
    [
        [10323, 3366],
        [11203, 5425],
        [7259, 6656],
        [5425, 2838],
    ],
]


class MadPodRacingEnv(SearchRaceEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        opponent_path: str | Path | None = None,
        is_opponent_blocker: bool = False,
    ) -> None:
        super().__init__(render_mode=render_mode)
        self.car_max_thrust = 100
        self.car_thrust_upper_bound = 1000

        self.background_img_path = ASSETS_PATH / "background.jpg"
        self.car_img_path = ASSETS_PATH / "space_ship_runner.png"

        self.opponent_model = PPO.load(opponent_path) if opponent_path else None
        self.is_opponent_blocker = is_opponent_blocker

    def _generate_checkpoints(
        self,
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        # https://github.com/Agade09/CSB-Runner-Arena/blob/master/Arena.cpp#L276
        checkpoints = np.array(
            MAPS[self.np_random.integers(0, len(MAPS))],
            dtype=np.float64,
        )
        shift = self.np_random.integers(0, len(checkpoints))
        checkpoints = np.roll(checkpoints, shift=shift, axis=0)
        delta = self.np_random.integers(-30, 31, checkpoints.shape)
        return checkpoints + delta

    def _generate_car(self) -> None:
        # https://github.com/robostac/coders-strike-back-referee/blob/master/csbref.go#L407
        self.cars = []

        cp1_minus_cp0 = self.checkpoints[1] - self.checkpoints[0]
        distance = np.linalg.norm(cp1_minus_cp0)
        cp1_minus_cp0 /= distance

        self.car = Car(
            x=self.checkpoints[0][0] + cp1_minus_cp0[1] * 500,
            y=self.checkpoints[0][1] + cp1_minus_cp0[0] * -500,
            angle=0,
            current_checkpoint=0,
        )
        self.car.angle = self.car.get_angle(
            x=self.checkpoints[1][0],
            y=self.checkpoints[1][1],
        )
        self.cars.append(self.car)

        if self.opponent_model:
            self.opponent_car = Car(
                x=self.checkpoints[0][0] + cp1_minus_cp0[1] * 1500,
                y=self.checkpoints[0][1] + cp1_minus_cp0[0] * -1500,
                angle=0,
                current_checkpoint=0,
            )
            self.opponent_car.angle = self.opponent_car.get_angle(
                x=self.checkpoints[1][0],
                y=self.checkpoints[1][1],
            )
            self.cars.append(self.opponent_car)
        else:
            self.opponent_car = None

    def _adjust_car(self) -> None:
        for car in self.cars:
            car.round_position()
            car.truncate_speed(friction=self.car_friction)


class MadPodRacingDiscreteEnv(MadPodRacingEnv):
    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)

        self.actions = list(
            product(
                list(
                    range(
                        -self.max_rotation_per_turn,
                        self.max_rotation_per_turn + 1,
                    )
                ),
                [0, self.car_max_thrust],
            )
        )
        self.action_space = spaces.Discrete(len(self.actions))

    def _convert_action_to_angle_thrust(
        self,
        action: np.ndarray,
    ) -> tuple[float, float]:
        return self.actions[action]
