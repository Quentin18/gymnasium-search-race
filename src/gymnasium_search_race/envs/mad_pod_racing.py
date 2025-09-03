from itertools import product
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from stable_baselines3 import PPO

from gymnasium_search_race.envs.models import Car, Unit
from gymnasium_search_race.envs.search_race import SCALE_FACTOR, SearchRaceEnv

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

START_POINT_MULT = [[500, -500], [-500, 500], [1500, -1500], [-1500, 1500]]


class MadPodRacingEnv(SearchRaceEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        laps: int = 3,
        car_max_thrust: int = 200,
        test_id: int | None = None,
        sequential_maps: bool = False,
        opponent_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            laps=laps,
            car_max_thrust=car_max_thrust,
            test_id=test_id,
            sequential_maps=sequential_maps,
        )
        self.car_radius = 400
        self.min_impulse = 120.0

        self.background_img_path = ASSETS_PATH / "background.jpg"
        self.car_img_path = ASSETS_PATH / "space_ship_runner.png"
        self.opponent_car_img_path = ASSETS_PATH / "space_ship_blocker.png"

        self.opponent_model = PPO.load(opponent_path) if opponent_path else None

    def _get_test_ids(self) -> list[int]:
        return list(range(len(MAPS)))

    def _get_test_checkpoints(self) -> list[np.ndarray]:
        return [
            np.array(checkpoints, dtype=self.observation_space.dtype)
            for checkpoints in MAPS
        ]

    def _get_runner_obs(self, car_index: int) -> ObsType:
        car = self.cars[car_index]
        obs = []

        # position and angle of the next 2 checkpoints relative to the car
        for i in range(2):
            x_cp, y_cp = self.checkpoints[
                (car.current_checkpoint + i + 1) % len(self.checkpoints)
            ]
            obs.append(self._get_diff_obs(car=car, x=x_cp, y=y_cp))

        # car speed
        obs.append(self._get_speed_obs(car=car))

        return np.concatenate(obs).astype(self.observation_space.dtype)

    def _get_blocker_obs(self, car_index: int) -> ObsType:
        runner_car_index = (car_index + 1) % len(self.cars)
        runner_car = self.cars[runner_car_index]
        blocker_car = self.cars[car_index]
        return np.concatenate(
            [
                self._get_diff_obs(car=blocker_car, x=runner_car.x, y=runner_car.y),
                self._get_speed_obs(car=blocker_car),
                self._get_runner_obs(car_index=runner_car_index),
            ]
        )

    def _get_obs(self) -> ObsType:
        return self._get_runner_obs(car_index=0)

    def _get_opponent_obs(self) -> ObsType:
        return self._get_blocker_obs(car_index=1)

    def _get_terminated(self) -> bool:
        return any(
            car.current_checkpoint >= self.total_checkpoints for car in self.cars
        )

    def _generate_checkpoints(
        self,
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        # https://github.com/Agade09/CSB-Runner-Arena/blob/master/Arena.cpp#L276
        checkpoints = super()._generate_checkpoints(options=options)
        shift = self.np_random.integers(0, len(checkpoints))
        checkpoints = np.roll(checkpoints, shift=shift, axis=0)
        delta = self.np_random.integers(-30, 31, checkpoints.shape)
        return checkpoints + delta

    def _generate_car(self) -> None:
        # https://github.com/robostac/coders-strike-back-referee/blob/master/csbref.go#L407
        self.cars = []

        start_point_mult_index = self.np_random.choice(
            len(START_POINT_MULT),
            size=2,
            replace=False,
        )

        cp1_minus_cp0 = self.checkpoints[1] - self.checkpoints[0]
        distance = np.linalg.norm(cp1_minus_cp0)
        cp1_minus_cp0 /= distance

        car_start_point_mult = START_POINT_MULT[start_point_mult_index[0]]
        self.car = Car(
            x=self.checkpoints[0][0] + cp1_minus_cp0[1] * car_start_point_mult[0],
            y=self.checkpoints[0][1] + cp1_minus_cp0[0] * car_start_point_mult[1],
        )
        self.car.angle = self.car.get_angle(
            x=self.checkpoints[1][0],
            y=self.checkpoints[1][1],
        )
        self.cars.append(self.car)

        if self.opponent_model:
            opponent_start_point_mult = START_POINT_MULT[start_point_mult_index[1]]
            self.opponent_car = Car(
                x=self.checkpoints[0][0]
                + cp1_minus_cp0[1] * opponent_start_point_mult[0],
                y=self.checkpoints[0][1]
                + cp1_minus_cp0[0] * opponent_start_point_mult[1],
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

    def _apply_angle_thrust(self, angle: float, thrust: float) -> None:
        super()._apply_angle_thrust(angle=angle, thrust=thrust)

        if self.opponent_car:
            observation = self._get_opponent_obs()
            action, _ = self.opponent_model.predict(observation, deterministic=True)
            angle, thrust = self._convert_action_to_angle_thrust(action=action)
            self.opponent_car.rotate(angle=angle)
            self.opponent_car.thrust_towards_heading(thrust=thrust)

    def _get_collision_reward(self) -> SupportsFloat:
        return 0

    def _get_checkpoint_visit_reward(self, car_index: int) -> SupportsFloat:
        return 1 if car_index == 0 else 0

    def _move_car(self) -> SupportsFloat:
        if not self.opponent_car:
            return super()._move_car()

        reward = 0
        t = 0.0

        while t < 1.0:
            first_collision = None
            car_index = None

            car_collision = self.car.get_collision(
                self.opponent_car,
                radius=2 * self.car_radius,
            )
            if car_collision is not None and car_collision.time + t < 1.0:
                first_collision = car_collision

            for i, car in enumerate(self.cars):
                checkpoint_index = (car.current_checkpoint + 1) % len(self.checkpoints)
                next_checkpoint = Unit(
                    x=self.checkpoints[checkpoint_index][0],
                    y=self.checkpoints[checkpoint_index][1],
                )
                checkpoint_collision = car.get_collision(
                    next_checkpoint,
                    radius=self.checkpoint_radius,
                )

                if (
                    checkpoint_collision is not None
                    and checkpoint_collision.time + t < 1.0
                    and (
                        first_collision is None
                        or checkpoint_collision.time < first_collision.time
                    )
                ):
                    first_collision = checkpoint_collision
                    car_index = i

            if first_collision is None:
                for car in self.cars:
                    car.move(t=1.0 - t)
                break

            for car in self.cars:
                car.move(t=first_collision.time)

            if isinstance(first_collision.second_unit, Car):
                first_collision.first_unit.bounce(
                    first_collision.second_unit,
                    min_impulse=self.min_impulse,
                    min_radius=2 * self.car_radius,
                )
                reward += self._get_collision_reward()
            else:  # checkpoint collision
                first_collision.first_unit.current_checkpoint += 1
                reward += self._get_checkpoint_visit_reward(car_index=car_index)

            t += first_collision.time

        return reward

    def _load_car_img(self) -> None:
        super()._load_car_img()
        self.opponent_car_img = (
            self._load_img(
                filename=self.opponent_car_img_path,
                width=self.checkpoint_radius,
            )
            if self.opponent_car
            else None
        )

    def _draw_car(self, canvas: pygame.Surface) -> None:
        for car, car_img in zip(self.cars, (self.car_img, self.opponent_car_img)):
            pygame.draw.circle(
                surface=canvas,
                color="white",
                center=[car.x / SCALE_FACTOR, car.y / SCALE_FACTOR],
                radius=self.car_radius / SCALE_FACTOR,
                width=40 // SCALE_FACTOR,
            )
            canvas.blit(
                pygame.transform.rotate(car_img, angle=-car.angle - 90),
                (
                    car.x / SCALE_FACTOR - car_img.get_width() / 2,
                    car.y / SCALE_FACTOR - car_img.get_height() / 2,
                ),
            )


class MadPodRacingBlockerEnv(MadPodRacingEnv):
    def __init__(
        self,
        opponent_path: str | Path,
        render_mode: str | None = None,
        laps: int = 3,
        car_max_thrust: int = 200,
        test_id: int | None = None,
        sequential_maps: bool = False,
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            laps=laps,
            car_max_thrust=car_max_thrust,
            opponent_path=opponent_path,
            test_id=test_id,
            sequential_maps=sequential_maps,
        )

        # opponent runner observation, blocker car
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(16,),
            dtype=np.float64,
        )

    def _get_obs(self) -> ObsType:
        return self._get_blocker_obs(car_index=0)

    def _get_opponent_obs(self) -> ObsType:
        return self._get_runner_obs(car_index=1)

    def _get_collision_reward(self) -> SupportsFloat:
        return 0.1

    def _get_checkpoint_visit_reward(self, car_index: int) -> SupportsFloat:
        return -1 if car_index == 1 else 0


class MadPodRacingDiscreteEnv(MadPodRacingEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        laps: int = 3,
        car_max_thrust: int = 200,
        test_id: int | None = None,
        sequential_maps: bool = False,
        opponent_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            laps=laps,
            car_max_thrust=car_max_thrust,
            test_id=test_id,
            sequential_maps=sequential_maps,
            opponent_path=opponent_path,
        )

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
        action: ActType,
    ) -> tuple[float, float]:
        return self.actions[action]


class MadPodRacingBlockerDiscreteEnv(MadPodRacingBlockerEnv):
    def __init__(
        self,
        opponent_path: str | Path,
        render_mode: str | None = None,
        laps: int = 3,
        car_max_thrust: int = 200,
        test_id: int | None = None,
        sequential_maps: bool = False,
    ) -> None:
        super().__init__(
            opponent_path=opponent_path,
            render_mode=render_mode,
            laps=laps,
            car_max_thrust=car_max_thrust,
            test_id=test_id,
            sequential_maps=sequential_maps,
        )

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
        action: ActType,
    ) -> tuple[float, float]:
        return self.actions[action]
