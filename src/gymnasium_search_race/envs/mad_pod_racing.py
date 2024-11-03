from itertools import product
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.core import ObsType
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


class MadPodRacingEnv(SearchRaceEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        opponent_path: str | Path | None = None,
    ) -> None:
        super().__init__(render_mode=render_mode)
        self.car_max_thrust = 100
        self.car_thrust_upper_bound = 1000
        self.car_radius = 400
        self.min_impulse = 120.0

        self.background_img_path = ASSETS_PATH / "background.jpg"
        self.car_img_path = ASSETS_PATH / "space_ship_runner.png"
        self.opponent_car_img_path = ASSETS_PATH / "space_ship_blocker.png"

        self.opponent_model = PPO.load(opponent_path) if opponent_path else None

    def _get_car_obs(self, car: Car) -> ObsType:
        return np.array(
            [
                car.x / self.width,
                car.y / self.height,
                car.vx / self.car_thrust_upper_bound,
                car.vy / self.car_thrust_upper_bound,
                car.angle / self.car_angle_upper_bound,
            ],
            dtype=np.float64,
        )

    def _get_runner_obs(self, car_index: int) -> ObsType:
        car = self.cars[car_index]
        car_obs = self._get_car_obs(car=car)

        next_checkpoint_index = (car.current_checkpoint + 1) % len(self.checkpoints)
        next_next_checkpoint_index = (next_checkpoint_index + 1) % len(self.checkpoints)
        checkpoints_obs = np.array(
            [
                float(car.current_checkpoint >= (self.total_checkpoints - 1)),
                self.checkpoints[next_checkpoint_index][0] / self.width,
                self.checkpoints[next_checkpoint_index][1] / self.height,
                self.checkpoints[next_next_checkpoint_index][0] / self.width,
                self.checkpoints[next_next_checkpoint_index][1] / self.height,
            ],
            dtype=np.float64,
        )

        return np.concatenate((checkpoints_obs, car_obs))

    def _get_blocker_obs(self, car_index: int) -> ObsType:
        runner_obs = self._get_runner_obs(car_index=(car_index + 1) % len(self.cars))
        blocker_car = self._get_car_obs(car=self.cars[car_index])
        return np.concatenate((runner_obs, blocker_car))

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

    def _get_default_reward(self) -> SupportsFloat:
        return -0.1

    def _get_checkpoint_visit_reward(self, car_index: int) -> SupportsFloat:
        return 1000 / self.total_checkpoints if car_index == 0 else 0.0

    def _move_car(self) -> SupportsFloat:
        if not self.opponent_car:
            return super()._move_car()

        reward = self._get_default_reward()
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
    ) -> None:
        super().__init__(render_mode=render_mode, opponent_path=opponent_path)

        # opponent runner observation, blocker car
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, -1, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            dtype=np.float64,
        )

    def _get_obs(self) -> ObsType:
        return self._get_blocker_obs(car_index=0)

    def _get_opponent_obs(self) -> ObsType:
        return self._get_runner_obs(car_index=1)

    def _get_default_reward(self) -> SupportsFloat:
        return 0.1

    def _get_checkpoint_visit_reward(self, car_index: int) -> SupportsFloat:
        return -1000 / self.total_checkpoints if car_index == 1 else 0.0


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
