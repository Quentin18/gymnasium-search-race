import json
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame

from gymnasium_search_race.envs.models import Car, Point

SCALE_FACTOR = 20
CHECKPOINT_COLOR = (52, 52, 52)
NEXT_CHECKPOINT_COLOR = (122, 176, 219)

FONT_NAME = "Monospace"
FONT_COLOR = (255, 255, 255)
FONT_SIZE = 400
ROOT_PATH = Path(__file__).resolve().parent
ASSETS_PATH = ROOT_PATH / "assets" / "search_race"
MAPS_PATH = ROOT_PATH / "maps"


def get_test_ids() -> list[int]:
    return sorted(int(path.stem.replace("test", "")) for path in MAPS_PATH.iterdir())


class SearchRaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        laps: int = 3,
        car_max_thrust: float = 200,
        test_id: int | None = None,
        sequential_maps: bool = False,
    ) -> None:
        self.laps = laps
        self.car_max_thrust = car_max_thrust
        self.width = 16000
        self.height = 9000
        self.checkpoint_radius = 600
        self.max_rotation_per_turn = 18
        self.car_friction = 0.15

        self.distance_upper_bound = np.linalg.norm([self.width, self.height])
        self.car_thrust_upper_bound = car_max_thrust * 10

        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(8,),
            dtype=np.float64,
        )

        # rotation angle, thrust
        self.action_space = spaces.Box(
            low=np.array([-1, 0]),
            high=np.array([1, 1]),
            dtype=np.float64,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.test_ids = self._get_test_ids()
        self.test_checkpoints = self._get_test_checkpoints()
        self.test_id = test_id
        self.sequential_maps = sequential_maps
        self.test_index = -1

        self.window = None
        self.clock = None
        self.font = None
        self.background_img = None
        self.background_img_path = ASSETS_PATH / "back.jpg"
        self.car_img = None
        self.car_img_path = ASSETS_PATH / "car.png"

    def _get_test_ids(self) -> list[int]:
        return get_test_ids()

    def _get_test_checkpoints(self) -> list[np.ndarray]:
        checkpoints = []

        for test_id in self._get_test_ids():
            test_map_path = MAPS_PATH / f"test{test_id}.json"
            test_map = json.loads(test_map_path.read_text(encoding="UTF-8"))
            checkpoints.append(
                np.array(
                    [
                        [int(i) for i in checkpoint.split()]
                        for checkpoint in test_map["testIn"].split(";")
                    ],
                    dtype=np.float64,
                )
            )

        return checkpoints

    def _get_diff_obs(self, car: Car, x: float, y: float) -> ObsType:
        return np.array(
            [
                (x - car.x) / self.distance_upper_bound,
                (y - car.y) / self.distance_upper_bound,
                ((car.get_radians(x, y) - car.radians() + np.pi) % (2 * np.pi) - np.pi)
                / np.pi,
            ],
            dtype=np.float64,
        )

    def _get_speed_obs(self, car: Car) -> ObsType:
        return np.array(
            [
                car.vx / self.car_thrust_upper_bound,
                car.vy / self.car_thrust_upper_bound,
            ],
            dtype=np.float64,
        )

    def _get_obs(self) -> ObsType:
        obs = []

        # position and angle of the next 2 checkpoints relative to the car
        for i in range(2):
            x_cp, y_cp = self.checkpoints[
                (self.car.current_checkpoint + i + 1) % len(self.checkpoints)
            ]
            obs.append(self._get_diff_obs(car=self.car, x=x_cp, y=y_cp))

        # car speed
        obs.append(self._get_speed_obs(car=self.car))

        return np.concatenate(obs)

    def _get_terminated(self) -> bool:
        return self.car.current_checkpoint >= self.total_checkpoints

    def _get_info(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "x": self.car.x,
            "y": self.car.y,
            "vx": self.car.vx,
            "vy": self.car.vy,
            "angle": self.car.angle,
            "max_rotation_per_turn": self.max_rotation_per_turn,
            "car_max_thrust": self.car_max_thrust,
            "distance_upper_bound": self.distance_upper_bound,
            "car_thrust_upper_bound": self.car_thrust_upper_bound,
            "checkpoints": self.checkpoints,
            "total_checkpoints": self.total_checkpoints,
            "current_checkpoint": self.car.current_checkpoint,
        }

    def _generate_checkpoints(
        self,
        options: dict[str, Any] | None = None,
    ) -> np.ndarray:
        test_id = (
            self.test_id
            if options is None or "test_id" not in options
            else options["test_id"]
        )

        if test_id is not None:
            self.test_index = self.test_ids.index(test_id)
        elif self.sequential_maps:
            self.test_index = (self.test_index + 1) % len(self.test_ids)
        else:
            self.test_index = self.np_random.choice(len(self.test_ids))

        return self.test_checkpoints[self.test_index]

    def _generate_car(self) -> None:
        self.car = Car(
            x=self.checkpoints[0][0],
            y=self.checkpoints[0][1],
        )
        self.car.angle = self.car.get_angle(
            x=self.checkpoints[1][0],
            y=self.checkpoints[1][1],
        )

    def _adjust_car(self) -> None:
        self.car.truncate_position()
        self.car.round_angle()
        self.car.truncate_speed(friction=self.car_friction)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.checkpoints = self._generate_checkpoints(options=options)
        self.total_checkpoints = len(self.checkpoints) * self.laps
        self._generate_car()
        self._adjust_car()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _convert_action_to_angle_thrust(
        self,
        action: ActType,
    ) -> tuple[float, float]:
        angle, thrust = action
        assert -1.0 <= angle <= 1.0
        assert 0.0 <= thrust <= 1.0

        angle = np.rint(angle * self.max_rotation_per_turn)
        thrust = np.rint(thrust * self.car_max_thrust)

        return angle, thrust

    def _apply_angle_thrust(self, angle: float, thrust: float) -> None:
        self.car.rotate(angle=angle)
        self.car.thrust_towards_heading(thrust=thrust)

    def _get_next_checkpoint_index(self) -> int:
        return (self.car.current_checkpoint + 1) % len(self.checkpoints)

    def _move_car(self) -> SupportsFloat:
        reward = 0
        checkpoint_index = self._get_next_checkpoint_index()

        self.car.move(t=1.0)
        if (
            self.car.distance(
                Point(
                    x=self.checkpoints[checkpoint_index][0],
                    y=self.checkpoints[checkpoint_index][1],
                )
            )
            <= self.checkpoint_radius
        ):
            self.car.current_checkpoint += 1
            reward += 1

        return reward

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        angle, thrust = self._convert_action_to_angle_thrust(action=action)

        self._apply_angle_thrust(angle=angle, thrust=thrust)
        reward = self._move_car()
        self._adjust_car()

        observation = self._get_obs()
        terminated = self._get_terminated()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    @staticmethod
    def _load_img(filename: str | Path, width: int) -> pygame.Surface:
        img = pygame.image.load(filename)
        return pygame.transform.scale_by(img, (width / SCALE_FACTOR) / img.get_width())

    def _load_background_img(self) -> None:
        self.background_img = self._load_img(
            filename=self.background_img_path,
            width=self.width,
        )

    def _load_car_img(self) -> None:
        self.car_img = self._load_img(
            filename=self.car_img_path,
            width=self.checkpoint_radius,
        )

    def _draw_checkpoints(self, canvas: pygame.Surface) -> None:
        next_checkpoint_index = self._get_next_checkpoint_index()

        for i, checkpoint in enumerate(self.checkpoints):
            center = (checkpoint / SCALE_FACTOR).tolist()
            pygame.draw.circle(
                surface=canvas,
                color=(
                    NEXT_CHECKPOINT_COLOR
                    if i == next_checkpoint_index
                    else CHECKPOINT_COLOR
                ),
                center=center,
                radius=self.checkpoint_radius / SCALE_FACTOR,
                width=40 // SCALE_FACTOR,
            )
            text_surface = self.font.render(str(i), True, FONT_COLOR)
            canvas.blit(
                source=text_surface,
                dest=(
                    center[0] - text_surface.get_width() / 2,
                    center[1] - text_surface.get_height() / 2,
                ),
            )

    def _draw_car(self, canvas: pygame.Surface) -> None:
        canvas.blit(
            pygame.transform.rotate(self.car_img, angle=-self.car.angle - 90),
            (
                self.car.x / SCALE_FACTOR - self.car_img.get_width() / 2,
                self.car.y / SCALE_FACTOR - self.car_img.get_height() / 2,
            ),
        )

    def _draw_car_text(self, canvas: pygame.Surface) -> None:
        for i, (name, value) in enumerate(asdict(self.car).items()):
            if name == "current_checkpoint":
                continue

            text_surface = self.font.render(
                f"{name:<6} {value:0.0f}",
                True,
                FONT_COLOR,
            )
            canvas.blit(
                source=text_surface,
                dest=(
                    canvas.get_width() * 0.01,
                    canvas.get_height() * 0.01 + i * self.font.get_height(),
                ),
            )

    def _draw_checkpoint_text(self, canvas: pygame.Surface) -> None:
        text_surface = self.font.render(
            f"{self._get_next_checkpoint_index()} ({self.car.current_checkpoint})",
            True,
            FONT_COLOR,
        )
        canvas.blit(
            source=text_surface,
            dest=(
                canvas.get_width() * 0.99 - text_surface.get_width(),
                canvas.get_height() * 0.01,
            ),
        )

    def _render_frame(self) -> RenderFrame | list[RenderFrame]:
        window_size = self.width / SCALE_FACTOR, self.height / SCALE_FACTOR

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(window_size)
            pygame.display.set_caption("Search Race")

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont(
                FONT_NAME,
                FONT_SIZE // SCALE_FACTOR,
                bold=True,
            )

        if self.background_img is None:
            self._load_background_img()

        if self.car_img is None:
            self._load_car_img()

        canvas = pygame.Surface(window_size)
        canvas.blit(self.background_img, (0, 0))

        self._draw_checkpoints(canvas=canvas)
        self._draw_car(canvas=canvas)
        self._draw_car_text(canvas=canvas)
        self._draw_checkpoint_text(canvas=canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)),
                axes=(1, 0, 2),
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class SearchRaceDiscreteEnv(SearchRaceEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        laps: int = 3,
        car_max_thrust: float = 200,
        test_id: int | None = None,
        sequential_maps: bool = False,
    ) -> None:
        super().__init__(
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
