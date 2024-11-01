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

from gymnasium_search_race.envs.car import Car

SCALE_FACTOR = 20
CHECKPOINT_COLOR = (52, 52, 52)
NEXT_CHECKPOINT_COLOR = (122, 176, 219)

FONT_NAME = "Monospace"
FONT_COLOR = (255, 255, 255)
FONT_SIZE = 400
ROOT_PATH = Path(__file__).resolve().parent
ASSETS_PATH = ROOT_PATH / "assets" / "search_race"
MAPS_PATH = ROOT_PATH / "maps"


class SearchRaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        test_id: int | None = None,
    ) -> None:
        self.laps = 3
        self.width = 16000
        self.height = 9000
        self.checkpoint_radius = 600
        self.max_rotation_per_turn = 18
        self.car_max_thrust = 200
        self.car_friction = 0.15

        self.car_thrust_upper_bound = 2000
        self.car_angle_upper_bound = 360

        # is last checkpoint, next checkpoint, checkpoint after next checkpoint
        # position, horizontal speed, vertical speed, angle
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, -1, -1, 0]),
            high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
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

        self.test_id = test_id

        self.window = None
        self.clock = None
        self.font = None
        self.background_img = None
        self.background_img_path = ASSETS_PATH / "back.jpg"
        self.car_img = None
        self.car_img_path = ASSETS_PATH / "car.png"

    def _get_obs(self) -> ObsType:
        next_checkpoint_index = (self.current_checkpoint + 1) % len(self.checkpoints)
        next_next_checkpoint_index = (next_checkpoint_index + 1) % len(self.checkpoints)
        return np.array(
            [
                float(self.current_checkpoint >= (self.total_checkpoints - 1)),
                self.checkpoints[next_checkpoint_index][0] / self.width,
                self.checkpoints[next_checkpoint_index][1] / self.height,
                self.checkpoints[next_next_checkpoint_index][0] / self.width,
                self.checkpoints[next_next_checkpoint_index][1] / self.height,
                self.car.x / self.width,
                self.car.y / self.height,
                self.car.vx / self.car_thrust_upper_bound,
                self.car.vy / self.car_thrust_upper_bound,
                self.car.angle / self.car_angle_upper_bound,
            ],
            dtype=np.float64,
        )

    def _get_info(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "max_rotation_per_turn": self.max_rotation_per_turn,
            "car_max_thrust": self.car_max_thrust,
            "car_thrust_upper_bound": self.car_thrust_upper_bound,
            "car_angle_upper_bound": self.car_angle_upper_bound,
            "checkpoints": self.checkpoints,
            "total_checkpoints": self.total_checkpoints,
            "current_checkpoint": self.current_checkpoint,
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

        if test_id is None:
            maps_paths = sorted(MAPS_PATH.glob("*.json"))
            test_map_path = self.np_random.choice(maps_paths)
        else:
            test_map_path = MAPS_PATH / f"test{test_id}.json"

        test_map = json.loads(test_map_path.read_text(encoding="UTF-8"))

        return np.array(
            [
                [int(i) for i in checkpoint.split()]
                for checkpoint in test_map["testIn"].split(";")
            ],
            dtype=np.float64,
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
        self.current_checkpoint = 0
        self.car = Car(
            x=self.checkpoints[0][0],
            y=self.checkpoints[0][1],
            angle=0,
        )
        self.car.angle = self.car.get_angle(
            x=self.checkpoints[1][0],
            y=self.checkpoints[1][1],
        )
        self._adjust_car()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _convert_action_to_angle_thrust(
        self,
        action: np.ndarray,
    ) -> tuple[float, float]:
        angle, thrust = action
        assert -1.0 <= angle <= 1.0
        assert 0.0 <= thrust <= 1.0

        angle = np.rint(angle * self.max_rotation_per_turn)
        thrust = np.rint(thrust * self.car_max_thrust)

        return angle, thrust

    def _get_next_checkpoint_index(self) -> int:
        return (self.current_checkpoint + 1) % len(self.checkpoints)

    def step(
        self,
        action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        angle, thrust = self._convert_action_to_angle_thrust(action=action)
        checkpoint_index = self._get_next_checkpoint_index()
        reward = -0.1

        self.car.rotate(angle=angle)
        self.car.thrust_towards_heading(thrust=thrust)
        self.car.move(t=1.0)
        if (
            self.car.distance(
                x=self.checkpoints[checkpoint_index][0],
                y=self.checkpoints[checkpoint_index][1],
            )
            <= self.checkpoint_radius
        ):
            self.current_checkpoint += 1
            reward = 1000 / self.total_checkpoints

        self._adjust_car()

        observation = self._get_obs()
        terminated = self.current_checkpoint >= self.total_checkpoints
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _load_background_img(self) -> pygame.Surface:
        background_img = pygame.image.load(self.background_img_path)
        return pygame.transform.scale_by(
            background_img,
            (self.width / SCALE_FACTOR) / background_img.get_width(),
        )

    def _load_car_img(self) -> pygame.Surface:
        car_img = pygame.image.load(self.car_img_path)
        return pygame.transform.scale_by(
            car_img,
            (self.checkpoint_radius / SCALE_FACTOR) / car_img.get_width(),
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
            f"{self._get_next_checkpoint_index()} ({self.current_checkpoint})",
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
            self.background_img = self._load_background_img()

        if self.car_img is None:
            self.car_img = self._load_car_img()

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
        test_id: int | None = None,
    ) -> None:
        super().__init__(render_mode=render_mode, test_id=test_id)

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
