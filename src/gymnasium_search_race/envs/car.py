from dataclasses import dataclass

import numpy as np


@dataclass
class Car:
    x: float
    y: float
    angle: float  # in degrees
    vx: float = 0.0
    vy: float = 0.0

    def get_angle(self, x: float, y: float) -> float:
        return np.rad2deg(np.atan2(y - self.y, x - self.x)) % 360

    def rotate(self, angle: float) -> None:
        self.angle = (self.angle + angle) % 360

    def thrust_towards_heading(self, thrust: float) -> None:
        radians = np.radians(self.angle)
        self.vx += np.cos(radians) * thrust
        self.vy += np.sin(radians) * thrust

    def move(self, t: float) -> None:
        self.x += self.vx * t
        self.y += self.vy * t

    def truncate_position(self) -> None:
        self.x = np.trunc(self.x)
        self.y = np.trunc(self.y)

    def round_position(self) -> None:
        self.x = np.rint(self.x)
        self.y = np.rint(self.y)

    def round_angle(self) -> None:
        self.angle = np.rint(self.angle)

    def truncate_speed(self, friction: float) -> None:
        self.vx = np.trunc(self.vx * (1 - friction))
        self.vy = np.trunc(self.vy * (1 - friction))

    def distance(self, x: float, y: float) -> float:
        return float(np.linalg.norm([self.x - x, self.y - y]))
