from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPSILON = 0.00001


@dataclass
class Point:
    x: float
    y: float

    def distance(self, point: Point) -> float:
        return float(np.linalg.norm([self.x - point.x, self.y - point.y]))


@dataclass
class Unit(Point):
    vx: float = 0.0
    vy: float = 0.0

    def get_collision(self, unit: Unit, radius: float) -> Collision | None:
        # https://github.com/Illedan/CGSearchRace/blob/master/SearchRace/src/main/java/com/codingame/game/Unit.java#L36
        # check instant collision
        if self.distance(unit) <= radius:
            return Collision(time=0, first_unit=self, second_unit=unit)

        # both units have same speed
        if self.vx == unit.vx and self.vy == unit.vy:
            return None

        # change referencial: unit is motionless at point (0, 0)
        dx = self.x - unit.x
        dy = self.y - unit.y
        dvx = self.vx - unit.vx
        dvy = self.vy - unit.vy

        a = dvx**2 + dvy**2

        if a <= 0.0:
            return None

        b = 2.0 * (dx * dvx + dy * dvy)
        c = dx**2 + dy**2 - radius**2
        delta = b**2 - 4.0 * a * c

        if delta < 0.0:
            return None

        t = (-b - np.sqrt(delta)) / (2.0 * a)

        if t <= 0.0:
            return None

        return Collision(time=t, first_unit=self, second_unit=unit)

    def bounce(self, unit: Unit, min_impulse: float, min_radius: float) -> None:
        # https://github.com/SpiritusSancti5/codinGame/blob/master/Referees/Coders%20Strike%20Back/Referee.java#L476
        # https://github.com/robostac/coders-strike-back-referee/blob/master/csbref.go#L132
        normal = np.array([unit.x - self.x, unit.y - self.y])
        distance = np.linalg.norm(normal)
        normal /= distance

        relative_velocity = np.array([self.vx - unit.vx, self.vy - unit.vy])

        self_mass = unit_mass = 1
        force = np.dot(normal, relative_velocity) / (1 / self_mass + 1 / unit_mass)
        force += min_impulse if force < min_impulse else force

        impulse = -normal * force
        self.vx += impulse[0] * (1 / self_mass)
        self.vy += impulse[1] * (1 / self_mass)
        unit.vx -= impulse[0] * (1 / unit_mass)
        unit.vy -= impulse[1] * (1 / unit_mass)

        if distance <= min_radius:
            distance -= min_radius
            self.x += normal[0] * -(-distance / 2 + EPSILON)
            self.y += normal[1] * -(-distance / 2 + EPSILON)
            unit.x += normal[0] * (-distance / 2 + EPSILON)
            unit.y += normal[1] * (-distance / 2 + EPSILON)


@dataclass
class Collision:
    time: float
    first_unit: Unit
    second_unit: Unit


@dataclass
class Car(Unit):
    angle: float = 0.0  # in degrees
    current_checkpoint: int = 0

    def radians(self) -> float:
        return np.radians(self.angle)

    def get_radians(self, x: float, y: float) -> float:
        return np.arctan2(y - self.y, x - self.x)

    def get_angle(self, x: float, y: float) -> float:
        return np.degrees(self.get_radians(x, y)) % 360

    def rotate(self, angle: float) -> None:
        self.angle = (self.angle + angle) % 360

    def thrust_towards_heading(self, thrust: float) -> None:
        radians = self.radians()
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
