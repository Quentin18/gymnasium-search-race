import pytest

from gymnasium_search_race.envs.car import Car


@pytest.mark.parametrize(
    "car,x,y,expected",
    (
        (
            Car(x=10353, y=1986, angle=161),
            2757,
            4659,
            161,
        ),
    ),
)
def test_get_angle(car: Car, x: float, y: float, expected: float):
    actual = car.get_angle(x=x, y=y)
    assert round(actual) == expected
