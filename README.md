# Gymnasium Search Race

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

Gymnasium environment for
the [Search Race CodinGame optimization puzzle](https://www.codingame.com/multiplayer/optimization/search-race).

https://github.com/user-attachments/assets/1862b04b-9e33-4f55-a309-ad665a1db2f1

<table>
    <tbody>
        <tr>
            <td>Action Space</td>
            <td><code>Box([-1, 0], [1, 1], float64)</code></td>
        </tr>
        <tr>
            <td>Observation Space</td>
            <td><code>Box([0, 0, 0, 0, 0, 0, -1, -1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1], float64)</code></td>
        </tr>
        <tr>
            <td>import</td>
            <td><code>gymnasium.make("gymnasium_search_race:gymnasium_search_race/SearchRace-v0")</code></td>
        </tr>
    </tbody>
</table>

## Installation

To install `gymnasium-search-race` with pip, execute:

```bash
pip install gymnasium_search_race
```

From source:

```bash
git clone https://github.com/Quentin18/gymnasium-search-race
cd gymnasium-search-race/
pip install -e .
```

## Environment

### Action Space

The action is a `ndarray` with 2 continuous variables:

- The rotation angle between -18 and 18 degrees, normalized between -1 and 1.
- The thrust between 0 and 200, normalized between 0 and 1.

### Observation Space

The observation is a `ndarray` of 9 continuous variables:

- The x and y coordinates of the next checkpoint.
- The x and y coordinates of the checkpoint after next checkpoint.
- The x and y coordinates of the car.
- The horizontal speed vx and vertical speed vy of the car.
- The facing angle of the car.

The values are normalized between 0 and 1, or -1 and 1 if negative values are allowed.

### Reward

The goal is to visit all checkpoints as quickly as possible, as such the agent is penalised with a reward of `-0.1` for
each timestep.
When a checkpoint is visited, the agent is awarded with a reward of `1000/total_checkpoints`.

### Starting State

The starting state is generated by choosing a random CodinGame test case.

### Episode End

The episode ends if either of the following happens:

1. Termination: The car visit all checkpoints before the time is out.
2. Truncation: Episode length is greater than 600.

### Arguments

- `test_id`: test case id to generate the checkpoints (see
  choices [here](https://github.com/Quentin18/gymnasium-search-race/tree/main/src/gymnasium_search_race/envs/maps)). The
  default value is `None` which selects a test case randomly when the `reset` method is called.

```python
import gymnasium as gym

gym.make("gymnasium_search_race:gymnasium_search_race/SearchRace-v0", test_id=1)
```

## Usage

You can use [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) to train and evaluate agents:

```bash
pip install rl_zoo3
```

### Train an Agent

The hyperparameters are defined in `hyperparams/ppo.yml`.

To train a PPO agent for the Search Race game, execute:

```bash
python -m rl_zoo3.train \
  --algo ppo \
  --env gymnasium_search_race/SearchRace-v0 \
  --tensorboard-log logs \
  --eval-freq 10000 \
  --eval-episodes 10 \
  --gym-packages gymnasium_search_race \
  --conf-file hyperparams/ppo.yml \
  --progress
```

### Enjoy a Trained Agent

To see a trained agent in action on random test cases, execute:

```bash
python -m rl_zoo3.enjoy \
  --algo ppo \
  --env gymnasium_search_race/SearchRace-v0 \
  --n-timesteps 10000 \
  --deterministic \
  --gym-packages gymnasium_search_race \
  --load-best \
  --progress
```

### Run Test Cases

To run test cases with a trained agent, execute:

```bash
python -m scripts.run_test_cases \
  --path rl-trained-agents/ppo/best_model.zip \
  --record-video \
  --record-metrics
```

## Tests

To run tests, execute:

```bash
pytest
```

## Citing

To cite the repository in publications:

```bibtex
@misc{gymnasium-search-race,
  author = {Quentin Deschamps},
  title = {Gymnasium Search Race},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Quentin18/gymnasium-search-race}},
}
```

## References

- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [CGSearchRace](https://github.com/Illedan/CGSearchRace)

## Author

[Quentin Deschamps](mailto:quentindeschamps18@gmail.com)
